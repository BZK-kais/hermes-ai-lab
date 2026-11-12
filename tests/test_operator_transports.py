"""Verify operator-only denials over the authenticated HTTP and MCP interfaces."""

import asyncio
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from ai_lab_core import http_api
from ai_lab_core.operator_policy import operator_scope


ROOT = Path(__file__).resolve().parents[1]


class TransportFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        artifact = root / 'artifact.bin'
        artifact.write_bytes(b'local fixture')
        integrations = root / 'integrations.json'
        self.state = root / 'state'
        integrations.write_text(json.dumps({
            'workspace_root': str(root),
            'state_dir': str(self.state),
            'integrations': [
                {'id': 'gpu', 'type': 'runpod', 'credential_ref': 'hermes:AI_LAB_RUNPOD_API_KEY'},
                {'id': 'upstream', 'type': 'mcp', 'base_url': 'http://127.0.0.1:1'},
                {'id': 'archive', 'type': 's3', 'credential_ref': 'hermes:AI_LAB_S3_ACCESS_KEY_ID',
                 'region': 'eu-west-3', 'bucket': 'test-bucket'},
            ],
        }), encoding='utf-8')
        self.env = {'HOME': str(root), 'HERMES_HOME': str(root / 'profile'),
                    'AI_LAB_INTEGRATIONS': str(integrations),
                    'AI_LAB_CONFIG': str(root / 'missing-connections.json')}
        self.cases = (
            ('ai_lab_provider_action', {'connection_id': 'gpu', 'action': 'stop',
                                        'parameters': {'resource_id': 'test'}}),
            ('ai_lab_provider_action', {'connection_id': 'upstream', 'action': 'list', 'parameters': {}}),
            ('ai_lab_run', {'operation': 'storage.upload', 'parameters': {
                'connection_id': 'archive', 'local_path': str(artifact), 'key': artifact.name}}),
        )

    def assert_denied(self, result):
        self.assertFalse(result['ok'], result)
        self.assertEqual(result['error']['code'], 'operator_required')
        self.assertFalse(self.state.exists())


class HTTPOperatorTests(TransportFixtures):
    def test_authenticated_requests_cannot_gain_operator_scope(self):
        token = 'synthetic-http-key-with-more-than-32-chars'
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
        with patch.dict(os.environ, self.env), patch.object(http_api, 'resolve_secret', return_value=token), \
             operator_scope('provider-action'), http_api.create_server(0) as server:
            # The HTTP server copies its creator's context into each worker.
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for name, args in self.cases:
                    with self.subTest(name=name, args=args):
                        conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
                        try:
                            body = json.dumps({'name': name, 'arguments': args})
                            conn.request('POST', '/tools/call', body, headers)
                            response = conn.getresponse()
                            self.assertEqual(response.status, 200)
                            self.assert_denied(json.loads(response.read()))
                        finally:
                            conn.close()
            finally:
                server.shutdown()
                worker.join(timeout=2)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional official MCP SDK is unavailable')
class MCPOperatorTests(TransportFixtures):
    def test_stdio_calls_cannot_gain_operator_scope(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def exercise():
            params = StdioServerParameters(command=sys.executable, args=['-m', 'ai_lab_core', 'mcp'],
                                           cwd=str(ROOT), env=self.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    for name, args in self.cases:
                        with self.subTest(name=name, args=args):
                            response = await session.call_tool(name, args)
                            self.assertTrue(response.isError)
                            self.assert_denied(json.loads(response.content[0].text))

        asyncio.run(asyncio.wait_for(exercise(), timeout=20))
