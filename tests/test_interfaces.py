"""Shared contract checks and real local HTTP/MCP transport integration."""
import asyncio
import http.client
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from ai_lab_core import tool_api, http_api

PROJECT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_closed_schemas_and_invalid_inputs(self):
        self.assertEqual(len(tool_api.schemas()), 22)
        for spec in tool_api.schemas():
            self.assertFalse(spec['parameters']['additionalProperties'])
        for name, args in [
            ('status', {'command': 'anything'}), ('list_local_models', {'limit': True}),
            ('generate', {'connection_id': 'test', 'model': 'test', 'prompt': 'test', 'max_tokens': 2049}),
            ('run', {'operation': 'test', 'parameters': [], 'extra': 1}),
            ('artifact_inspect', {'path': '/tmp/test', 'hash_files': 1}),
            ('job', {'action': 'cancel'}), ('job', {'action': 'list', 'limit': 201}),
        ]:
            with self.subTest(name=name):
                self.assertFalse(tool_api.dispatch('ai_lab_' + name, args)['ok'])
        self.assertEqual(tool_api.dispatch('missing', {})['error']['code'], 'unknown_tool')
        self.assertFalse(tool_api.dispatch('ai_lab_plan', {'operation': 'test', 'parameters': {'large': 'x' * tool_api.MAX_JSON_BYTES}})['ok'])
        self.assertFalse(tool_api.dispatch('ai_lab_plan', {'operation': 'test', 'parameters': {'nan': float('nan')}})['ok'])
        recursive = {}; recursive['self'] = recursive
        self.assertFalse(tool_api.dispatch('ai_lab_plan', {'operation': 'test', 'parameters': recursive})['ok'])

    def test_service_exceptions_and_oversized_results_are_safe(self):
        with patch('ai_lab_core.service.status', side_effect=RuntimeError('private-value')):
            result = tool_api.dispatch('ai_lab_status', {})
            self.assertNotIn('private-value', tool_api.encode(result))
            self.assertEqual(result['error']['code'], 'service_error')
        with patch('ai_lab_core.service.status', return_value={'ok': True, 'large': 'x' * tool_api.MAX_JSON_BYTES}):
            self.assertEqual(tool_api.dispatch('ai_lab_status', {})['error']['code'], 'response_too_large')

    def test_cli_uses_same_contract_and_exact_parameter_file(self):
        env = dict(os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            env['AI_LAB_INTEGRATIONS'] = str(Path(tmp) / 'not-present.json')
            env['AI_LAB_CONFIG'] = str(Path(tmp) / 'not-present-config.json')
            params = Path(tmp) / 'parameters.json'; params.write_text('{}')
            for argv in (['operations'], ['call', 'ai_lab_operations'], ['plan', 'missing', '--params-file', str(params)]):
                result = subprocess.run([sys.executable, '-m', 'ai_lab_core'] + argv,
                                        cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=15)
                body = json.loads(result.stdout)
                self.assertNotEqual(body.get('error', {}).get('code'), 'service_error')
                if argv[0] != 'plan':
                    self.assertTrue(body['ok'])
                    self.assertEqual(result.returncode, 0)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ai-lab-api-test-')
        self.token_file = Path(self.tmp.name) / 'api.token'
        self.token = 'synthetic-api-key-with-more-than-32-chars'
        with patch.object(http_api, 'resolve_secret', return_value=self.token):
            self.server = http_api.create_server(0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None, auth=False):
        supplied = dict(headers or {})
        if auth:
            supplied['Authorization'] = 'Bearer ' + self.token
        if isinstance(body, dict):
            body = json.dumps(body)
            supplied.setdefault('Content-Type', 'application/json')
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=supplied)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            conn.close()

    def test_auth_health_and_real_operations(self):
        status, headers, body = self.request('GET', '/health')
        self.assertEqual(status, 200); self.assertEqual(body['api_version'], '1.1.0')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertNotIn(self.token, json.dumps(body))
        self.assertFalse(self.token_file.exists())
        self.assertEqual(self.request('GET', '/tools')[0], 401)
        self.assertEqual(len(self.request('GET', '/tools', auth=True)[2]['tools']), 22)
        with patch.dict(os.environ, {'AI_LAB_INTEGRATIONS': str(Path(self.tmp.name) / 'absent'), 'AI_LAB_CONFIG': str(Path(self.tmp.name) / 'absent-config')}):
            status, _, result = self.request('POST', '/tools/call', {'name': 'ai_lab_operations', 'arguments': {}}, auth=True)
        self.assertEqual(status, 200); self.assertTrue(result['ok'])

    def test_origin_host_body_and_transfer_guards(self):
        self.assertEqual(self.request('GET', '/health', headers={'Origin': 'http://localhost'})[0], 403)
        self.assertEqual(self.request('GET', '/health', headers={'Host': 'attacker.example'})[0], 403)
        self.assertEqual(self.request('GET', '/tools?token=unused', auth=True)[0], 400)
        self.assertEqual(self.request('POST', '/tools/call', '{}', {'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked'}, auth=True)[0], 400)
        self.assertEqual(self.request('POST', '/tools/call', {'name': 'ai_lab_operations', 'arguments': {}, 'extra': 1}, auth=True)[0], 400)
        # The server rejects the declared size before reading any body. Sending
        # a payload here races its intentional early connection close.
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            with patch('ai_lab_core.http_api.tool_api.dispatch') as dispatch:
                conn.putrequest('POST', '/tools/call')
                conn.putheader('Authorization', 'Bearer ' + self.token)
                conn.putheader('Content-Type', 'application/json')
                conn.putheader('Content-Length', str(tool_api.MAX_JSON_BYTES + 1))
                conn.endheaders()
                response = conn.getresponse()
                self.assertEqual(response.status, 413)
                self.assertEqual(json.loads(response.read())['error']['code'], 'request_too_large')
                dispatch.assert_not_called()
        finally:
            conn.close()


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional official MCP SDK is unavailable')
class MCPTests(unittest.TestCase):
    def test_real_stdio_handshake_list_and_call(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        async def exercise():
            with tempfile.TemporaryDirectory(prefix='ai-lab-mcp-test-') as tmp:
                env = {'PATH': os.environ.get('PATH', ''), 'HOME': tmp,
                       'AI_LAB_CONFIG': str(Path(tmp) / 'absent-config'),
                       'AI_LAB_INTEGRATIONS': str(Path(tmp) / 'absent-integrations')}
                params = StdioServerParameters(command=sys.executable, args=['-m', 'ai_lab_core', 'mcp'],
                                               cwd=str(PROJECT), env=env)
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        listing = await session.list_tools()
                        self.assertEqual({tool.name for tool in listing.tools}, {tool['name'] for tool in tool_api.schemas()})
                        result = await session.call_tool('ai_lab_operations', {})
                        self.assertFalse(result.isError)
                        self.assertTrue(json.loads(result.content[0].text)['ok'])
                        invalid = await session.call_tool('ai_lab_list_local_models', {'limit': True})
                        self.assertTrue(invalid.isError)
                        self.assertEqual(json.loads(invalid.content[0].text)['error']['code'], 'invalid_arguments')
        asyncio.run(asyncio.wait_for(exercise(), timeout=20))


if __name__ == '__main__':
    unittest.main()
