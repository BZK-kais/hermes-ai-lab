"""Exercise the new release tools across their real local transport adapters."""

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

from ai_lab_core import http_api, onboarding, service, tool_api


PROJECT = Path(__file__).resolve().parents[1]
CHECK = ('ai_lab_onboarding', {'request': {'action': 'check'}})
MISSION = ('ai_lab_mission', {'request': {'action': 'prepare'}})
STATUS = ('ai_lab_registry', {'request': {'action': 'status'}})
OPERATOR_ONLY = {
    'model_root': '/test/model-root',
    'data_root': '/test/data-root',
    'mlx_python': '/test/python',
}
REJECTED = tuple((CHECK[0], {'request': {'action': 'setup', key: value}})
                 for key, value in OPERATOR_ONLY.items())
CASES = (CHECK, MISSION, STATUS, *REJECTED)


class ReleaseContractTests(unittest.TestCase):
    def test_operator_options_rejected_before_dispatch(self):
        with patch.object(onboarding, 'dispatch', wraps=onboarding.dispatch) as setup:
            for name, arguments in REJECTED:
                with self.subTest(option=next(key for key in arguments['request'] if key != 'action')):
                    self.assertEqual(tool_api.dispatch(name, arguments)['error']['code'], 'invalid_arguments')
                    self.assertEqual(service.onboarding(**arguments)['error']['code'], 'invalid_arguments')
            setup.assert_not_called()


class ReleaseTransportCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ai-lab-release-interfaces-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.env = {
            'HOME': str(self.root),
            'HERMES_HOME': str(self.root / 'profile'),
            'AI_LAB_CONFIG': str(self.root / 'missing-connections.json'),
            'AI_LAB_INTEGRATIONS': str(self.root / 'missing-integrations.json'),
            'AI_LAB_STATE_DIR': str(self.root / 'state'),
        }

    def assert_result(self, case, result, expected):
        self.assertEqual(result, expected)
        if case == CHECK:
            self.assertTrue(result['ok'])
            self.assertEqual(result['smoke_dataset']['splits'], 2)
            self.assertFalse(result['network_probed'])
            self.assertFalse(result['workspace']['ready'])
        elif case == MISSION:
            self.assertTrue(result['ok'])
            self.assertEqual(result['status'], 'needs_context')
            self.assertIn('dataset_dir', result['blockers'])
        elif case == STATUS:
            self.assertTrue(result['ok'])
            self.assertIsNone(result['active_entry_id'])
            self.assertFalse(result['rollback_available'])
        else:
            self.assertFalse(result['ok'])
            self.assertEqual(result['error']['code'], 'invalid_arguments')


class ReleaseHTTPTests(ReleaseTransportCase):
    def test_shared_schemas_and_tool_calls_over_loopback(self):
        token_file = self.root / 'api.token'
        token = 'synthetic-release-api-key-with-more-than-32-chars'
        with patch.dict(os.environ, self.env), patch.object(http_api, 'resolve_secret', return_value=token), http_api.create_server(0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            self.assertFalse(token_file.exists())
            headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
            port = server.server_address[1]

            def request(method, route, payload=None):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                try:
                    body = json.dumps(payload) if payload is not None else None
                    connection.request(method, route, body=body, headers=headers)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    return json.loads(response.read())
                finally:
                    connection.close()

            try:
                self.assertEqual(request('GET', '/tools')['tools'], tool_api.schemas())
                for name, arguments in CASES:
                    with self.subTest(name=name, arguments=arguments):
                        expected = tool_api.dispatch(name, arguments)
                        result = request('POST', '/tools/call', {'name': name, 'arguments': arguments})
                        self.assert_result((name, arguments), result, expected)
            finally:
                server.shutdown()
                worker.join(timeout=2)


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'optional official MCP SDK is unavailable')
class ReleaseMCPTests(ReleaseTransportCase):
    def test_shared_schemas_and_tool_calls_over_stdio(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def exercise():
            params = StdioServerParameters(command=sys.executable, args=['-m', 'ai_lab_core', 'mcp'],
                                           cwd=str(PROJECT), env=self.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    actual = {tool.name: tool.inputSchema for tool in listing.tools}
                    expected = {item['name']: item['parameters'] for item in tool_api.schemas()}
                    self.assertEqual(actual, expected)
                    for name, arguments in CASES:
                        with self.subTest(name=name, arguments=arguments):
                            expected = tool_api.dispatch(name, arguments)
                            response = await session.call_tool(name, arguments)
                            self.assertEqual(response.isError, not expected['ok'])
                            self.assertEqual(len(response.content), 1)
                            result = json.loads(response.content[0].text)
                            self.assert_result((name, arguments), result, expected)

        with patch.dict(os.environ, self.env):
            asyncio.run(asyncio.wait_for(exercise(), timeout=20))


if __name__ == '__main__':
    unittest.main()
