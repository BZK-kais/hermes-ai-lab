"""Remote control-plane restrictions through the shared and named CLI entrypoints."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from contextvars import Context, copy_context
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_lab_core import __main__ as cli
from ai_lab_core import integration_config, jobs, mcp_client, providers, tool_api, transfers
from ai_lab_core.operator_policy import operator_scope


PROVIDER = 'ai_lab_provider_action'
RUN = 'ai_lab_run'
DENIED = 'operator_required'
ROOT = Path(__file__).resolve().parents[1]


def provider_args(kind, action, **params):
    return {'connection_id': kind, 'action': action, 'parameters': params}


class SharedPolicyTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.file = self.root / 'artifact.bin'
        self.file.write_bytes(b'local fixture')
        connections = {
            'runpod': {'credential_ref': 'hermes:AI_LAB_RUNPOD_API_KEY'},
            'hf_jobs': {'credential_ref': 'hermes:AI_LAB_HF_TOKEN', 'project': 'test'},
            'hf_endpoints': {'credential_ref': 'hermes:AI_LAB_HF_TOKEN', 'project': 'test'},
            'hf_storage': {'credential_ref': 'hermes:AI_LAB_HF_TOKEN', 'repo_id': 'test/model'},
            'gcs': {'project': 'test', 'bucket': 'test-bucket'},
            's3': {'credential_ref': 'hermes:AI_LAB_S3_ACCESS_KEY_ID', 'bucket': 'test-bucket', 'region': 'eu-west-3'},
            'ssh': {'host': 'example.invalid'},
            'mcp': {'base_url': 'https://example.invalid'},
            'future': {},
        }
        settings = {'workspace_root': str(self.root), 'state_dir': str(self.root / 'state'),
                    'endpoints': [], 'integrations': [
                        {'id': kind, 'type': kind, 'workspace': str(self.root), **fields}
                        for kind, fields in connections.items()
                    ]}
        stack.enter_context(patch.object(integration_config, 'load_integrations', return_value=settings))

    def assert_code(self, result, code):
        self.assertFalse(result['ok'], result)
        self.assertEqual(result['error']['code'], code)

    def transfer_args(self, kind, operation='storage.upload', path=None):
        return {'operation': operation, 'parameters': {
            'connection_id': kind, 'key': 'artifact.bin', 'local_path': str(path or self.file)}}

    def cli_call(self, *argv):
        with patch('sys.argv', ['ailab', *argv]), redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as exit_status:
                cli.main()
        return exit_status.exception.code, json.loads(output.getvalue())

    def test_shared_provider_dispatch_denies_mutations_before_side_effects(self):
        create_configs = {
            'runpod': {'name': 'test', 'image': 'test', 'gpu': {'id': 'test', 'count': 1}},
            'hf_jobs': {'image': 'test', 'flavor': 'test', 'command': ['test']},
            'hf_endpoints': dict.fromkeys(
                ('name', 'repository', 'framework', 'accelerator', 'instance_size', 'instance_type', 'region', 'vendor'),
                'test',
            ),
        }
        mutations = {
            'runpod': ('create', 'stop', 'delete'),
            'hf_jobs': ('create', 'cancel'),
            'hf_endpoints': ('create', 'stop', 'delete'),
            'hf_storage': ('upload',), 's3': ('upload',), 'gcs': ('upload',),
            'ssh': ('exec', 'cancel'), 'future': ('list', 'execute'),
        }
        with ExitStack() as stack:
            guards = [stack.enter_context(patch.object(providers, name)) for name in (
                'resolve_secret', '_provider_secrets', '_intent', '_http_request', '_sdk_call', '_gcs_operation')]
            for kind, actions in mutations.items():
                for action in actions:
                    with self.subTest(kind=kind, action=action):
                        schema = providers.ACTION_SCHEMAS.get(kind, {}).get(action, {})
                        params = {'resource_id': 'test'} if 'resource_id' in schema else {}
                        if action == 'create':
                            params = {'config': create_configs[kind], 'budget_usd': 1,
                                      'timeout_seconds': 60, 'idempotency_key': 'test'}
                        if action == 'upload':
                            params = {'key': 'artifact.bin', 'local_path': self.file.name}
                        if action == 'exec':
                            params = {'plan_path': 'plan.json', 'run_id': 'test'}
                        if kind == 'ssh' and action == 'cancel':
                            params = {'run_id': 'test'}
                        self.assert_code(tool_api.dispatch(PROVIDER, provider_args(kind, action, **params)), DENIED)
            self.assert_code(tool_api.dispatch(PROVIDER, provider_args('runpod', 'future_action')), DENIED)
            for guard in guards:
                guard.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [self.file])

    def test_mcp_probe_lists_tools_and_all_mcp_tool_calls_require_operator(self):
        with patch.object(mcp_client, 'request', return_value={'ok': True}) as request:
            for action in ('probe', 'list', 'get', 'logs', 'evaluate', 'execute'):
                self.assert_code(tool_api.dispatch(PROVIDER, provider_args('mcp', action)), DENIED)
            request.assert_not_called()
            result = tool_api.dispatch('ai_lab_connection_probe', {'connection_id': 'mcp'})
            self.assertTrue(result['ok'])
            self.assertEqual(request.call_args.kwargs, {})
            self.assertEqual(len(request.call_args.args), 1)
            request.reset_mock()
            status, result = self.cli_call('provider-action', 'mcp', 'list')
            self.assertEqual(status, 0, result)
            request.assert_called_once_with(request.call_args.args[0], tool='list', arguments={})

    def test_reads_reach_provider_and_generate_remains_typed(self):
        with patch.object(providers, 'provider_action', return_value={'ok': True}) as read:
            for kind in providers.ACTION_SCHEMAS:
                for action in ('probe', 'list', 'get', 'logs'):
                    if action in providers.ACTION_SCHEMAS[kind]:
                        result = tool_api.dispatch(PROVIDER, provider_args(kind, action))
                        self.assertTrue(result['ok'], (kind, action, result))
                        self.assertEqual(read.call_args.args[1], action)
        result = tool_api.dispatch('ai_lab_generate', {'connection_id': 'mcp', 'model': 'test', 'prompt': 'test'})
        self.assert_code(result, 'invalid_input')

    def test_upload_execution_requires_named_run_for_both_storage_adapters(self):
        with patch.object(transfers, '_sdk_call', return_value={'ok': True}) as sdk, \
             patch.object(transfers, '_installed', return_value=True) as installed, \
             patch.object(jobs, 'run_live', side_effect=lambda plan, callback, **kwargs: callback()) as submit:
            for kind in ('hf_storage', 's3'):
                with self.subTest(kind=kind):
                    args = self.transfer_args(kind)
                    self.assert_code(tool_api.dispatch(RUN, args), DENIED)
                    status, result = self.cli_call('call', RUN, '--args', json.dumps(args))
                    self.assertEqual(status, 1)
                    self.assert_code(result, DENIED)
                    self.assert_code(tool_api.dispatch(RUN, args | {'approved': True}), 'invalid_arguments')
            installed.assert_not_called()
            sdk.assert_not_called()
            submit.assert_not_called()
            for kind in ('hf_storage', 's3'):
                with self.subTest(kind=kind):
                    args = self.transfer_args(kind)
                    self.assertTrue(tool_api.dispatch('ai_lab_plan', args)['plan']['metadata']['remote_write'])
                    sdk.assert_not_called()
                    status, result = self.cli_call('run', 'storage.upload', '--params', json.dumps(args['parameters']))
                    self.assertEqual(status, 0, result)
                    sdk.assert_called_once()
                    self.assertEqual(sdk.call_args.args[0]['type'], kind)
                    self.assertEqual(sdk.call_args.args[1], 'storage.upload')
                    submit.assert_called_once()
                    sdk.reset_mock()
                    submit.reset_mock()
            download = self.transfer_args('s3', 'storage.download', self.root / 'download.bin')
            self.assertTrue(tool_api.dispatch(RUN, download)['ok'])
            self.assertEqual(sdk.call_args.args[1], 'storage.download')

    def test_provider_cli_is_explicit_and_json_flags_do_not_grant_access(self):
        args = provider_args('runpod', 'stop', resource_id='test')
        with patch.object(providers, 'resolve_secret', return_value='synthetic'), \
             patch.object(providers, '_http_request', return_value={'id': 'test'}) as http:
            status, result = self.cli_call('call', PROVIDER, '--args', json.dumps(args))
            self.assertEqual(status, 1)
            self.assert_code(result, DENIED)
            for flag in ('approved', 'operator'):
                self.assert_code(tool_api.dispatch(PROVIDER, args | {flag: True}), 'invalid_arguments')
                nested = args | {'parameters': args['parameters'] | {flag: True}}
                self.assert_code(tool_api.dispatch(PROVIDER, nested), DENIED)
            http.assert_not_called()
            status, result = self.cli_call('provider-action', 'runpod', 'stop', '--params', json.dumps(args['parameters']))
            self.assertEqual(status, 0, result)
            http.assert_called_once()
            self.assertEqual(http.call_args.args[:2], ('POST', 'https://api.runpod.io/v2/pods/test/action'))
            self.assert_code(tool_api.dispatch(PROVIDER, args), DENIED)

    def test_scope_is_bound_to_command_and_resets_across_errors_and_contexts(self):
        args = provider_args('runpod', 'stop', resource_id='test')

        async def dispatch_in_task():
            return tool_api.dispatch(PROVIDER, args)

        async def check_task_isolation():
            with operator_scope('provider-action'):
                return await asyncio.create_task(dispatch_in_task())

        with patch.object(providers, 'provider_action', return_value={'ok': True}) as invoke:
            with self.assertRaisesRegex(RuntimeError, 'synthetic error'):
                with operator_scope('provider-action'):
                    self.assertTrue(tool_api.dispatch(PROVIDER, args)['ok'])
                    self.assert_code(tool_api.dispatch(RUN, self.transfer_args('s3')), DENIED)
                    self.assert_code(Context().run(tool_api.dispatch, PROVIDER, args), DENIED)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        isolated = pool.submit(copy_context().run, tool_api.dispatch, PROVIDER, args).result()
                    self.assert_code(isolated, DENIED)
                    retained = copy_context()
                    raise RuntimeError('synthetic error')
            self.assert_code(tool_api.dispatch(PROVIDER, args), DENIED)
            self.assert_code(retained.run(tool_api.dispatch, PROVIDER, args), DENIED)
            self.assert_code(asyncio.run(check_task_isolation()), DENIED)
            invoke.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec('hermes_cli'), 'optional Hermes SDK is unavailable')
class NativePolicyTests(unittest.TestCase):
    def test_real_registry_denies_direct_dispatch_and_native_cli_authorizes_once(self):
        from check_hermes_credentials import RUNPOD_KEY, load, scope
        from tools.registry import registry

        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary).resolve() / 'profile'
            manager, _ = load(ROOT, home, 'test-bucket')
            self.addCleanup(manager.unload)
            request = provider_args('runpod-test', 'stop', resource_id='test')
            args_file = home / 'args.json'
            args_file.write_text(json.dumps(request), encoding='utf-8')
            command = manager._cli_commands['ai-lab']
            parser = argparse.ArgumentParser()
            command['setup_fn'](parser)
            args = parser.parse_args(['provider-action', '--args-file', str(args_file)])
            with scope(home, {RUNPOD_KEY: 'synthetic'}), \
                 patch('urllib.request.build_opener') as transport:
                direct = json.loads(registry.dispatch(PROVIDER, request, scope=manager.scope_key))
                self.assertEqual(direct['error']['code'], DENIED)
                transport.assert_not_called()
                transport.return_value.open.return_value = io.BytesIO(b'{"id":"test"}')
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(command['handler_fn'](args), 0, output.getvalue())
                self.assertTrue(json.loads(output.getvalue())['ok'])
                transport.return_value.open.assert_called_once()
                direct = json.loads(registry.dispatch(PROVIDER, request, scope=manager.scope_key))
                self.assertEqual(direct['error']['code'], DENIED)


if __name__ == '__main__':
    unittest.main()
