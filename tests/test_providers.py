import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
import urllib.error
from unittest.mock import MagicMock, patch
from ai_lab_core import providers as p
from ai_lab_core.credentials import secret_scope

TOKEN = 'SYNTHETIC_SECRET_DO_NOT_OUTPUT'


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.c = {'id': 'gpu', 'type': 'runpod', 'credential_ref': 'hermes:AI_LAB_RUNPOD_API_KEY', 'workspace': self.tmp.name}
        self.scope = secret_scope({'AI_LAB_RUNPOD_API_KEY': TOKEN, 'AI_LAB_HF_TOKEN': TOKEN,
                                   'AI_LAB_S3_ACCESS_KEY_ID': 'synthetic-access-key',
                                   'AI_LAB_S3_SECRET_ACCESS_KEY': TOKEN})
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def create_params(self):
        return {'idempotency_key': 'test-1', 'budget_usd': 2, 'timeout_seconds': 60,
                'config': {'name': 'training', 'image': 'pytorch/pytorch:latest', 'gpu': {'id': 'NVIDIA B200', 'count': 1}, 'disk': 20}}

    def test_config_and_host_allowlist(self):
        self.assertTrue(p.validate_connection(self.c)['ok'])
        for url in ('https://evil.example/v2', 'https://api.runpod.io.evil/v2', 'https://api.runpod.io/v2?token=x', 'http://api.runpod.io/v2'):
            self.assertFalse(p.validate_connection(self.c | {'base_url': url})['ok'])
        for bad in ({}, {'id': 'x', 'type': []}, self.c | {'token': TOKEN}, self.c | {'credential_ref': TOKEN}):
            self.assertFalse(p.validate_connection(bad)['ok'])
        self.assertEqual(p.validate_connection({'id': 'h', 'type': 'hf_jobs', 'project': 'team'})['error']['code'], 'configuration_required')

    def test_list_probe_get_stop_delete_routes_and_filtering(self):
        with patch.object(p, '_http_request') as request:
            request.return_value = {'pods': [{'id': 'pod1', 'status': 'RUNNING', 'env': {'token': TOKEN}, 'ssh': {'command': TOKEN}}]}
            result = p.probe_connection(self.c)
            self.assertTrue(result['ok'])
            self.assertNotIn(TOKEN, json.dumps(result))
            request.assert_called_once_with('GET', 'https://api.runpod.io/v2/pods?limit=100', TOKEN)
            request.return_value = {'id': 'pod1', 'status': 'STOPPED', 'env': {'token': TOKEN}}
            for action, method, suffix, body in (('get', 'GET', '', None), ('stop', 'POST', '/action', {'action': 'stop'}), ('delete', 'DELETE', '', None)):
                result = p.provider_action(self.c, action, {'resource_id': 'pod1'})
                self.assertTrue(result['ok'])
                request.assert_called_with(method, 'https://api.runpod.io/v2/pods/pod1' + suffix, TOKEN, body)
                self.assertNotIn(TOKEN, json.dumps(result))

    def test_create_intent_precedes_post_and_never_duplicates(self):
        def transport(*args, **kwargs):
            self.assertTrue(list(Path(self.tmp.name).glob('.ai-lab/intents/*.json')))
            return {'id': 'pod1', 'name': 'training', 'env': {'key': TOKEN}}
        with patch.object(p, '_http_request', side_effect=transport) as request:
            result = p.provider_action(self.c, 'create', self.create_params())
            self.assertTrue(result['ok'])
            self.assertFalse(result['budget']['enforced'])
            self.assertFalse(result['timeout']['enforced'])
            self.assertEqual(p.provider_action(self.c, 'create', self.create_params())['error']['code'], 'reconciliation_required')
            self.assertEqual(request.call_count, 1)
        self.assertNotIn(TOKEN, ''.join(f.read_text() for f in Path(self.tmp.name).glob('.ai-lab/intents/*.json')))

    def test_ambiguous_create_requires_reconciliation(self):
        with patch.object(p, '_http_request', side_effect=TimeoutError(TOKEN)) as request:
            result = p.provider_action(self.c, 'create', self.create_params())
            self.assertFalse(result['ok'])
            self.assertNotIn(TOKEN, json.dumps(result))
            self.assertEqual(p.provider_action(self.c, 'create', self.create_params())['error']['code'], 'reconciliation_required')
            self.assertEqual(request.call_count, 1)

    def test_budget_and_input_validation_before_mutation(self):
        with patch.object(p, '_http_request') as request:
            for change in ({'budget_usd': 0}, {'budget_usd': float('nan')}, {'timeout_seconds': 0}, {'idempotency_key': '../x'}):
                self.assertFalse(p.provider_action(self.c, 'create', self.create_params() | change)['ok'])
            self.assertFalse(p.provider_action(self.c, 'get', {'resource_id': '../x'})['ok'])
            self.assertFalse(p.provider_action(self.c, 'create', self.create_params() | {'config': self.create_params()['config'] | {'env': {'KEY': TOKEN}}})['ok'])
            request.assert_not_called()

    def test_http_transport_headers_timeout_redirects_and_bounds(self):
        with patch.object(p.urllib.request, 'build_opener') as build:
            build.return_value.open.return_value = io.BytesIO(b'{"id":"pod1"}')
            result = p._http_request('GET', 'https://api.runpod.io/v2/pods/pod1', TOKEN)
            self.assertEqual(result['id'], 'pod1')
            req = build.return_value.open.call_args.args[0]
            self.assertEqual(req.headers['Authorization'], 'Bearer ' + TOKEN)
            self.assertEqual(build.return_value.open.call_args.kwargs['timeout'], 10)
            self.assertEqual(build.call_args.args[0].proxies, {})
            self.assertIsNone(build.call_args.args[1].redirect_request(None, None, 302, '', {}, 'https://evil'))
            build.return_value.open.return_value = io.BytesIO(b'x' * (p.MAX_BYTES + 1))
            with self.assertRaises(p._AdapterError):
                p._http_request('GET', 'https://api.runpod.io/v2/pods', TOKEN)
        with self.assertRaises(p._AdapterError):
            p._http_request('GET', 'https://evil.example/v2/pods', TOKEN)

    def test_http_errors_are_safe(self):
        for code, expected in ((302, 'redirect_refused'), (401, 'auth_required'), (403, 'forbidden'), (404, 'not_found'), (429, 'rate_limited')):
            with patch.object(p.urllib.request, 'build_opener') as build:
                build.return_value.open.side_effect = urllib.error.HTTPError('https://secret', code, TOKEN, {}, None)
                result = p.probe_connection(self.c)
            self.assertEqual(result['error']['code'], expected)
            self.assertNotIn(TOKEN, json.dumps(result))

    def test_sse_returns_metadata_only(self):
        with patch.object(p.urllib.request, 'build_opener') as build:
            build.return_value.open.return_value = io.BytesIO(('data: ' + json.dumps({'source': 'container', 'line': TOKEN}) + '\n\n').encode())
            result = p._http_request('GET', 'https://api.runpod.io/v2/pods/pod1/logs', TOKEN, stream=True, limit=1)
            self.assertTrue(result['content_withheld'])
            self.assertNotIn(TOKEN, json.dumps(result))

    def test_sdk_action_envelopes_and_hf_job_effective_timeout(self):
        c = {'id': 'hf', 'type': 'hf_jobs', 'project': 'team', 'credential_ref': 'hermes:AI_LAB_HF_TOKEN', 'workspace': self.tmp.name}
        params = self.create_params() | {'config': {'image': 'python:3.12', 'command': ['python', 'train.py'], 'flavor': 'cpu-basic'}}
        with patch.object(p, '_sdk_call', return_value={'ok': True, 'resource': {'id': 'job1'}}) as sdk:
            result = p.provider_action(c, 'create', params)
            self.assertTrue(result['timeout']['enforced'])
            self.assertEqual(sdk.call_args.args[2]['config']['timeout'], 60)
            self.assertEqual(sdk.call_args.args[2]['config']['attempts'], 1)

    def test_hf_sdk_exact_methods_and_filtering(self):
        api = MagicMock()
        api.list_jobs.return_value = [types.SimpleNamespace(id='j1', docker_image='python', flavor='cpu-basic', status=types.SimpleNamespace(stage='RUNNING'), env=TOKEN)]
        api.inspect_job.return_value = api.list_jobs.return_value[0]
        module = types.ModuleType('huggingface_hub')
        module.HfApi = MagicMock(return_value=api)
        module.get_token = MagicMock(side_effect=AssertionError("Native HF token lookup is forbidden"))
        c = {'id': 'hf', 'type': 'hf_jobs', 'project': 'team', 'credential_ref': 'hermes:AI_LAB_HF_TOKEN'}
        with patch.dict('sys.modules', {'huggingface_hub': module}), patch.object(p, '_guard_hf_http'):
            result = p._hf_operation(c, 'list', {})
            api.list_jobs.assert_called_once_with(namespace='team')
            self.assertNotIn(TOKEN, json.dumps(result))
            p._hf_operation(c, 'get', {'resource_id': 'j1'})
            api.inspect_job.assert_called_once_with(job_id='j1', namespace='team')
            p._hf_operation(c, 'cancel', {'resource_id': 'j1'})
            api.cancel_job.assert_called_once_with(job_id='j1', namespace='team')
            c['type'] = 'hf_endpoints'
            api.list_inference_endpoints.return_value = [types.SimpleNamespace(name='ep', status='running', raw=TOKEN, url='https://presigned')]
            result = p._hf_operation(c, 'list', {})
            self.assertNotIn('presigned', json.dumps(result))
            self.assertNotIn(TOKEN, json.dumps(result))
            p._hf_operation(c, 'stop', {'resource_id': 'ep'})
            api.pause_inference_endpoint.assert_called_once_with('ep', namespace='team')
            p._hf_operation(c, 'delete', {'resource_id': 'ep'})
            api.delete_inference_endpoint.assert_called_once_with('ep', namespace='team')

    def test_storage_confined_to_workspace_and_sdk_errors_safe(self):
        c = {'id': 'storage', 'type': 's3', 'region': 'eu-west-3', 'bucket': 'test-bucket', 'workspace': self.tmp.name,
             'credential_ref': 'hermes:AI_LAB_S3_ACCESS_KEY_ID'}
        (Path(self.tmp.name) / 'artifact.bin').write_bytes(b'123')
        with patch.object(p, '_sdk_call', return_value={'ok': True, 'uploaded': True}) as sdk:
            self.assertTrue(p.provider_action(c, 'upload', {'key': 'models/output', 'local_path': 'artifact.bin'})['ok'])
            self.assertEqual(sdk.call_args.args[2]['local_path'], str((Path(self.tmp.name) / 'artifact.bin').resolve()))
            for path in ('../other', '/tmp/other'):
                self.assertFalse(p.provider_action(c, 'upload', {'key': 'output', 'local_path': path})['ok'])
        with patch.object(p, '_storage_operation', side_effect=RuntimeError(TOKEN)):
            result = p._sdk_worker({'connection': c, 'action': 'probe', 'params': {}, 'secrets': {}})
            self.assertNotIn(TOKEN, json.dumps(result))

    def test_sdk_worker_missing_dependency(self):
        with patch.object(p, '_storage_operation', side_effect=ImportError(TOKEN)):
            result = p._sdk_worker({'connection': {'type': 's3'}, 'action': 'probe', 'params': {}, 'secrets': {}})
        self.assertEqual(result['error']['code'], 'dependency_missing')

    def test_hf_client_guards_redirects_destinations_and_duplicate_posts(self):
        hf = types.ModuleType('huggingface_hub')
        hf.set_client_factory = MagicMock()
        httpx = types.ModuleType('httpx')
        httpx.Client = MagicMock()
        with patch.dict('sys.modules', {'huggingface_hub': hf, 'httpx': httpx}):
            p._guard_hf_http('hf_jobs', create=True)
            hf.set_client_factory.call_args.args[0]()
            kwargs = httpx.Client.call_args.kwargs
            self.assertFalse(kwargs['follow_redirects'])
            self.assertFalse(kwargs['trust_env'])
            guard = kwargs['event_hooks']['request'][0]
            def request(host, method='GET'):
                return types.SimpleNamespace(url=types.SimpleNamespace(host=host, scheme='https', port=443, userinfo=b''), method=method)
            with self.assertRaises(p._AdapterError):
                guard(request('evil.example'))
            guard(request('huggingface.co', 'POST'))
            with self.assertRaises(p._AdapterError):
                guard(request('huggingface.co', 'POST'))
            with self.assertRaises(p._AdapterError):
                kwargs['event_hooks']['response'][0](types.SimpleNamespace(status_code=307))

    def test_hf_storage_guard_strips_bearer_from_transfer_hosts(self):
        hf = types.ModuleType('huggingface_hub')
        hf.set_client_factory = MagicMock()
        httpx = types.ModuleType('httpx')
        httpx.Client = MagicMock()
        headers = {'Authorization': 'Bearer ' + TOKEN}
        with patch.dict('sys.modules', {'huggingface_hub': hf, 'httpx': httpx}):
            for transfer in (False, True):
                p._guard_hf_http('hf_storage', transfer=transfer)
                hf.set_client_factory.call_args.args[0]()
                guard = httpx.Client.call_args.kwargs['event_hooks']['request'][0]
                for host in ('huggingface.co', 'files.hf.co', 'files.huggingface.co'):
                    with self.subTest(transfer=transfer, host=host):
                        request = types.SimpleNamespace(
                            url=types.SimpleNamespace(host=host, scheme='https', port=443, userinfo=b''),
                            method='POST', headers=headers.copy(),
                        )
                        guard(request)
                        self.assertEqual(request.headers, headers if host == 'huggingface.co' else {})

    def test_s3_explicit_sdk_credentials_and_filtered_operations(self):
        boto = types.ModuleType('boto3')
        client = MagicMock()
        boto.client = MagicMock(return_value=client)
        configmod = types.ModuleType('botocore.config')
        configmod.Config = MagicMock()
        c = {'id': 's', 'type': 's3', 'bucket': 'test-bucket', 'region': 'eu-west-3'}
        client.list_objects_v2.return_value = {'Contents': [{'Key': 'model.bin', 'Size': 20, 'secret': TOKEN}], 'IsTruncated': False}
        with patch.dict('sys.modules', {'boto3': boto, 'botocore': types.ModuleType('botocore'), 'botocore.config': configmod}):
            self.assertTrue(p._storage_operation(c, 'probe', {})['ok'])
            client.head_bucket.assert_called_once_with(Bucket='test-bucket')
            result = p._storage_operation(c, 'list', {'prefix': 'models/', 'limit': 5})
            client.list_objects_v2.assert_called_once_with(Bucket='test-bucket', Prefix='models/', MaxKeys=5)
            self.assertNotIn(TOKEN, json.dumps(result))
            guard = client.meta.events.register.call_args.args[1]
            with self.assertRaises(p._AdapterError):
                guard(types.SimpleNamespace(url='https://evil.example/upload'))
            artifact = Path(self.tmp.name) / 'artifact.bin'
            artifact.write_bytes(b'data')
            self.assertTrue(p._storage_operation(c, 'upload', {'key': 'model.bin', 'local_path': str(artifact)})['ok'])
            self.assertEqual(client.put_object.call_args.kwargs['IfNoneMatch'], '*')
            self.assertEqual(configmod.Config.call_args.kwargs['retries'], {'total_max_attempts': 1})
            self.assertEqual(boto.client.call_args.kwargs['aws_access_key_id'], 'synthetic-access-key')
            self.assertEqual(boto.client.call_args.kwargs['aws_secret_access_key'], TOKEN)
            self.assertIsNone(boto.client.call_args.kwargs['aws_session_token'])

    def test_hf_upload_uses_fixed_repo_and_selected_commit(self):
        module = types.ModuleType('huggingface_hub')
        api = MagicMock()
        module.HfApi = MagicMock(return_value=api)
        module.get_token = MagicMock(side_effect=AssertionError("Native HF token lookup is forbidden"))
        api.upload_file.return_value = types.SimpleNamespace(oid='abc', commit_url='https://secret', secret=TOKEN)
        c = {'id': 'hf', 'type': 'hf_storage', 'repo_id': 'team/model', 'credential_ref': 'hermes:AI_LAB_HF_TOKEN'}
        with patch.dict('sys.modules', {'huggingface_hub': module}), patch.object(p, '_guard_hf_http'):
            result = p._hf_operation(c, 'upload', {'local_path': '/tmp/file', 'key': 'model.bin'})
            api.upload_file.assert_called_once_with(path_or_fileobj='/tmp/file', path_in_repo='model.bin', repo_id='team/model', repo_type='model', commit_message='AI Lab artifact')
            self.assertEqual(result, {'ok': True, 'commit': {'oid': 'abc'}, 'uploaded': True})

    def test_sdk_bootstrap_is_independent_of_namespace_and_caller_cwd(self):
        # Synthetic worker validates the constant bootstrap import path without any SDK/account call.
        import sys
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'ai_lab_core'
            package.mkdir()
            (package / '__init__.py').write_text('')
            (package / 'providers.py').write_text('MAX_BYTES = 1024\ndef _sdk_worker(payload):\n return {"ok":True,"synthetic":True}\n')
            result = subprocess.run([sys.executable, '-I', '-c', p._SDK_CODE, directory], cwd=self.tmp.name,
                                    input=json.dumps({'connection': {}, 'action': 'probe', 'params': {}, 'secrets': {}}).encode(),
                                    capture_output=True, timeout=5, check=False)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout), {'ok': True, 'synthetic': True})
