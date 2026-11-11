import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from ai_lab_core import providers, service, transfers
from ai_lab_core.credentials import resolve_secret, secret_scope


HF = {'id': 'hf', 'type': 'hf_storage', 'repo_id': 'org/model',
      'credential_ref': 'hermes:AI_LAB_HF_TOKEN'}
GCS = {'id': 'gcs', 'type': 'gcs', 'project': 'test-project', 'bucket': 'test-bucket'}
TOKEN = 'synthetic-provider-key'


class ProviderCredentialTests(unittest.TestCase):
    def test_adapter_rejects_legacy_and_cross_provider_references(self):
        for ref in ('native', 'env:HF_TOKEN', 'file:/tmp/hf-token',
                    'gcp:projects/p/secrets/s/versions/latest', 'hermes:AI_LAB_RUNPOD_API_KEY'):
            with self.subTest(ref=ref):
                result = providers.provider_action({**HF, 'credential_ref': ref}, 'probe', {})
                self.assertEqual(result['error']['code'], 'invalid_config')
        self.assertFalse(providers.validate_connection({**GCS, 'credential_ref': 'native'})['ok'])

    def test_missing_hermes_key_never_launches_sdk(self):
        with secret_scope({}), patch.dict(os.environ, {'HF_TOKEN': TOKEN}), patch.object(providers.subprocess, 'run') as run:
            result = providers.provider_action(HF, 'probe', {})
        self.assertEqual(result['error']['code'], 'credential_unavailable')
        run.assert_not_called()

    def test_sdk_auth_only_uses_stdin_and_cannot_return_secret(self):
        def worker(argv, **kwargs):
            self.assertNotIn(TOKEN, json.dumps(argv))
            self.assertNotIn(TOKEN, json.dumps(kwargs['env']))
            self.assertNotIn('AWS_SHARED_CREDENTIALS_FILE', kwargs['env'])
            self.assertNotEqual(kwargs['env']['HOME'], os.environ.get('HOME'))
            self.assertEqual(kwargs['env']['HF_HUB_DISABLE_IMPLICIT_TOKEN'], '1')
            self.assertEqual(json.loads(kwargs['input'])['secrets'], {'AI_LAB_HF_TOKEN': TOKEN})
            self.assertEqual(kwargs['stdout'], subprocess.PIPE)
            return subprocess.CompletedProcess(argv, 0, json.dumps({'ok': True, 'echo': TOKEN}).encode())

        scope = ModuleType('agent.secret_scope')
        scope.get_secret = Mock(return_value=TOKEN)
        with patch.dict(sys.modules, {'agent': ModuleType('agent'), 'agent.secret_scope': scope}), \
                patch.dict(os.environ, {'HF_TOKEN': TOKEN, 'AWS_SHARED_CREDENTIALS_FILE': '/sensitive/path'}), \
                patch.object(providers.subprocess, 'run', side_effect=worker):
            result = providers.provider_action(HF, 'probe', {})
        scope.get_secret.assert_called_once_with('AI_LAB_HF_TOKEN')
        self.assertEqual(result['echo'], '[redacted]')

    def test_sdk_worker_uses_only_its_payload_scope(self):
        def sdk(connection, action, parameters):
            return {'ok': True, 'secret_echo': resolve_secret(connection['credential_ref'])}

        with secret_scope({'AI_LAB_HF_TOKEN': 'outer-fixture'}), patch.object(providers, '_hf_operation', side_effect=sdk):
            result = providers._sdk_worker({'connection': HF, 'action': 'probe', 'params': {},
                                            'secrets': {'AI_LAB_HF_TOKEN': TOKEN}})
            self.assertEqual(resolve_secret(HF['credential_ref']), 'outer-fixture')
        self.assertEqual(result['secret_echo'], '[redacted]')

    def test_gcs_sign_in_uses_google_helper_with_fixed_origin_and_projected_fields(self):
        def response(data, status=200):
            return SimpleNamespace(status=status, body=json.dumps(data).encode(), json=lambda: data)

        class NotSignedIn(RuntimeError):
            pass

        helper = ModuleType('hermes_cli.plugin_provider_requests')
        helper.ProviderNotSignedIn = NotSignedIn
        helper.credentialed_provider_request = Mock(side_effect=[
            response({'name': 'test-bucket'}),
            response({'items': [{'name': 'model.bin', 'size': '42', 'generation': '3', 'owner': TOKEN}],
                      'nextPageToken': TOKEN}),
            response({'name': 'a/model.bin', 'size': '42', 'generation': '3'}),
            response({}, 302),
            NotSignedIn(TOKEN),
        ])
        with patch.dict(sys.modules, {'hermes_cli': ModuleType('hermes_cli'),
                                      'hermes_cli.plugin_provider_requests': helper}):
            self.assertTrue(providers.probe_connection(GCS)['bucket_verified'])
            listing = providers.provider_action(GCS, 'list', {'prefix': 'models', 'limit': 2})
            self.assertTrue(listing['is_truncated'])
            self.assertEqual(listing['objects'][0], {'name': 'model.bin', 'size': 42, 'generation': 3})
            self.assertNotIn(TOKEN, json.dumps(listing))
            self.assertEqual(providers.provider_action(GCS, 'get', {'key': 'a/model.bin'})['object']['name'],
                             'a/model.bin')
            self.assertEqual(providers.probe_connection(GCS)['error']['code'], 'redirect_refused')
            result = providers.probe_connection(GCS)
            self.assertEqual(result['error']['code'], 'credential_unavailable')
            self.assertNotIn(TOKEN, json.dumps(result))
            self.assertEqual(providers.provider_action(GCS, 'upload', {})['error']['code'], 'unsupported_transport')
        requests = helper.credentialed_provider_request.call_args_list
        self.assertEqual(len(requests), 5)
        for request in requests:
            self.assertEqual(request.args[:2], ('google', 'GET'))
            self.assertEqual(urlsplit(request.args[2]).netloc, 'storage.googleapis.com')
            self.assertEqual(request.kwargs, {'timeout': 10})
        self.assertEqual(parse_qs(urlsplit(requests[1].args[2]).query)['maxResults'], ['2'])
        self.assertTrue(urlsplit(requests[2].args[2]).path.endswith('/o/a%2Fmodel.bin'))

    def test_storage_catalog_and_plan_reject_gcs_byte_transfers(self):
        settings = {'integrations': [GCS]}
        plan = transfers.plan_operation('storage.download', {
            'connection_id': GCS['id'], 'key': 'model.bin', 'local_path': '/tmp/model.bin',
        }, settings)
        self.assertEqual(plan['error']['code'], 'unsupported_transport')
        connection = transfers.catalog(settings)['operations'][0]['connections'][0]
        self.assertFalse(connection['available'])
        self.assertEqual(connection['missing_requirements'][0]['code'], 'unsupported_transport')

    def test_live_transfer_records_a_safe_durable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'model.bin'
            state = root / 'state'
            settings = {'integrations': [HF], 'workspace_root': directory, 'state_dir': str(state)}

            def worker(argv, **kwargs):
                request = json.loads(kwargs['input'])
                self.assertEqual(request['action'], 'storage.download')
                self.assertEqual(request['secrets'], {'AI_LAB_HF_TOKEN': TOKEN})
                self.assertNotIn(TOKEN, json.dumps(argv))
                self.assertNotIn(TOKEN, json.dumps(kwargs['env']))
                Path(request['params']['local_path']).write_bytes(b'artifact')
                result = {'ok': True, 'bytes': 8, 'echo': TOKEN}
                return subprocess.CompletedProcess(argv, 0, json.dumps(result).encode())

            with patch('ai_lab_core.integration_config.load_integrations', return_value=settings), \
                    patch.object(transfers, '_installed', return_value=True), \
                    patch.object(providers.subprocess, 'run', side_effect=worker), \
                    secret_scope({'AI_LAB_HF_TOKEN': TOKEN}):
                result = service.run('storage.download', {'connection_id': 'hf', 'key': 'model.bin',
                                                          'local_path': str(target)})
                receipt = service.job('get', result['job']['id'])['job']['receipt']
            self.assertEqual(result['job']['state'], 'succeeded')
            self.assertEqual(target.read_bytes(), b'artifact')
            self.assertEqual(receipt['echo'], '[redacted]')
            self.assertNotIn(TOKEN, json.dumps(result))
            for file in state.iterdir():
                if file.is_file():
                    self.assertNotIn(TOKEN.encode(), file.read_bytes())

    def test_direct_legacy_worker_refuses_authentication(self):
        worker = Path(transfers.__file__).with_name('transfer_task.py')
        result = subprocess.run([sys.executable, str(worker), 'storage.download'], capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)['error']['code'], 'credential_unavailable')
