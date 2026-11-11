from contextvars import ContextVar
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from ai_lab_core.credentials import (
    CredentialError, SECRET_NAMES, resolve_credential, resolve_secret, secret_scope, validate_reference,
)


class CredentialsTests(unittest.TestCase):
    def setUp(self):
        self.hermes_agent = ModuleType('agent')
        self.hermes_scope = ModuleType('agent.secret_scope')
        self.hermes_scope.get_secret = Mock()
        modules = patch.dict(sys.modules, {'agent': self.hermes_agent, 'agent.secret_scope': self.hermes_scope})
        modules.start()
        self.addCleanup(modules.stop)

    def test_fixed_references(self):
        self.assertEqual(SECRET_NAMES, {
            'AI_LAB_RUNPOD_API_KEY', 'AI_LAB_HF_TOKEN', 'AI_LAB_INFERENCE_API_KEY',
            'AI_LAB_MCP_API_KEY', 'AI_LAB_HTTP_API_KEY', 'AI_LAB_S3_ACCESS_KEY_ID',
            'AI_LAB_S3_SECRET_ACCESS_KEY', 'AI_LAB_S3_SESSION_TOKEN',
        })
        for name in SECRET_NAMES:
            self.assertTrue(validate_reference('hermes:' + name))
        for ref in ('env:AI_LAB_HF_TOKEN', 'file:/tmp/token',
                    'gcp:projects/test-project/secrets/token/versions/latest', 'native',
                    'hermes:OTHER_TOKEN', 'hermes:AI_LAB_HF_TOKEN\n', None):
            self.assertFalse(validate_reference(ref, allow_native=True))
            with self.assertRaisesRegex(CredentialError, '^Credential reference is invalid\\.$'):
                resolve_secret(ref)
        self.hermes_scope.get_secret.assert_not_called()

    def test_resolves_only_via_hermes(self):
        self.hermes_scope.get_secret.return_value = 'scoped-test-value'
        with patch.dict(os.environ, {'AI_LAB_HF_TOKEN': 'ambient-test-value'}):
            self.assertEqual(resolve_credential('hermes:AI_LAB_HF_TOKEN'), 'scoped-test-value')
        self.hermes_scope.get_secret.assert_called_once_with('AI_LAB_HF_TOKEN')

    def test_missing_hermes_scope_value_and_sdk_fail_closed(self):
        ref = 'hermes:AI_LAB_HF_TOKEN'
        with patch.dict(os.environ, {'AI_LAB_HF_TOKEN': 'ambient-test-value'}):
            self.hermes_scope.get_secret.side_effect = RuntimeError('private implementation detail')
            with self.assertRaisesRegex(CredentialError, '^Credential unavailable\\.$'):
                resolve_secret(ref)
            self.hermes_scope.get_secret.side_effect = None
            self.hermes_scope.get_secret.return_value = None
            with self.assertRaisesRegex(CredentialError, '^Credential unavailable\\.$'):
                resolve_secret(ref)
            with patch.dict(sys.modules, {'agent.secret_scope': None}):
                with self.assertRaisesRegex(CredentialError, '^Credential unavailable\\.$'):
                    resolve_secret(ref)

    def test_rejects_invalid_header_values(self):
        for value in ('', '   ', 'x' * 16385, 'line\r\nAuthorization: injected',
                      '\nleading', 'trailing\r', 'nul\x00here', '\ud800'):
            self.hermes_scope.get_secret.return_value = value
            with self.assertRaises(CredentialError):
                resolve_secret('hermes:AI_LAB_HF_TOKEN')

    def test_sequential_profile_switch_has_no_cache(self):
        scope = ContextVar('test_hermes_profile', default={})
        self.hermes_scope.get_secret.side_effect = lambda name: scope.get().get(name)
        for value in ('first-profile-value', 'second-profile-value'):
            token = scope.set({'AI_LAB_HF_TOKEN': value})
            try:
                self.assertEqual(resolve_secret('hermes:AI_LAB_HF_TOKEN'), value)
            finally:
                scope.reset(token)
        with self.assertRaises(CredentialError):
            resolve_secret('hermes:AI_LAB_HF_TOKEN')

    def test_internal_ipc_scope_is_authoritative_and_restored(self):
        self.hermes_scope.get_secret.return_value = 'hermes-session-value'
        with secret_scope({'AI_LAB_HF_TOKEN': 'child-payload-value'}):
            self.assertEqual(resolve_secret('hermes:AI_LAB_HF_TOKEN'), 'child-payload-value')
            with secret_scope({}):
                with self.assertRaises(CredentialError):
                    resolve_secret('hermes:AI_LAB_HF_TOKEN')
            self.assertEqual(resolve_secret('hermes:AI_LAB_HF_TOKEN'), 'child-payload-value')
        self.hermes_scope.get_secret.assert_not_called()
        self.assertEqual(resolve_secret('hermes:AI_LAB_HF_TOKEN'), 'hermes-session-value')
        for values in ({'OTHER_TOKEN': 'test'}, {'AI_LAB_HF_TOKEN': 'bad\nvalue'}):
            with self.assertRaises(CredentialError):
                with secret_scope(values):
                    self.fail('Invalid IPC credentials were accepted.')
