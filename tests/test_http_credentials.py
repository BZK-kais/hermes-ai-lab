"""Credential routing for HTTP and inference without touching real credentials."""
from contextvars import ContextVar
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ai_lab_core import http_api, inference
from ai_lab_core.credentials import CredentialError


class HTTPSecretsTests(unittest.TestCase):
    def test_key_resolved_before_binding_and_never_written(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(http_api, 'resolve_secret', return_value='s' * 40) as secret, patch.object(http_api, '_Server') as server:
            result = http_api.create_server(0)
            secret.assert_called_once_with('hermes:AI_LAB_HTTP_API_KEY')
            self.assertEqual(result._api_token, 's' * 40)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            server.assert_called_once()

    def test_missing_or_weak_key_never_binds(self):
        for value in ('short', 'é' * 40):
            with patch.object(http_api, 'resolve_secret', return_value=value), patch.object(http_api, '_Server') as server:
                with self.assertRaises(CredentialError):
                    http_api.create_server(0)
                server.assert_not_called()
        with patch.object(http_api, 'resolve_secret', side_effect=CredentialError('Unavailable.')), patch.object(http_api, '_Server') as server:
            with self.assertRaises(CredentialError):
                http_api.create_server(0)
            server.assert_not_called()

    def test_server_thread_uses_starting_profile_context(self):
        from contextvars import copy_context
        profile = ContextVar('test_profile', default='unscoped')
        token = profile.set('profile-a')
        captured = copy_context()
        profile.reset(token)
        instance = object.__new__(http_api._Server)
        instance._request_context = captured
        seen = []
        with patch('http.server.ThreadingHTTPServer.process_request_thread', side_effect=lambda *a: seen.append(profile.get())):
            instance.process_request_thread(object(), ('127.0.0.1', 1))
        self.assertEqual(seen, ['profile-a'])
        self.assertEqual(profile.get(), 'unscoped')

    def test_inference_cannot_resolve_another_provider_key(self):
        from ai_lab_core import credentials
        with patch.object(credentials, 'resolve_credential', return_value='synthetic-key') as resolve:
            self.assertEqual(inference._credential({'type': 'inference', 'credential_ref': 'hermes:AI_LAB_INFERENCE_API_KEY'}), 'synthetic-key')
            resolve.assert_called_once()
            with self.assertRaises(CredentialError):
                inference._credential({'type': 'inference', 'credential_ref': 'hermes:AI_LAB_HF_TOKEN'})
            self.assertEqual(resolve.call_count, 1)
