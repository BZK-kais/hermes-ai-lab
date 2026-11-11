"""No native SSH credential discovery until Hermes supplies an auth route."""
import unittest
from unittest.mock import patch
from ai_lab_core import remote


class RemoteTests(unittest.TestCase):
    def test_valid_connection_is_not_execution_ready(self):
        connection = {'id': 'gpu', 'type': 'ssh', 'host': 'gpu.example', 'user': 'operator'}
        result = remote.validate_connection(connection)
        self.assertTrue(result['ok'])
        self.assertFalse(result['execution_available'])
        with patch('subprocess.run') as run, patch('subprocess.Popen') as popen:
            for action, params in [('probe', {}), ('exec', {'run_id': 'one', 'plan_path': 'plan.json'}), ('status', {'run_id': 'one'}), ('cancel', {'run_id': 'one'})]:
                self.assertEqual(remote.remote_action(connection, action, params)['error']['code'], 'auth_route_unavailable')
            run.assert_not_called()
            popen.assert_not_called()

    def test_invalid_host_and_arguments_rejected(self):
        connection = {'id': 'gpu', 'type': 'ssh', 'host': 'gpu.example'}
        for change in ({'host': '-ProxyCommand=evil'}, {'host': '../host'}, {'user': 'a;sh'}, {'port': True}, {'script': '../runner.py'}):
            self.assertFalse(remote.validate_connection(connection | change)['ok'])
        self.assertEqual(remote.remote_action(connection, 'exec', {'command': 'sh'})['error']['code'], 'invalid_input')
