"""Credential isolation and live job receipts using only synthetic data."""
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

from ai_lab_core import jobs
from ai_lab_core.credentials import secret_scope, resolve_secret
from ai_lab_core.process_env import runtime_env


class CredentialJobTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name).resolve()
        self.state = self.workspace / 'state'
        self.plan = {
            'operation': 'storage.download', 'backend': 'storage',
            'cwd': str(self.workspace), 'outputs': [], 'timeout_seconds': 5, 'metadata': {},
        }

    def wait(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = jobs.get_job(job_id, self.state)
            self.assertTrue(result['ok'], result)
            job = result['job']
            if job['state'] in jobs._TERMINAL:
                return job
            time.sleep(.02)
        self.fail('Job did not reach a terminal state.')

    def test_detached_environment_is_allowlisted_and_never_persisted(self):
        ambient = {
            'AI_LAB_HF_TOKEN': 'synthetic-ambient-value', 'HF_TOKEN': 'synthetic-ambient-value',
            'AWS_ACCESS_KEY_ID': 'synthetic-ambient-value', 'HTTP_PROXY': 'synthetic-ambient-value',
            'PYTHONPATH': 'synthetic-ambient-value', 'HF_HUB_OFFLINE': '0',
        }
        code = ('import os; '
                'assert os.environ["HF_HUB_OFFLINE"] == "1"; '
                'assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"; '
                'assert os.environ["PYTHONUNBUFFERED"] == "1"; '
                'assert not any(key in os.environ for key in ' + repr(tuple(k for k in ambient if k != 'HF_HUB_OFFLINE')) + ')')
        plan = {**self.plan, 'backend': 'python', 'operation': 'example',
                'command': [sys.executable, '-c', code], 'env': {'PYTHONUNBUFFERED': '1'}}
        with patch.dict(os.environ, ambient):
            self.assertEqual(runtime_env({'HF_HUB_OFFLINE': '0'})['HF_HUB_OFFLINE'], '0')
            with self.assertRaises(ValueError):
                runtime_env({'AI_LAB_HF_TOKEN': 'synthetic-override-value'})
            rejected = jobs.submit({**plan, 'env': {'AI_LAB_HF_TOKEN': 'synthetic-override-value'}}, self.state)
            self.assertEqual(rejected['error']['code'], 'submission_failed')
            result = jobs.submit(plan, self.state)
        self.assertTrue(result['ok'], result)
        job = self.wait(result['job']['id'])
        self.assertEqual(job['state'], 'succeeded')
        with jobs._db(jobs._root(self.state)) as db:
            persisted = db.execute('SELECT plan FROM jobs').fetchall()
        self.assertEqual(len(persisted), 1)
        self.assertNotIn('synthetic-ambient-value', persisted[0]['plan'])
        self.assertNotIn('synthetic-override-value', persisted[0]['plan'])

    def test_legacy_storage_job_never_launches(self):
        plan = {**self.plan, 'command': [sys.executable, '-c',
                                        'from pathlib import Path; Path("escaped").touch()']}
        self.assertEqual(jobs.submit(plan, self.state)['error']['code'], 'session_required')
        job_id = str(uuid.uuid4())
        with jobs._db(jobs._root(self.state)) as db:
            db.execute('INSERT INTO jobs(id,state,plan,created) VALUES(?,?,?,?)',
                       (job_id, 'queued', json.dumps(plan), time.time()))
        self.assertTrue(jobs.reconcile(self.state)['ok'])
        job = self.wait(job_id)
        self.assertEqual((job['state'], job['reason']), ('failed', 'session_required'))
        self.assertFalse((self.workspace / 'escaped').exists())

    def test_live_job_uses_session_and_persists_receipt(self):
        output = self.workspace / 'result.bin'
        plan = {**self.plan, 'outputs': [{'path': str(output), 'kind': 'file'}],
                'command': ['unused', 'synthetic-argument-value']}

        def transfer():
            self.assertEqual(resolve_secret('hermes:AI_LAB_HF_TOKEN'), 'synthetic-session-value')
            current = jobs.list_jobs(state_dir=self.state)['jobs'][0]
            self.assertEqual(current['state'], 'running')
            self.assertEqual(jobs.cancel_job(current['id'], self.state)['error']['code'], 'cancel_unsupported')
            output.write_bytes(b'result')
            return {'ok': True, 'bytes': 6, 'token': 'synthetic-session-value'}

        with secret_scope({'AI_LAB_HF_TOKEN': 'synthetic-session-value'}):
            result = jobs.run_live(plan, transfer, self.state)
        self.assertTrue(result['ok'], result)
        job = result['job']
        self.assertEqual(job['state'], 'succeeded')
        self.assertEqual(job['receipt'], {'ok': True, 'bytes': 6, 'token': '[REDACTED]'})
        self.assertEqual(jobs.get_job(job['id'], self.state), result)
        with jobs._db(jobs._root(self.state)) as db:
            persisted = db.execute('SELECT plan FROM jobs').fetchone()['plan']
        self.assertNotIn('synthetic-session-value', persisted)
        self.assertNotIn('synthetic-argument-value', persisted)

    def test_live_failures_are_terminal_and_lock_is_released(self):
        def fail():
            raise RuntimeError('private provider exception')

        failure = jobs.run_live(self.plan, fail, self.state)
        self.assertEqual((failure['job']['state'], failure['job']['reason']), ('failed', 'execution_error'))
        self.assertNotIn('private provider exception', str(failure))
        callback_result = {'ok': False, 'error': {'code': 'credential_unavailable', 'message': 'private value'}}
        result = jobs.run_live(self.plan, lambda: callback_result, self.state)
        self.assertEqual(result['job']['reason'], 'credential_unavailable')
        self.assertNotIn('private value', str(result))
        self.assertEqual(jobs.run_live(self.plan, lambda: {'ok': True}, self.state)['job']['state'], 'succeeded')

    def test_live_busy_does_not_run_or_enqueue(self):
        resource = jobs._owned_file(jobs._root(self.state) / 'execution.lock')
        try:
            fcntl.flock(resource, fcntl.LOCK_EX)
            result = jobs.run_live(self.plan, lambda: self.fail('Callback ran during lock contention.'), self.state)
            self.assertEqual(result['error']['code'], 'job_busy')
            self.assertEqual(jobs.list_jobs(state_dir=self.state)['jobs'], [])
        finally:
            os.close(resource)
