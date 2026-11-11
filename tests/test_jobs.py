"""Real, short subprocess job contracts; no GPU, model, or network needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ai_lab_core import jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ai-lab-jobs-test-')
        self.root = Path(self.temp.name)
        self.state = self.root / 'state'

    def tearDown(self):
        # All tests wait for terminal state before removing their isolated home.
        self.temp.cleanup()

    def plan(self, code, timeout=5, outputs=None, env=None):
        return {'operation': 'synthetic', 'backend': 'python', 'command': [sys.executable, '-c', code],
                'cwd': str(self.root), 'timeout_seconds': timeout, 'outputs': outputs or [],
                'metadata': {'purpose': 'test'}, 'env': env or {}}

    def submit(self, code, **kwargs):
        result = jobs.submit(self.plan(code, **kwargs), self.state)
        self.assertTrue(result['ok'], result)
        return result['job']['id']

    def wait(self, job_id, state=None, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = jobs.get_job(job_id, self.state)
            self.assertTrue(result['ok'], result)
            job = result['job']
            if job['state'] == state or (state is None and job['state'] in jobs._TERMINAL):
                return job
            time.sleep(.04)
        self.fail('job did not reach expected state: ' + str(result))

    def test_success_persistence_outputs_and_safe_return(self):
        output = self.root / 'result.txt'
        job_id = self.submit("from pathlib import Path; print('done'); Path('result.txt').write_text('ok')", outputs=[{'path': str(output), 'kind': 'file'}])
        self.assertEqual(self.wait(job_id)['state'], 'succeeded')
        script = 'from ai_lab_core.jobs import get_job; import json; print(json.dumps(get_job(' + repr(job_id) + ',' + repr(str(self.state)) + ')))'
        # A second independent client reads the same durable database.
        raw = subprocess.check_output([sys.executable, '-c', script], cwd=str(Path(__file__).resolve().parents[1]), text=True)
        result = json.loads(raw)
        self.assertEqual(result['job']['state'], 'succeeded')
        self.assertNotIn('command', result['job'])
        self.assertNotIn('env', result['job'])
        self.assertIn('done', jobs.job_logs(job_id, state_dir=self.state)['text'])
        self.assertEqual(os.stat(self.state / 'jobs.sqlite3').st_mode & 0o777, 0o600)

    def test_nonzero_and_missing_output_fail(self):
        bad = self.submit('raise SystemExit(7)')
        absent = self.submit('pass', outputs=[{'path': str(self.root / 'absent'), 'kind': 'file'}])
        self.assertEqual(self.wait(bad)['reason'], 'exit_nonzero')
        self.assertEqual(self.wait(absent)['reason'], 'outputs_missing')

    def test_cancel_owned_process_group(self):
        marker = self.root / 'escaped.txt'
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; time.sleep(2); Path('escaped.txt').write_text('bad')\"]); print('ready',flush=True); time.sleep(20)"
        job_id = self.submit(code, timeout=10)
        self.wait(job_id, 'running')
        deadline = time.monotonic() + 3
        while 'ready' not in jobs.job_logs(job_id, state_dir=self.state)['text'] and time.monotonic() < deadline:
            time.sleep(.03)
        jobs.cancel_job(job_id, self.state)
        self.assertEqual(self.wait(job_id)['state'], 'cancelled')
        time.sleep(2.1)
        self.assertFalse(marker.exists())

    def test_timeout_and_serialized_queue(self):
        first = self.submit("import time; time.sleep(20)", timeout=1)
        second = self.submit("print('next')")
        a, b = self.wait(first), self.wait(second)
        self.assertEqual(a['reason'], 'timeout')
        self.assertEqual(b['state'], 'succeeded')
        self.assertGreaterEqual(b['started'], a['finished'])

    def test_queued_cancellation_and_log_bound_redaction(self):
        first = self.submit('import time; time.sleep(.5)')
        queued = self.submit("raise SystemExit('must not run')")
        self.assertEqual(jobs.cancel_job(queued, self.state)['job']['state'], 'cancelled')
        self.wait(first)
        loud = self.submit("print('x'*1500000); print('token=test-private-value')")
        self.assertEqual(self.wait(loud)['state'], 'succeeded')
        log = jobs.job_logs(loud, limit=64000, state_dir=self.state)
        self.assertLessEqual((self.state / (loud + '.log')).stat().st_size, jobs._MAX_LOG + 32768)
        self.assertNotIn('test-private-value', log['text'])
        self.assertIn('[REDACTED]', log['text'])
        self.assertLessEqual(len(log['text']), 64000)

    def test_submitter_can_exit_and_reconciliation_is_explicit(self):
        plan = self.plan("import time; time.sleep(.2); print('survived')")
        script = ('from ai_lab_core.jobs import submit; import json; '
                  'print(json.dumps(submit(' + repr(plan) + ',' + repr(str(self.state)) + ')))')
        raw = subprocess.check_output([sys.executable, '-c', script],
                                      cwd=str(Path(__file__).resolve().parents[1]), text=True)
        job_id = json.loads(raw)['job']['id']
        self.assertEqual(self.wait(job_id)['state'], 'succeeded')
        # A stale running record is marked interrupted, never relaunched.
        import uuid
        stale = str(uuid.uuid4())
        with jobs._db(jobs._root(self.state)) as db:
            db.execute('INSERT INTO jobs(id,state,plan,created) VALUES(?,?,?,?)',
                       (stale, 'running', json.dumps(self.plan('pass')), time.time()))
        self.assertFalse(jobs.reconcile(self.state)['resume_supported'])
        self.assertEqual(self.wait(stale)['state'], 'interrupted')

    def test_queued_duplicate_output_is_not_overwritten(self):
        output = self.root / 'shared.bin'
        outputs = [{'path': str(output), 'kind': 'file'}]
        first_plan = self.plan("import time; from pathlib import Path; time.sleep(.4); Path('shared.bin').write_bytes(b'first-version')", outputs=outputs)
        second_plan = self.plan("from pathlib import Path; Path('second-executed').touch(); Path('shared.bin').write_bytes(b'overwritten')", outputs=outputs)
        first = jobs.submit(first_plan, self.state)['job']['id']
        second = jobs.submit(second_plan, self.state)['job']['id']
        self.assertEqual(self.wait(first)['state'], 'succeeded')
        result = self.wait(second)
        self.assertEqual((result['state'], result['reason']), ('failed', 'output_exists'))
        self.assertIsNone(result['exit_code'])
        self.assertEqual(output.read_bytes(), b'first-version')
        self.assertFalse((self.root / 'second-executed').exists())
        # Unrelated existing files do not prevent operations without outputs.
        self.assertEqual(self.wait(self.submit('pass'))['state'], 'succeeded')

    def test_queued_output_parent_symlink_escape_is_rejected(self):
        parent = self.root / 'outputs'
        parent.mkdir()
        output = parent / 'model.bin'
        blocker = self.submit("import time; from pathlib import Path\nwhile not Path('release').exists(): time.sleep(.01)")
        victim = self.submit("from pathlib import Path; Path('victim-executed').touch(); Path('outputs/model.bin').write_bytes(b'bad')",
                             outputs=[{'path': str(output), 'kind': 'file'}])
        with tempfile.TemporaryDirectory(prefix='ai-lab-outside-workspace-') as outside:
            parent.rmdir()
            parent.symlink_to(outside, target_is_directory=True)
            (self.root / 'release').touch()
            self.assertEqual(self.wait(blocker)['state'], 'succeeded')
            result = self.wait(victim)
            self.assertEqual((result['state'], result['reason']), ('failed', 'output_path_invalid'))
            self.assertFalse((Path(outside) / 'model.bin').exists())
            self.assertFalse((self.root / 'victim-executed').exists())

    def test_invalid_id_and_symlink_state_rejected(self):
        self.assertFalse(jobs.get_job('../bad', self.state)['ok'])
        self.assertFalse(jobs.submit(self.plan('pass', timeout=0), self.state)['ok'])
        target = self.root / 'target'
        target.mkdir()
        link = self.root / 'link'
        link.symlink_to(target, target_is_directory=True)
        self.assertFalse(jobs.submit(self.plan('pass'), link)['ok'])

    def test_job_logs_tolerate_a_journal_removed_by_a_concurrent_commit(self):
        with patch.object(jobs, '_launch'):
            job_id = self.submit('pass')
        journal = self.state.resolve() / 'jobs.sqlite3-journal'
        journal.touch()
        lstat = Path.lstat

        def commit_then_lstat(path, *args, **kwargs):
            if path == journal:
                journal.unlink(missing_ok=True)
            return lstat(path, *args, **kwargs)

        with patch.object(Path, 'lstat', commit_then_lstat):
            result = jobs.job_logs(job_id, state_dir=self.state)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['text'], '')

        journal.symlink_to(self.root / 'unrelated-file')
        self.assertEqual(jobs.job_logs(job_id, state_dir=self.state)['error']['code'], 'invalid_request')


if __name__ == '__main__':
    unittest.main()
