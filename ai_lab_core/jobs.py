"""Durable local jobs. Plans are trusted internal inputs, never an API shell surface.

SQLite is shared directly by clients and detached workers; no socket service is
required. Log redaction is best effort, not a guarantee of secret-free output.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import time
import uuid

from .process_env import runtime_env, validate_overrides

_TERMINAL = {'cancelled', 'succeeded', 'failed', 'interrupted'}
_SECRET = re.compile(r'(secret|password|passwd|token|api[_-]?key|authorization|credential)', re.I)
_MAX_LOG = 1024 * 1024


def _error(code, message):
    return {'ok': False, 'error': {'code': code, 'message': message}}


def _root(state_dir=None):
    if not state_dir:
        # The detached worker loads this module by filename with an explicit root.
        from .profile import bound_home, data_root

        override = os.environ.get('AI_LAB_STATE_DIR') if bound_home() is None else None
        state_dir = override or data_root() / 'state'
    path = Path(state_dir).absolute()
    if path.is_symlink():
        raise ValueError('unsafe state path')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid():
        raise ValueError('state owner')
    os.chmod(path, 0o700)
    return path.resolve()


def _owned_file(path):
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        os.close(fd)
        raise ValueError('unsafe state file')
    os.fchmod(fd, 0o600)
    return fd


@contextmanager
def _db(root):
    path = root / 'jobs.sqlite3'
    os.close(_owned_file(path))
    for suffix in ('-journal', '-wal', '-shm'):
        other = Path(str(path) + suffix)
        try:
            info = other.lstat()
        except FileNotFoundError:
            # SQLite may unlink a rollback journal when another job commits.
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError('unsafe database sidecar')
    db = sqlite3.connect(str(path), timeout=15)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=15000')
    db.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, state TEXT NOT NULL, plan TEXT NOT NULL,
        created REAL NOT NULL, started REAL, finished REAL,
        exit_code INTEGER, reason TEXT)''')
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def _id(value):
    return isinstance(value, str) and str(uuid.UUID(value)) == value


def _scrub(text):
    text = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', text)
    text = re.sub(r'\b(?:hf_|sk-)[A-Za-z0-9_-]{8,}', '[REDACTED]', text)
    return text


def _clean(value, depth=0):
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {str(k)[:80]: '[REDACTED]' if _SECRET.search(str(k)) else _clean(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [_clean(v, depth + 1) for v in value[:40]]
    if isinstance(value, str):
        return _scrub(value[:1000])
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def _public(row):
    p = json.loads(row['plan'])
    job = {k: row[k] for k in ('id', 'state', 'created', 'started', 'finished', 'exit_code', 'reason')} | {
        'operation': p['operation'], 'backend': p['backend'], 'metadata': _clean(p.get('metadata', {}))}
    if p.get('_live') is True and 'receipt' in p:
        job['receipt'] = p['receipt']
    return job


def _validate(plan, *, live=False):
    if not isinstance(plan, dict):
        raise ValueError('plan')
    if not all(isinstance(plan.get(k), str) and 0 < len(plan[k]) <= 100 and re.fullmatch(r'[A-Za-z0-9_.-]+', plan[k]) for k in ('operation', 'backend')):
        raise ValueError('identity')
    cmd = plan.get('command')
    if not live and (not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and '\x00' not in x for x in cmd)):
        raise ValueError('command')
    if not isinstance(plan.get('cwd'), str) or not Path(plan['cwd']).is_absolute() or not Path(plan['cwd']).is_dir():
        raise ValueError('cwd')
    timeout = plan.get('timeout_seconds')
    if type(timeout) is not int or not 1 <= timeout <= 7 * 24 * 3600:
        raise ValueError('timeout')
    outputs = plan.get('outputs')
    if not isinstance(outputs, list) or not all(isinstance(o, dict) and isinstance(o.get('path'), str) and Path(o['path']).is_absolute() and o.get('kind') in ('file', 'directory') for o in outputs):
        raise ValueError('outputs')
    if not isinstance(plan.get('metadata'), dict):
        raise ValueError('metadata')
    validate_overrides(plan.get('env', {}))
    # Preserve the intended workspace before a queued job can observe changed
    # symlink components. This field is internal and never trusted from callers.
    if live:
        encoded_plan = {key: plan[key] for key in ('operation', 'backend', 'cwd', 'timeout_seconds', 'outputs')}
        encoded_plan['metadata'] = _clean(plan['metadata'])
        encoded_plan['_live'] = True
    else:
        keys = ('operation', 'backend', 'command', 'cwd', 'timeout_seconds', 'outputs', 'metadata', 'env', 'report_receipt')
        encoded_plan = {key: plan[key] for key in keys if key in plan}
    encoded_plan['outputs'] = [{'path': item['path'], 'kind': item['kind']} for item in plan['outputs']]
    encoded_plan['_output_workspace'] = str(Path(plan['cwd']).resolve(strict=True))
    return json.dumps(encoded_plan, allow_nan=False)


def _launch(root):
    # Execute a filesystem script: works for ordinary and Hermes-namespaced imports.
    launcher = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).with_name('job_worker.py')), '--detach', str(root)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True, env=runtime_env())
    if launcher.wait(timeout=10) != 0:
        raise OSError('worker launch failed')


def submit(plan, state_dir=None):
    try:
        if isinstance(plan, dict) and _requires_session(plan):
            return _error('session_required', 'Storage transfers require a live Hermes session.')
        encoded = _validate(plan)
        root = _root(state_dir)
        job_id = str(uuid.uuid4())
        with _db(root) as db:
            db.execute('INSERT INTO jobs(id,state,plan,created) VALUES(?,?,?,?)', (job_id, 'queued', encoded, time.time()))
        _launch(root)
        return get_job(job_id, root)
    except (ValueError, TypeError, OSError, sqlite3.Error):
        return _error('submission_failed', 'The job could not be submitted.')


def _requires_session(plan):
    return plan.get('backend') == 'storage' or plan.get('operation') in ('storage.download', 'storage.upload')


def _output_path_reason(plan, require_absent=False):
    """Recheck output destinations immediately before execution."""
    try:
        workspace = (Path(plan['_output_workspace']) if '_output_workspace' in plan
                     else Path(plan['cwd']).resolve(strict=True))
        if Path(plan['cwd']).resolve(strict=True) != workspace:
            return 'output_path_invalid'
        for item in plan['outputs']:
            path = Path(item['path'])
            if require_absent and (path.exists() or path.is_symlink()):
                return 'output_exists'
            resolved = path.resolve(strict=False)
            if resolved == workspace:
                return 'output_path_invalid'
            resolved.relative_to(workspace)
    except (OSError, ValueError, RuntimeError):
        return 'output_path_invalid'
    return None


def _outputs_present(plan):
    return all(
        (Path(item['path']).is_file() if item['kind'] == 'file' else Path(item['path']).is_dir())
        and not Path(item['path']).is_symlink()
        for item in plan['outputs']
    )


def _run_live_callback(plan, callback):
    reason = _output_path_reason(plan, require_absent=True)
    if reason:
        return 'failed', None, reason

    result = callback()
    if not isinstance(result, dict) or type(result.get('ok')) is not bool:
        raise ValueError('callback result')
    if not result['ok']:
        error = result.get('error')
        code = error.get('code') if isinstance(error, dict) else None
        reason = code if isinstance(code, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', code) else 'execution_error'
        plan['receipt'] = _error(reason, 'The live operation failed.')
        return 'failed', None, reason

    reason = _output_path_reason(plan)
    if reason:
        return 'failed', None, reason
    if not _outputs_present(plan):
        return 'failed', None, 'outputs_missing'
    receipt = {**_clean(result), 'ok': True}
    json.dumps(receipt, allow_nan=False)
    plan['receipt'] = receipt
    return 'succeeded', 0, None


def run_live(plan, callback, state_dir=None):
    """Run a bounded callback in the active session and record its safe receipt.

    The trusted callback owns its timeout and returns public, JSON-compatible data.
    """
    try:
        if not callable(callback):
            raise ValueError('callback')
        encoded = _validate(plan, live=True)
        stored_plan = json.loads(encoded)
        root = _root(state_dir)
        resource = _owned_file(root / 'execution.lock')
        try:
            try:
                fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return _error('job_busy', 'Another job is using the execution lock.')
            job_id = str(uuid.uuid4())
            started = time.time()
            with _db(root) as db:
                db.execute('INSERT INTO jobs(id,state,plan,created,started) VALUES(?,?,?,?,?)',
                           (job_id, 'running', encoded, started, started))
            state, code, reason = 'failed', None, 'execution_error'
            try:
                state, code, reason = _run_live_callback(stored_plan, callback)
            except Exception:
                # Keep the fixed failure state for exceptions from the callback.
                pass
            finally:
                with _db(root) as db:
                    db.execute('UPDATE jobs SET state=?,finished=?,exit_code=?,reason=?,plan=? WHERE id=?',
                               (state, time.time(), code, reason, json.dumps(stored_plan, allow_nan=False), job_id))
            return get_job(job_id, root)
        finally:
            os.close(resource)
    except (ValueError, TypeError, OSError, sqlite3.Error):
        return _error('submission_failed', 'The job could not be submitted.')


def get_job(job_id, state_dir=None):
    try:
        if not _id(job_id):
            raise ValueError('id')
        with _db(_root(state_dir)) as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        return {'ok': True, 'job': _public(row)} if row else _error('not_found', 'The job was not found.')
    except (ValueError, TypeError, AttributeError, OSError, sqlite3.Error):
        return _error('invalid_job', 'The job or state directory is invalid.')


def list_jobs(limit=20, state_dir=None):
    try:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('limit')
        with _db(_root(state_dir)) as db:
            rows = db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT ?', (limit,)).fetchall()
        return {'ok': True, 'jobs': [_public(row) for row in rows]}
    except (ValueError, OSError, sqlite3.Error):
        return _error('invalid_request', 'The job list could not be read.')


def cancel_job(job_id, state_dir=None):
    try:
        if not _id(job_id):
            raise ValueError('id')
        root = _root(state_dir)
        with _db(root) as db:
            row = db.execute('SELECT state,plan FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row and row['state'] == 'running' and json.loads(row['plan']).get('_live') is True:
                return _error('cancel_unsupported', 'A live session job cannot be cancelled from the job queue.')
            db.execute("UPDATE jobs SET state='cancelled',finished=?,reason='cancelled' WHERE id=? AND state='queued'", (time.time(), job_id))
            db.execute("UPDATE jobs SET state='cancel_requested' WHERE id=? AND state='running'", (job_id,))
        return get_job(job_id, root)
    except (ValueError, TypeError, AttributeError, OSError, sqlite3.Error):
        return _error('invalid_job', 'The job or state directory is invalid.')


def job_logs(job_id, limit=4000, state_dir=None):
    try:
        if not _id(job_id) or type(limit) is not int or not 1 <= limit <= 64000:
            raise ValueError('arguments')
        root = _root(state_dir)
        with _db(root) as db:
            row = db.execute('SELECT id FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            return _error('not_found', 'The job was not found.')
        path = root / (job_id + '.log')
        if not path.exists():
            return {'ok': True, 'job_id': job_id, 'text': '', 'redaction': 'best_effort'}
        fd = _owned_file(path)
        with os.fdopen(fd, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            text = stream.read(limit).decode('utf-8', 'replace')
        return {'ok': True, 'job_id': job_id, 'text': _scrub(text), 'redaction': 'best_effort'}
    except (ValueError, TypeError, AttributeError, OSError, sqlite3.Error):
        return _error('invalid_request', 'The job log could not be read.')


def reconcile(state_dir=None):
    try:
        root = _root(state_dir)
        _launch(root)
        return {'ok': True, 'reconciliation': 'scheduled', 'resume_supported': False}
    except (ValueError, OSError, sqlite3.Error):
        return _error('reconcile_failed', 'Job reconciliation could not be scheduled.')
