"""Durable local jobs. Plans are trusted internal inputs, never an API shell surface.

SQLite is shared directly by clients and detached workers; no socket service is
required. Log redaction is best effort, not a guarantee of secret-free output.
"""
from contextlib import contextmanager
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
        if other.exists() or other.is_symlink():
            info = other.lstat()
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


def _scrub(text, secrets=()):
    for value in secrets:
        if value and len(value) >= 4:
            text = text.replace(value, '[REDACTED]')
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
    return {k: row[k] for k in ('id', 'state', 'created', 'started', 'finished', 'exit_code', 'reason')} | {
        'operation': p['operation'], 'backend': p['backend'], 'metadata': _clean(p.get('metadata', {}))}


def _validate(plan):
    if not isinstance(plan, dict):
        raise ValueError('plan')
    if not all(isinstance(plan.get(k), str) and 0 < len(plan[k]) <= 100 and re.fullmatch(r'[A-Za-z0-9_.-]+', plan[k]) for k in ('operation', 'backend')):
        raise ValueError('identity')
    cmd = plan.get('command')
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and '\x00' not in x for x in cmd):
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
    env = plan.get('env', {})
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and '=' not in k and '\x00' not in k + v for k, v in env.items()):
        raise ValueError('env')
    # Preserve the intended workspace before a queued job can observe changed
    # symlink components. This field is internal and never trusted from callers.
    encoded_plan = dict(plan)
    encoded_plan['_output_workspace'] = str(Path(plan['cwd']).resolve(strict=True))
    return json.dumps(encoded_plan, allow_nan=False)


def _launch(root):
    # Execute a filesystem script: works for ordinary and Hermes-namespaced imports.
    launcher = subprocess.Popen([sys.executable, str(Path(__file__).with_name('job_worker.py')), '--detach', str(root)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)
    if launcher.wait(timeout=10) != 0:
        raise OSError('worker launch failed')


def submit(plan, state_dir=None):
    try:
        encoded = _validate(plan)
        root = _root(state_dir)
        job_id = str(uuid.uuid4())
        with _db(root) as db:
            db.execute('INSERT INTO jobs(id,state,plan,created) VALUES(?,?,?,?)', (job_id, 'queued', encoded, time.time()))
        _launch(root)
        return get_job(job_id, root)
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
            row = db.execute('SELECT plan FROM jobs WHERE id=?', (job_id,)).fetchone()
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
        env = json.loads(row['plan']).get('env', {})
        return {'ok': True, 'job_id': job_id, 'text': _scrub(text, [v for k, v in env.items() if _SECRET.search(k)]), 'redaction': 'best_effort'}
    except (ValueError, TypeError, AttributeError, OSError, sqlite3.Error):
        return _error('invalid_request', 'The job log could not be read.')


def reconcile(state_dir=None):
    try:
        root = _root(state_dir)
        _launch(root)
        return {'ok': True, 'reconciliation': 'scheduled', 'resume_supported': False}
    except (ValueError, OSError, sqlite3.Error):
        return _error('reconcile_failed', 'Job reconciliation could not be scheduled.')
