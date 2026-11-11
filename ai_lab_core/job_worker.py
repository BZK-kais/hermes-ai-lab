"""Detached POSIX singleton worker. Never signal a PID read from the database."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import select
import signal
import stat
import subprocess
import sys
import time

if __package__:
    from . import jobs as _jobs
    from .process_env import runtime_env
else:
    # The detached launcher uses an isolated interpreter and an absolute script path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ai_lab_core import jobs as _jobs
    from ai_lab_core.process_env import runtime_env


def _finish(db, job_id, state, code=None, reason=None, *, plan=None):
    fields = (state, time.time(), code, reason)
    if plan is None:
        db.execute('UPDATE jobs SET state=?,finished=?,exit_code=?,reason=? WHERE id=?', (*fields, job_id))
    else:
        db.execute('UPDATE jobs SET state=?,finished=?,exit_code=?,reason=?,plan=? WHERE id=?',
                   (*fields, json.dumps(plan, allow_nan=False), job_id))
    db.commit()


def _report_receipt(plan):
    if plan.get('report_receipt') is not True:
        return None
    report = Path(plan['metadata']['report_path'])
    if not any(item == {'path': str(report), 'kind': 'file'} for item in plan['outputs']):
        raise ValueError('invalid report receipt')
    if report.resolve(strict=True) != report:
        raise ValueError('invalid report path')
    limit = 4 * 1024**2
    with os.fdopen(os.open(report, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or not 0 < info.st_size <= limit):
            raise ValueError('invalid report file')
        data = stream.read(limit + 1)
    if len(data) != info.st_size:
        raise ValueError('report changed while reading')
    plan['metadata']['report_sha256'] = hashlib.sha256(data).hexdigest()
    return plan


def _kill_owned(proc):
    # Popen remains unreaped until after killpg: its PID cannot have been reused.
    # Do not poll/wait before sending signals to its process group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run(root, db, row, resource_fd):
    job_id = row['id']
    plan = json.loads(row['plan'])
    if _jobs._requires_session(plan):
        _finish(db, job_id, 'failed', reason='session_required')
        return
    fd = _jobs._owned_file(root / (job_id + '.log'))
    proc = None
    deadline = time.monotonic() + plan['timeout_seconds']
    reason = None
    try:
        # Transaction claim already occurred. Cancellation before spawn is safe:
        # it is a durable request and is honored immediately after spawn.
        if db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()[0] == 'cancel_requested':
            _finish(db, job_id, 'cancelled', reason='cancelled')
            return
        output_reason = _jobs._output_path_reason(plan, require_absent=True)
        if output_reason:
            _finish(db, job_id, 'failed', reason=output_reason)
            return
        overrides = plan.get('env', {})
        if plan['operation'] == 'hf.download':
            overrides = {'HF_HUB_OFFLINE': '0', **overrides}
        proc = subprocess.Popen(plan['command'], cwd=plan['cwd'], env=runtime_env(overrides),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True, pass_fds=(resource_fd,))
        os.set_blocking(proc.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        eof = False
        termination_sent = False
        process_events = None
        ended = False
        if not hasattr(os, 'waitid'):
            process_events = select.kqueue()
            try:
                process_events.control([select.kevent(proc.pid, filter=select.KQ_FILTER_PROC, flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT)], 0, 0)
            except ProcessLookupError:
                ended = True
        while True:
            state = db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()[0]
            if state == 'cancel_requested':
                reason = 'cancelled'
            elif time.monotonic() >= deadline:
                reason = 'timeout'
            if reason and not termination_sent:
                _kill_owned(proc)
                termination_sent = True
            for key, _ in selector.select(.05):
                data = os.read(key.fd, 16384)
                if not data:
                    eof = True
                    selector.unregister(key.fileobj)
                else:
                    safe = _jobs._scrub(data.decode('utf-8', 'replace')).encode('utf-8')
                    os.write(fd, safe)
                    if os.lseek(fd, 0, os.SEEK_CUR) > _jobs._MAX_LOG:
                        os.lseek(fd, -(_jobs._MAX_LOG // 2), os.SEEK_END)
                        tail = os.read(fd, _jobs._MAX_LOG // 2)
                        os.lseek(fd, 0, os.SEEK_SET)
                        os.write(fd, tail)
                        os.ftruncate(fd, len(tail))
            # waitid with WNOWAIT detects completion without reaping the PID.
            if hasattr(os, 'waitid'):
                ended = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
            elif not ended:
                ended = bool(process_events.control(None, 1, 0))
            if ended and eof:
                break
            # Drain buffered output after leader exit. A descendant retaining
            # stdout remains bounded by the same deadline and cancellation.
            if reason and eof:
                break
        selector.close()
        if process_events is not None:
            process_events.close()
        code = proc.wait()
        if reason == 'cancelled':
            _finish(db, job_id, 'cancelled', code, reason)
        elif reason:
            _finish(db, job_id, 'failed', code, reason)
        elif code != 0:
            received_plan = None
            if plan['operation'] == 'registry.generate' and not _jobs._output_path_reason(plan):
                try:
                    received_plan = _report_receipt(plan)
                except (OSError, ValueError, RuntimeError):
                    pass
            _finish(db, job_id, 'failed', code, 'exit_nonzero', plan=received_plan)
        elif _jobs._output_path_reason(plan):
            _finish(db, job_id, 'failed', code, 'output_path_invalid')
        elif not _jobs._outputs_present(plan):
            _finish(db, job_id, 'failed', code, 'outputs_missing')
        else:
            try:
                received_plan = _report_receipt(plan)
            except (OSError, ValueError, RuntimeError):
                _finish(db, job_id, 'failed', code, 'report_receipt_failed')
            else:
                _finish(db, job_id, 'succeeded', code, plan=received_plan)
    except Exception:
        if proc is not None:
            try:
                _kill_owned(proc)
                proc.wait()
            except PermissionError:
                _finish(db, job_id, 'interrupted', reason='termination_denied')
                return
        _finish(db, job_id, 'failed', reason='execution_error')
    finally:
        os.close(fd)
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()


def main():
    os.umask(0o077)
    root = _jobs._root(sys.argv[1])
    singleton = _jobs._owned_file(root / 'worker.lock')
    fcntl.flock(singleton, fcntl.LOCK_EX)
    resource = _jobs._owned_file(root / 'execution.lock')
    # A command inherits this lock: abrupt supervisor loss does not immediately
    # admit another job while the ordinary child still owns the descriptor.
    fcntl.flock(resource, fcntl.LOCK_EX)
    with _jobs._db(root) as db:
        db.execute("UPDATE jobs SET state='interrupted',finished=?,reason='worker_lost' WHERE state IN ('running','cancel_requested')", (time.time(),))
        db.commit()
        while True:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                db.commit()
                break
            db.execute("UPDATE jobs SET state='running',started=? WHERE id=? AND state='queued'", (time.time(), row['id']))
            db.commit()
            _run(root, db, row, resource)
    os.close(resource)
    os.close(singleton)


if __name__ == '__main__':
    if sys.argv[1] == '--detach':
        if os.fork():
            os._exit(0)
        os.setsid()
        sys.argv.pop(1)
    main()
