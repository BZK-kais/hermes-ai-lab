"""Wait for the per-user MLX lock, then run one named backend command."""
import fcntl
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_lab_core.jobs import _owned_file
from ai_lab_core.profile import device_lock_path


def main(argv):
    try:
        if not argv or not Path(argv[0]).is_absolute() or any('\0' in arg for arg in argv):
            raise ValueError('invalid command')
        lock = device_lock_path()
        lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with os.fdopen(_owned_file(lock), 'r+b') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            # The job worker owns the timeout and process group. Exec preserves
            # the PID, streams, exit status and inherited job-worker lock.
            os.set_inheritable(stream.fileno(), True)
            os.execv(argv[0], argv)
    except (OSError, TypeError, ValueError):
        print('The local MLX command could not start.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
