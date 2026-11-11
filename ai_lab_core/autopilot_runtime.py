"""Shared runtime identity and one-time holdout receipt for autonomous jobs."""
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ai_lab_core.controller_contract import sha256
    from ai_lab_core.controller_engine import _json
else:
    from .controller_contract import sha256
    from .controller_engine import _json


SOURCE_NAMES = (
    "controller_contract.py", "controller_engine.py", "controller_measure.py",
    "controller_worker.py", "hardware_profiles.py", "inference_worker.py",
    "skill_runtime.py", "skill_worker.py", "json_constraints.py", "autopilot_runtime.py",
)


def runtime_identity():
    packages = {}
    for name in ("mlx", "mlx-lm"):
        dist = distribution(name)
        record = dist.read_text("RECORD") or ""
        packages[name] = [dist.version, hashlib.sha256(record.encode()).hexdigest(), str(dist.locate_file(""))]
    executable = Path(sys.executable).resolve(strict=True)
    info = executable.stat()
    return {"python": sys.version, "executable": str(executable), "executable_stat": [
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    ], "packages": packages, "system_release": platform.release()}


def fingerprint(runtime, hardware, sources):
    facts = {key: hardware.get(key) for key in (
        "system", "cpu_architecture", "profiles", "hardware", "devices", "memory_kind",
    )}
    return hashlib.sha256(_json({"runtime": runtime, "hardware": facts, "sources": sources}).encode()).hexdigest()


def probe(python, hardware):
    command = [str(python), "-I", "-B", str(Path(__file__).resolve())]
    try:
        result = subprocess.run(command, capture_output=True, timeout=5, check=True)
        if len(result.stdout) > 16_384:
            return None
        runtime = json.loads(result.stdout)
        if type(runtime) is not dict:
            return None
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
        return fingerprint(runtime, hardware, sources)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _private(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or path.resolve(strict=True) != path):
        raise ValueError("unsafe autonomous state directory")


def _marker(state, digest):
    if not sha256(digest):
        raise ValueError("invalid holdout hash")
    return Path(state) / "autonomous-holdouts-v1" / digest


def consumed(state, digest):
    path = _marker(state, digest)
    if not os.path.lexists(path.parent):
        return False
    _private(path.parent.parent)
    _private(path.parent)
    return os.path.lexists(path)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def consume(state, digest, request_sha):
    """Irrevocably reserve this hash before decoding final cases, even on failure."""
    path = _marker(state, digest)
    _private(path.parent.parent)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    _private(path.parent)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write((_json({"test_sha256": digest, "request_sha256": request_sha}) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)
    _sync_dir(path.parent.parent)


if __name__ == "__main__":
    print(_json(runtime_identity()))
