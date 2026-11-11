"""Isolated MLX worker for one local controller campaign."""
import fcntl
import hashlib
import hmac
from importlib.metadata import version
import json
import os
from pathlib import Path
import stat
import sys
from time import monotonic

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_lab_core.controller_contract import VERSION, sha256, validate
from ai_lab_core.autopilot_runtime import SOURCE_NAMES, consume, fingerprint, runtime_identity
from ai_lab_core.controller_measure import MLXMeasure, PinnedAssets
from ai_lab_core.hardware_profiles import detect_profile
from ai_lab_core.inference_worker import _write_report
from ai_lab_core.profile import device_lock_path

MAX_REQUEST_BYTES = 512 * 1024
DATA_SOURCES = {
    "manifest": ("manifest.sha256", 1024),
    "dev": ("dev.jsonl", 16 * 1024 * 1024),
    "test": ("test.jsonl", 16 * 1024 * 1024),
}
ERRORS = {
    "invalid_request": "The controller request could not be verified.",
    "integrity_error": "A controller source does not match the submitted campaign.",
    "compute_busy": "The local MLX device is busy.",
    "runtime_unavailable": "The controller requires a local Apple Silicon MLX Metal runtime.",
    "runtime_failed": "The local controller campaign encountered a technical failure.",
}


def _failure(code):
    return {
        "kind": VERSION, "ok": False, "status": code,
        "accepted_variant_id": None, "activation_performed": False, "optimization_gain_verified": False,
        "error": {"code": code, "message": ERRORS[code]},
    }


def _private_directory(path, workspace):
    info = path.stat()
    return (
        path.is_absolute() and workspace in path.parents and path.resolve(strict=True) == path
        and stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077
    )


def _read_request(path, expected, workspace):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                or info.st_size > MAX_REQUEST_BYTES):
            raise ValueError("invalid request file")
        data = stream.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected):
        raise ValueError("invalid request digest")
    payload = json.loads(data)
    if (type(payload) is not dict
            or payload.keys() - {"autonomy"} != {
                "workspace", "protocol_path", "protocol", "dataset_dir", "output_dir", "source_sha256",
            }
            or payload["workspace"] != str(workspace) or payload["output_dir"] != str(path.parent)):
        raise ValueError("invalid request workspace")
    protocol_path, dataset_dir = Path(payload["protocol_path"]), Path(payload["dataset_dir"])
    if (protocol_path.resolve(strict=True) != protocol_path or not protocol_path.is_file()
            or dataset_dir.resolve(strict=True) != dataset_dir or not dataset_dir.is_dir()):
        raise ValueError("invalid source paths")
    expected_sources = payload["source_sha256"]
    if (type(expected_sources) is not dict or expected_sources.keys() != {"protocol", *DATA_SOURCES}
            or not all(sha256(value) for value in expected_sources.values())):
        raise ValueError("invalid source digests")
    payload["protocol"] = validate(payload["protocol"])
    if "autonomy" in payload:
        autonomy = payload["autonomy"]
        if (type(autonomy) is not dict or autonomy.keys() != {"profile_state", "runtime_fingerprint"}
                or not Path(autonomy["profile_state"]).is_absolute() or not sha256(autonomy["runtime_fingerprint"])):
            raise ValueError("invalid autonomous request")
    return payload


def _file_sha(path, *, limit=None):
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or (limit is not None and info.st_size > limit):
            raise ValueError("invalid source file")
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            size += len(chunk)
            if limit is not None and size > limit:
                raise ValueError("source is too large")
            digest.update(chunk)
    return digest.hexdigest()


def _verify_request_sources(payload):
    protocol_path, dataset_dir = Path(payload["protocol_path"]), Path(payload["dataset_dir"])
    expected = payload["source_sha256"]
    if (_file_sha(protocol_path, limit=MAX_REQUEST_BYTES) != expected["protocol"]
            or any(_file_sha(dataset_dir / filename, limit=limit) != expected[key]
                   for key, (filename, limit) in DATA_SOURCES.items())):
        raise ValueError("controller inputs changed")


def _execute(payload, request_sha, progress):
    from ai_lab_core.controller_engine import run

    lock = device_lock_path()
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _failure("compute_busy")

        protocol = payload["protocol"]
        deadline = monotonic() + protocol["budget"]["max_wall_seconds"]
        try:
            _verify_request_sources(payload)
        except (OSError, ValueError):
            return _failure("integrity_error")
        hardware = detect_profile()
        if "apple-silicon-mlx" not in hardware["profiles"]:
            return _failure("runtime_unavailable")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import mlx.core as mx

            if not mx.metal.is_available():
                return _failure("runtime_unavailable")
            mx.set_default_device(mx.gpu)
            from mlx_lm import load, stream_generate
        except (ImportError, OSError, RuntimeError):
            return _failure("runtime_unavailable")

        source_dir = Path(__file__).resolve().parent
        source_hashes = {name: _file_sha(source_dir / name) for name in SOURCE_NAMES}
        before_holdout = None
        if "autonomy" in payload:
            autonomy = payload["autonomy"]
            observed = fingerprint(runtime_identity(), hardware, source_hashes)
            if observed != autonomy["runtime_fingerprint"]:
                return _failure("integrity_error")
            hardware["autonomy_runtime_sha256"] = observed
            if protocol.get("evaluation_scope", "full") == "full":
                def before_holdout():
                    consume(autonomy["profile_state"], payload["source_sha256"]["test"], request_sha)

        def verify():
            _verify_request_sources(payload)
            if any(_file_sha(source_dir / name) != digest for name, digest in source_hashes.items()):
                raise ValueError("controller source changed")

        hardware.update({
            "metal_available": True,
            "mlx_version": version("mlx"),
            "mlx_lm_version": version("mlx-lm"),
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "request_sha256": request_sha,
            "protocol_source_sha256": payload["source_sha256"]["protocol"],
            "source_data_sha256": {key: payload["source_sha256"][key] for key in DATA_SOURCES},
            "source_sha256": source_hashes,
            "measurement_scope": {
                "generation": "tokenization, loading, warmup and grading excluded",
                "ttft": "first streamed token event, including immediate EOS",
                "tokens": "sampled tokens include EOS and warmup counts toward the campaign budget",
                "kv_cache": "fresh per generation",
                "mlx_memory": "trial peak active and sampled cache include load and warmup",
                "process_rss": "cumulative process high-water mark, not a ranking metric",
                "file_cache": "uncontrolled and warmed by fingerprinting",
            },
        })
        assets = PinnedAssets()
        measure = MLXMeasure(mx, load, stream_generate, assets)

        return run(
            protocol, Path(payload["dataset_dir"]), Path(payload["output_dir"]),
            measure, hardware, progress, identify=assets.identify, deadline=deadline, verify=verify,
            before_holdout=before_holdout,
        )


def main():
    # Only the explicit progress writer may access the original stdout pipe.
    with os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1) as progress_stream:
        with open(os.devnull, "wb") as discard:
            os.dup2(discard.fileno(), 1)
            os.dup2(discard.fileno(), 2)
        if len(sys.argv) != 3:
            return 1
        path = Path(sys.argv[1])
        workspace = Path.cwd().resolve()

        def progress(event):
            print(json.dumps(event, separators=(",", ":"), allow_nan=False), file=progress_stream)

        try:
            if path.name != "request.json" or not _private_directory(path.parent, workspace):
                return 1
            report_path = path.parent / "report.json"
            if os.path.lexists(report_path):
                return 1
            try:
                payload = _read_request(path, sys.argv[2], workspace)
            except (KeyError, OSError, TypeError, ValueError, RecursionError):
                report = _failure("invalid_request")
            else:
                try:
                    report = _execute(payload, sys.argv[2], progress)
                except Exception:
                    report = _failure("runtime_failed")
            _write_report(report_path, report)
            return 0 if report["ok"] else 1
        except (OSError, ValueError):
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
