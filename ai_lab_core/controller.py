"""Profile-bound submission and compact reporting for finite MLX searches."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from . import jobs
from .controller_contract import VERSION, name, sha256, validate
from .endpoint import failure
from .integration_config import permitted_path

MAX_REQUEST_BYTES = 512 * 1024
MAX_REPORT_BYTES = 4 * 1024**2
SUCCESS = {"qualified_candidate", "retained_baseline", "no_qualified_configuration", "development_complete"}
ERRORS = {
    "incomplete", "invalid_request", "compute_busy", "runtime_unavailable", "runtime_failed",
    "integrity_error", "budget_exhausted", "measurement_error", "cancelled",
    "holdout_consumed",
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("nonfinite constant")


def _read(path, limit, *, owned=False):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size > limit
                or (owned and info.st_uid != os.getuid())):
            raise ValueError("file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("file size")
    return raw


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def _start(request, settings, *, autonomy=None):
    if set(request) != {"action", "protocol_path", "dataset_dir", "output_dir"}:
        raise ValueError("arguments")
    workspace = Path(settings["workspace_root"]).resolve()
    protocol_path = permitted_path(request["protocol_path"], settings)
    dataset = permitted_path(request["dataset_dir"], settings)
    output = permitted_path(request["output_dir"], settings, output=True, must_exist=False)
    if (not workspace.is_dir() or not dataset.is_dir() or not output.parent.is_dir()
            or os.path.lexists(output) or Path(request["output_dir"]).is_symlink()):
        raise ValueError("directories")
    protocol_raw = _read(protocol_path, MAX_REQUEST_BYTES)
    protocol = validate(_json(protocol_raw))
    for variant in protocol["variants"]:
        for field in ("model", "adapter"):
            if field in variant:
                path = permitted_path(variant[field], settings)
                if not path.is_dir() or path == output or path in output.parents or output in path.parents:
                    raise ValueError("model or adapter")
                variant[field] = str(path)
    source_sha256 = {"protocol": hashlib.sha256(protocol_raw).hexdigest()}
    for field, filename, limit in (
        ("manifest", "manifest.sha256", 1024),
        ("dev", "dev.jsonl", 16 * 1024**2),
        ("test", "test.jsonl", 16 * 1024**2),
    ):
        path = permitted_path(str(dataset / filename), settings)
        # Keep dataset paths literal: resolving a file symlink would let it
        # differ from the worker's pinned dataset directory.
        if path != dataset / filename:
            raise ValueError("dataset symlink")
        source_sha256[field] = hashlib.sha256(_read(path, limit)).hexdigest()
    python = Path(settings.get("backend_paths", {}).get("mlx.python", ""))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        return failure("configuration_required", "Configure an executable as backend_paths.mlx.python.")
    payload = {
        "workspace": str(workspace), "protocol_path": str(protocol_path), "protocol": protocol,
        "dataset_dir": str(dataset), "output_dir": str(output), "source_sha256": source_sha256,
    }
    if autonomy is not None:
        if source_sha256 != autonomy["source_sha256"]:
            return failure("integrity_error", "The dataset or protocol changed after autonomous planning.")
        payload["autonomy"] = {"profile_state": settings["state_dir"], "runtime_fingerprint": autonomy["runtime_fingerprint"]}
    raw = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request size")
    output.mkdir(mode=0o700)
    request_path = output / "request.json"
    with os.fdopen(os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)
    report_path = output / "report.json"
    request_sha256 = hashlib.sha256(raw).hexdigest()
    return jobs.submit({
        "operation": "optimization.search", "backend": "mlx", "cwd": str(workspace),
        "command": [str(python), "-I", "-B", str(Path(__file__).with_name("controller_worker.py")),
                    str(request_path), request_sha256],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "timeout_seconds": protocol["budget"]["max_wall_seconds"],
        "outputs": [{"path": str(report_path), "kind": "file"}],
        "report_receipt": True,
        "metadata": {"profile_state": settings["state_dir"], "report_path": str(report_path),
                     **({"autonomy": True} if autonomy is not None else {}),
                     "request_sha256": request_sha256, "cloud": False},
    }, state_dir=settings["state_dir"])


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("metric")
    return value


def _summary(split):
    if type(split) is not dict or len(split) > 8 or any(not name(key) for key in split):
        raise ValueError("split")
    output = {}
    for identifier, result in split.items():
        safe = {}
        for field in ("complete", "eligible"):
            if type(result[field]) is not bool:
                raise ValueError("flag")
            safe[field] = result[field]
        gates = result["failed_gates"]
        if type(gates) is not list or len(gates) > 36 or any(
            gate not in {"quality", "latency", "memory", "incomplete_measurement"}
            and not (type(gate) is str and gate.startswith("slice:") and name(gate[6:])) for gate in gates
        ):
            raise ValueError("gate")
        safe["failed_gates"] = gates
        if result["complete"]:
            for field in ("quality", "p95_latency_ms", "mlx_active_plus_cache_bound_bytes", "requests_per_second"):
                safe[field] = _number(result[field])
            slices = result["slice_quality"]
            if type(slices) is not dict or len(slices) > 256 or any(not name(key) for key in slices):
                raise ValueError("slices")
            safe["slice_quality"] = {key: _number(value) for key, value in slices.items()}
            if safe["quality"] > 1 or any(value > 1 for value in safe["slice_quality"].values()):
                raise ValueError("quality range")
            for field in ("output_parseable_fraction", "output_schema_matches_fraction", "constraint_verified_fraction"):
                if field in result:
                    safe[field] = _number(result[field])
                    if safe[field] > 1:
                        raise ValueError("diagnostic range")
            schema = safe.get("output_schema_matches_fraction")
            parseable = safe.get("output_parseable_fraction")
            if schema is not None and parseable is not None and not safe["quality"] <= schema <= parseable:
                raise ValueError("diagnostic order")
        output[identifier] = safe
    return output


def _report(request, settings):
    if set(request) != {"action", "report_path"}:
        raise ValueError("arguments")
    path = permitted_path(request["report_path"], settings, output=True)
    raw = _read(path, MAX_REPORT_BYTES, owned=True)
    report = _json(raw)
    if type(report) is not dict or report.get("kind") != VERSION or type(report.get("ok")) is not bool:
        raise ValueError("report")
    status = report.get("status")
    if report["ok"] is False:
        if status not in ERRORS:
            raise ValueError("status")
        return {"kind": VERSION, "status": status, "report_path": str(path),
                **failure(status, "The controller did not complete. Check the durable job state.")}
    if status not in SUCCESS or report.get("activation_performed") is not False:
        raise ValueError("status")
    safe = {"ok": True, "kind": VERSION, "status": status, "report_path": str(path),
            "report_sha256": hashlib.sha256(raw).hexdigest(), "activation_performed": False}
    scope = report.get("evaluation_scope", "full")
    development_only = status == "development_complete"
    if scope != ("development" if development_only else "full"):
        raise ValueError("evaluation scope")
    if "evaluation_scope" in report:
        safe["evaluation_scope"] = scope
    for field in ("baseline_variant_id", "selected_variant_id", "accepted_variant_id"):
        value = report[field]
        if value is not None and not name(value):
            raise ValueError("identifier")
        safe[field] = value
    baseline, selected, accepted = (safe[field] for field in (
        "baseline_variant_id", "selected_variant_id", "accepted_variant_id",
    ))
    basis = report.get("selection_basis", "development")
    if basis not in ("development", "predeclared_pair"):
        raise ValueError("selection basis")
    safe["selection_basis"] = basis
    if basis == "predeclared_pair":
        declared = report.get("holdout_candidate_id")
        if not name(declared) or declared != selected or declared == baseline:
            raise ValueError("predeclared candidate")
        safe["holdout_candidate_id"] = declared
    if (baseline is None
            or (status in ("no_qualified_configuration", "development_complete") and accepted is not None)
            or (status == "retained_baseline" and accepted != baseline)
            or (status == "qualified_candidate" and (accepted is None or accepted != selected or accepted == baseline))
            or (status != "qualified_candidate" and report.get("optimization_gain_verified") is not False)
            or type(report.get("generated_tokens_accounted")) is not int):
        raise ValueError("decision")
    if type(report["optimization_gain_verified"]) is not bool or not sha256(report["selection_sha256"]):
        raise ValueError("result")
    safe.update(optimization_gain_verified=report["optimization_gain_verified"],
                selection_sha256=report["selection_sha256"],
                elapsed_seconds=_number(report["elapsed_seconds"]),
                generated_tokens_accounted=_number(report["generated_tokens_accounted"]))
    for split in ("dev", "test"):
        safe[split] = _summary(report[split])
    if development_only:
        dev = safe["dev"]
        eligible = {identifier for identifier in dev if _eligible(dev, identifier)}
        if (basis != "development" or "holdout_candidate_id" in report or safe["test"]
                or baseline not in dev or not all(result["complete"] for result in dev.values())
                or (selected not in eligible if eligible else selected is not None)):
            raise ValueError("development evidence")
        return safe
    finalists = {baseline, selected} if selected is not None else set()
    if (set(safe["test"]) != finalists
            or (basis == "development" and selected is not None and not _eligible(safe["dev"], selected))
            or (basis == "predeclared_pair" and not finalists <= set(safe["dev"]))
            or (accepted is not None and not _eligible(safe["test"], accepted))
            or (basis == "predeclared_pair" and accepted is not None and not _eligible(safe["dev"], accepted))):
        raise ValueError("decision evidence")
    return safe


def _eligible(split, identifier):
    result = split.get(identifier)
    return result is not None and result["complete"] and result["eligible"]


def dispatch(request, settings):
    try:
        if type(request) is dict:
            if request.get("action") == "start":
                return _start(request, settings)
            if request.get("action") == "report":
                return _report(request, settings)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    return failure("invalid_optimization", "Supply a frozen local protocol and dataset, or a controller report inside this profile workspace.")
