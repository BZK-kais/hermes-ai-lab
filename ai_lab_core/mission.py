"""Compile a declared local mission into a controller campaign without running it."""

import hashlib
import os
from pathlib import Path

from . import controller_contract as contract
from .controller import MAX_REQUEST_BYTES
from .controller_engine import _case_count, _json, _parse_cases, _snapshot
from .controller_measure import MAX_PROMPT_TOKENS, _identity
from .endpoint import failure
from .hardware_profiles import detect_profile
from .integration_config import permitted_path
from .optimization import analyze


_REQUIRED = (
    "dataset_dir", "output_dir", "baseline", "grader", "repetitions",
    "minimum_relative_gain", "quality_noninferiority_tolerance", "max_generated_tokens",
)
_CANDIDATES = ("prompt_candidates", "model_candidates", "adapter_candidates", "prefill_candidates")
_OPTIONAL = (*_CANDIDATES, "minimum_slice_quality", "evaluation_scope")
_BASELINE = ("model", "system_prompt", "input_encoding", "max_tokens", "prefill_step_size")
_GATES = ("max_memory_bytes", "max_p95_latency_ms")
_EXACT_METRICS = {"accuracy", "exact_match", "exact_match_accuracy"}


def _context(request, mission, analysis):
    blockers = list(analysis["blockers"])
    questions = list(analysis["questions"])
    if request.get("evaluation_scope") == "development" and "evidence.fresh_holdout" in blockers:
        blockers.remove("evidence.fresh_holdout")
        if not any(field.startswith("evidence.") for field in blockers):
            questions = [question for question in questions if "fresh holdout" not in question]
    blockers.extend(f"constraints.{field}" for field in _GATES if field not in mission.get("constraints", {}))
    missing = [field for field in _REQUIRED if field not in request]
    baseline = request.get("baseline")
    if type(baseline) is dict:
        missing.extend(f"baseline.{field}" for field in _BASELINE if field not in baseline)
    if not blockers and not missing:
        return None
    if any(f"constraints.{field}" in blockers for field in _GATES):
        questions.append("What limits apply to p95 generation latency and MLX active plus cache memory?")
    if missing:
        questions.append("Which local reference, dataset, exact grader, campaign path, and controller limits should apply?")
    return {"ok": True, "status": "needs_context", "blockers": [*blockers, *missing], "questions": questions}


def _supported(request, mission):
    objective, workload = mission["objective"], mission["workload"]
    constraints = mission["constraints"]
    grader = request["grader"]
    if (grader not in ("json_exact", "exact_text")
            or objective["quality_metric"] not in _EXACT_METRICS | {grader}):
        return failure("unsupported_grader", "Choose json_exact or exact_text and a matching exact accuracy metric.")
    if (mission["task"]["type"] not in ("classification", "extraction", "generation")
            or objective["primary"] not in ("quality", "latency", "memory", "throughput")
            or workload["mode"] not in ("interactive", "batch") or workload["concurrency"] != 1
            or workload.get("shared_prefix") is True
            or workload["input_tokens_p95"] > MAX_PROMPT_TOKENS or workload["output_tokens_p95"] > 128
            or "max_ttft_ms" in constraints or "max_cost_per_request" in constraints
            or mission["budget"]["max_spend"] != 0):
        return failure("unsupported_mission", "Use a sequential local exact-match task with fresh KV caches and zero external spend.")
    return None


def _directory(value, settings):
    contract.check(contract.text(value, 4096) and Path(value).is_absolute())
    path = permitted_path(value, settings)
    contract.check(path.is_dir())
    return str(path)


def _variants(request, settings):
    baseline = request["baseline"]
    contract.check(contract.fields(baseline, _BASELINE, ("adapter", "output_constraint")))
    baseline = {"id": "baseline", **baseline}
    baseline["model"] = _directory(baseline["model"], settings)
    if "adapter" in baseline:
        baseline["adapter"] = _directory(baseline["adapter"], settings)
    variants = [baseline]
    for field in _CANDIDATES:
        values = request.get(field, [])
        contract.check(type(values) is list and len(values) <= 7)
        for index, value in enumerate(values, 1):
            kind = field.removesuffix("_candidates")
            variant = {**baseline, "id": f"{kind}-{index}"}
            if kind == "prompt":
                variant["system_prompt"] = value
            elif kind == "model":
                contract.check(contract.fields(value, ("model",), ("adapter",)))
                variant["model"] = _directory(value["model"], settings)
                contract.check(variant["model"] != baseline["model"])
                variant.pop("adapter", None)
                if "adapter" in value:
                    variant["adapter"] = _directory(value["adapter"], settings)
            elif kind == "adapter":
                if value is None:
                    variant.pop("adapter", None)
                else:
                    variant["adapter"] = _directory(value, settings)
            else:
                variant["prefill_step_size"] = value
            variants.append(variant)
    return variants


def _protocol(request, mission, variants):
    constraints = mission["constraints"]
    budget = mission["budget"]
    protocol = contract.validate({
        "version": contract.VERSION,
        "evaluation_scope": request.get("evaluation_scope", "full"),
        "usage": {"description": mission["task"]["description"], "task_type": mission["task"]["type"]},
        "objective": mission["objective"]["primary"],
        "baseline_id": "baseline",
        "grader": {"kind": request["grader"]},
        "constraints": {
            "quality_min": mission["objective"]["quality_min"],
            "max_mlx_memory_bytes": constraints["max_memory_bytes"],
            "max_p95_latency_ms": constraints["max_p95_latency_ms"],
            "minimum_slice_quality": request.get("minimum_slice_quality", {}),
        },
        "budget": {
            "max_wall_seconds": budget["max_wall_seconds"], "max_trials": budget["max_trials"],
            "max_generated_tokens": request["max_generated_tokens"],
        },
        "repetitions": request["repetitions"],
        "minimum_relative_gain": request["minimum_relative_gain"],
        "quality_noninferiority_tolerance": request["quality_noninferiority_tolerance"],
        "variants": variants,
    })
    configurations = {_json({k: v for k, v in variant.items() if k != "id"}) for variant in variants}
    contract.check(len(configurations) == len(variants))
    return protocol


def _preflight_dataset(directory, protocol):
    raw, provenance = _snapshot(directory)
    dev = _parse_cases(raw["dev.jsonl"], protocol["grader"]["kind"], protocol["constraints"]["minimum_slice_quality"])
    # The base and all one-factor variants share one output cap. Reserve the
    # same warm-up and finalist costs as the controller, without decoding test.
    variants = len(protocol["variants"])
    cases = (len(dev) + 1) * variants
    if protocol["evaluation_scope"] == "full":
        cases += (_case_count(raw["test.jsonl"]) + 1) * min(variants, 2)
    tokens = cases * protocol["repetitions"] * protocol["variants"][0]["max_tokens"]
    return tokens, provenance


def _preflight_paths(request, settings, variants):
    dataset = Path(_directory(request["dataset_dir"], settings))
    workspace = Path(settings["workspace_root"]).resolve(strict=True)
    contract.check(contract.text(request["output_dir"], 4096) and Path(request["output_dir"]).is_absolute())
    output = permitted_path(request["output_dir"], settings, output=True, must_exist=False)
    contract.check(output.parent.is_dir() and not os.path.lexists(output) and not Path(request["output_dir"]).is_symlink())
    sources = {dataset}
    assets = {(variant[key], key == "adapter") for variant in variants for key in ("model", "adapter") if key in variant}
    for value, adapter in assets:
        path = Path(value)
        sources.add(path)
        _, identities = _identity(path, adapter=adapter)
        files = {name for name, identity in identities if identity[2] > 0}
        contract.check(len(files) == len(identities))
        if not adapter:
            contract.check("config.json" in files and any(name.endswith(".safetensors") for name in files))
    contract.check(all(output != source and source not in output.parents and output not in source.parents for source in sources))
    contract.check(workspace.is_dir())
    return dataset, output


def _freeze(output, raw):
    output.mkdir(mode=0o700)
    path = output / "protocol.json"
    created = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        # Remove only a file this invocation opened for writing.
        if created:
            path.unlink(missing_ok=True)
        output.rmdir()
        raise
    return str(path)


def prepare(request, settings):
    mission = request.get("mission", {})
    analysis = analyze(mission, {})
    if not analysis["ok"]:
        return analysis
    hardware = detect_profile()
    analysis = analyze(mission, {"ok": hardware["ok"], "hardware": hardware})
    context = _context(request, mission, analysis)
    if context:
        return context
    if "apple-silicon-mlx" not in hardware["profiles"]:
        return failure("runtime_unavailable", "This campaign requires a local Apple Silicon MLX runtime.")
    unsupported = _supported(request, mission)
    if unsupported:
        return unsupported
    python = Path(settings.get("backend_paths", {}).get("mlx.python", ""))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        return failure("configuration_required", "Configure an executable as backend_paths.mlx.python.")
    variants = _variants(request, settings)
    if len(variants) > min(8, mission["budget"]["max_trials"]):
        return failure("budget_exhausted", "The baseline and declared candidates exceed the trial cap.")
    protocol = _protocol(request, mission, variants)
    if variants[0]["max_tokens"] < mission["workload"]["output_tokens_p95"]:
        return failure("unsupported_mission", "The generation cap must cover the declared output token workload.")
    dataset, output = _preflight_paths(request, settings, variants)
    tokens, provenance = _preflight_dataset(dataset, protocol)
    if tokens > request["max_generated_tokens"]:
        return {**failure("budget_exhausted", "The campaign token ceiling exceeds the declared budget."),
                "generated_token_preflight_ceiling": tokens}
    raw = (_json(protocol) + "\n").encode("utf-8")
    contract.check(len(raw) <= MAX_REQUEST_BYTES)
    protocol_path = _freeze(output, raw)
    return {
        "ok": True, "status": "protocol_ready", "mission_hash": analysis["mission_hash"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(), "protocol_path": protocol_path,
        "start_request": {"action": "start", "protocol_path": protocol_path, "dataset_dir": str(dataset),
                          "output_dir": str(output / "campaign")},
        "variant_ids": [variant["id"] for variant in variants],
        "generated_token_preflight_ceiling": tokens, **provenance,
    }


def dispatch(request, settings):
    from . import autopilot

    if type(request) is dict and request.get("action") in tuple(autopilot.ACTIONS):
        return autopilot.dispatch(request, settings)
    try:
        if contract.fields(request, ("action",), ("mission", *_REQUIRED, *_OPTIONAL)) and request["action"] == "prepare":
            return prepare(request, settings)
    except (KeyError, OSError, TypeError, ValueError, OverflowError, RuntimeError):
        pass
    return failure("invalid_mission", "Supply a valid local mission, controller settings and permitted paths.")
