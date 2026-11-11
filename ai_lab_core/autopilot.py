"""Compile a bounded local optimization mission and launch the existing controller."""
from copy import deepcopy
import os
from pathlib import Path
import re
import sqlite3
import uuid

from . import autopilot_memory as memory, autopilot_runtime as runtime, controller, mission, registry
from . import controller_contract as contract
from .controller_engine import _case_count, _json, _parse_cases, _snapshot
from .controller_measure import MAX_PROMPT_TOKENS, PinnedAssets
from .endpoint import failure
from .json_constraints import canonical_choices
from .optimization import analyze


ACTIONS = {"autoplan", "autostart", "autoreport", "lessons"}
_MIA = "https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold"
_REVISION = "0a5297c912e118490b8bb3734d9e20107df7f6f2"
_PREFILL = _MIA + "/blob/" + _REVISION + "/CHANGELOG.md"
_MLX = "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py"
_EXCLUSIONS = (
    {"recipe": "tensorfold_prefill_scheduling", "reason": "This controller runs sequentially and cannot interleave requests.", "source": _PREFILL},
    {"recipe": "speculative_decoding", "reason": "The controller has no evaluated draft-model route.", "source": _MIA + "/blob/" + _REVISION + "/README.md"},
    {"recipe": "kv_sharding", "reason": "The controller uses fresh local KV caches on one device.", "source": _MIA + "/commit/" + _REVISION},
    {"recipe": "moe_top_k", "reason": "Router changes require a supported implementation and semantic verification.", "source": "https://arxiv.org/html/2609.04575v1"},
    {"recipe": "new_quantization", "reason": "This campaign can compare declared local artifacts; it does not quantize or train weights.", "source": "https://mia-ai.net/eaq"},
)


def _needs(request):
    spec = request.get("mission", {})
    base = request.get("baseline", {})
    contract.check(type(spec) is dict and type(base) is dict)
    missing = [key for key, present in (
        ("task", "task" in request or "description" in spec.get("task", {})),
        ("objective", "objective" in request or "primary" in spec.get("objective", {})),
        ("model", "model" in request or "model" in base),
        ("dataset_dir", "dataset_dir" in request),
    ) if not present]
    return {"ok": True, "status": "needs_context", "blockers": missing} if missing else None


def _memory_limit(hardware):
    physical = hardware.get("hardware", {}).get("memory")
    if type(physical) is str:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|TB|TiB)", physical, re.IGNORECASE)
        physical = int(float(match[1]) * (2**40 if match[2].lower().startswith("t") else 2**30)) if match else None
    return physical * 3 // 4 if type(physical) is int and 0 < physical <= 2**60 else 4 * 2**30


def _normalize(request, raw, output, hardware):
    declared = deepcopy(request.get("mission", {}))
    analysis = analyze(declared, {})
    if not analysis["ok"]:
        return analysis
    task = declared.setdefault("task", {})
    objective = declared.setdefault("objective", {})
    workload = declared.setdefault("workload", {})
    gates = declared.setdefault("constraints", {})
    budget = declared.setdefault("budget", {})
    evidence = declared.get("evidence", {})
    if "task" in request:
        contract.check(not task)
        given = request["task"]
        if type(given) is str:
            task.update(description=given, type="generation")
        else:
            contract.check(contract.fields(given, ("description", "type"), ("language",)))
            task.update(given)
    if "objective" in request:
        contract.check("primary" not in objective)
        objective["primary"] = request["objective"]
    task.setdefault("type", "generation")
    task.setdefault("language", "und")
    baseline = deepcopy(request.get("baseline", {}))
    if "model" in request:
        contract.check("model" not in baseline or baseline["model"] == request["model"])
        baseline["model"] = request["model"]
    choices = None
    if "output_choices" in request:
        choices = {"kind": "json_choices", "values": request["output_choices"]}
        canonical_choices(choices)
    dev = _parse_cases(raw["dev.jsonl"], "json_exact", request.get("minimum_slice_quality", {}))
    metric = objective.get("quality_metric")
    grader = request.get("grader")
    if grader is None:
        if metric in ("json_exact", "exact_text"):
            grader = metric
        elif choices is not None or "output_constraint" in baseline or any(type(row["expected"]) is not str for row in dev):
            grader = "json_exact"
        else:
            grader = "exact_text"
    contract.check(choices is None or grader == "json_exact")
    objective.setdefault("quality_metric", grader)
    blockers = []
    if "quality_min" not in objective:
        blockers.append("mission.objective.quality_min")
    blockers.extend(f"mission.constraints.{field}" for field in mission._GATES if field not in gates)
    if (evidence.get("consumed_holdout") is not False
            or not all(key in evidence for key in ("evaluator_ref", "representative_inputs_ref"))):
        blockers.append("mission.evidence.fresh_holdout")

    defaults = {}

    def set_default(group, group_name, key, value):
        if key not in group:
            group[key] = value
            defaults[f"{group_name}{key}"] = value if key != "system_prompt" else "task description"

    set_default(objective, "objective.", "quality_min", 0)
    set_default(gates, "constraints.", "max_memory_bytes", _memory_limit(hardware))
    set_default(gates, "constraints.", "max_p95_latency_ms", 5_000)
    set_default(gates, "constraints.", "data_policy", "local_only")
    for key, value in {"mode": "interactive", "concurrency": 1, "input_tokens_p95": MAX_PROMPT_TOKENS,
                       "output_tokens_p95": baseline.get("max_tokens", 32)}.items():
        set_default(workload, "workload.", key, value)
    for key, value in {"max_trials": 4, "max_wall_seconds": 300, "max_spend": 0}.items():
        set_default(budget, "budget.", key, value)
    for key, value in {"system_prompt": task["description"], "input_encoding": "plain",
                       "max_tokens": max(32, workload["output_tokens_p95"]), "prefill_step_size": 256}.items():
        set_default(baseline, "baseline.", key, value)
    normalized = {**request, "baseline": baseline, "grader": grader, "output_dir": output}
    if "grader" not in request:
        defaults["grader"] = grader
    for key, value in {"repetitions": 2, "minimum_relative_gain": 0.05,
                       "quality_noninferiority_tolerance": 0, "max_generated_tokens": 32_768}.items():
        set_default(normalized, "", key, value)
    analysis = analyze(declared, {})
    if not analysis["ok"]:
        return analysis
    unsupported = mission._supported(normalized, declared)
    if unsupported:
        return unsupported
    if gates["data_policy"] != "local_only" or baseline["max_tokens"] < workload["output_tokens_p95"]:
        return failure("unsupported_mission", "Use a local data policy and an output cap covering the declared workload.")
    _parse_cases(raw["dev.jsonl"], grader, request.get("minimum_slice_quality", {}))
    return normalized, declared, choices, defaults, blockers, len(dev)


def _preflight_cost(protocol, dev_count, test_count, variants):
    cases = (dev_count + 1) * len(variants)
    if test_count is not None:
        cases += (test_count + 1) * min(len(variants), 2)
    return cases * protocol["repetitions"] * variants[0]["max_tokens"]


def _automatic(normalized, spec, baseline):
    primary = spec["objective"]["primary"]
    if primary in ("latency", "throughput", "memory") and "prefill_candidates" not in normalized:
        steps = (64, 2048, 256) if primary == "memory" else (2048, 64, 256)
        return [({**baseline, "id": f"prefill-{step}", "prefill_step_size": step}, False,
                 "Measure an MLX prefill chunk size with identical weights, prompt and decoding. Mia's scheduling result is a hypothesis source for isolating prefill, not a transferable setting.",
                 [_PREFILL, _MLX]) for step in steps if step != baseline["prefill_step_size"]]
    if primary == "quality" and "prompt_candidates" not in normalized:
        hint = "Return only the requested JSON value, without markdown or commentary." if normalized["grader"] == "json_exact" else "Return only the requested answer, without commentary."
        prompt = baseline["system_prompt"] + "\n\n" + hint
        if contract.text(prompt, contract.MAX_INPUT_BYTES):
            return [({**baseline, "id": "prompt-exact", "system_prompt": prompt}, False,
                     "Test a stricter answer format without inferring any labels from evaluation data.", "controller:exact_grader")]
    return []


def plan(request, settings, *, start=False):
    missing = _needs(request)
    if missing:
        return missing
    scope = request.get("evaluation_scope")
    contract.check(scope in (None, "full", "development"))
    dataset = Path(mission._directory(request["dataset_dir"], settings))
    raw, provenance = _snapshot(dataset)
    output = request.get("output_dir", str(Path(settings["workspace_root"]) / ("autonomy-" + uuid.uuid4().hex)))
    hardware = mission.detect_profile()
    result = _normalize(request, raw, output, hardware)
    if type(result) is dict:
        return result
    normalized, spec, choices, defaults, blockers, dev_count = result
    data = {"manifest": provenance["dataset_manifest_sha256"],
            **{key: provenance["dataset_sha256"][key + ".jsonl"] for key in ("dev", "test")}}
    used = runtime.consumed(settings["state_dir"], data["test"])
    if used:
        blockers.append("holdout.already_consumed")
    if used and scope != "development" and blockers == ["holdout.already_consumed"] and start:
        return failure("holdout_consumed", "This final test was already consumed in this profile. Use a fresh split or explicitly run development only.")
    scope = "development" if blockers or scope == "development" else "full"
    normalized["evaluation_scope"] = scope
    test_count = _case_count(raw["test.jsonl"]) if scope == "full" else None
    raw = None
    variants = mission._variants(normalized, settings)
    baseline = variants[0]
    contract.check("output_constraint" not in baseline or normalized["grader"] == "json_exact")
    protocol = mission._protocol(normalized, spec, [baseline])
    python = Path(settings.get("backend_paths", {}).get("mlx.python", ""))
    runtime_fingerprint = (runtime.probe(python, hardware) if "apple-silicon-mlx" in hardware["profiles"]
                           and python.is_absolute() and python.is_file() and os.access(python, os.X_OK) else None)
    if start and runtime_fingerprint is None:
        return failure("runtime_unavailable", "Configure a local Apple Silicon MLX Python runtime to start the planned search.")
    candidates = [(v, True, "Caller declared this local candidate; model and template compatibility are checked in the worker.", "caller")
                  for v in variants[1:]]
    if choices is not None:
        candidates.append(({**baseline, "id": "constraint-1", "output_constraint": choices}, False,
                           "Compare exact finite JSON responses supplied explicitly by the caller.", "controller:json_choices"))
    candidates.extend(_automatic(normalized, spec, baseline))
    dataset, output = mission._preflight_paths(normalized, settings, [baseline, *(row[0] for row in candidates)])
    rejected = {}
    assets = PinnedAssets()
    previous = memory.reconcile(settings) if runtime_fingerprint else []
    if previous and candidates:
        base_key = memory.variant_key(baseline, assets.identify(baseline))
        context = memory.context_key(protocol, data, runtime_fingerprint, base_key)
        for entry in previous:
            if entry["context_sha256"] != context:
                continue
            for candidate in entry["candidates"]:
                rejected.setdefault(candidate["variant_key"], (candidate["outcome"], entry["job_id"]))
    chosen = [baseline]
    decisions = [{"variant_id": "baseline", "decision": "trial", "reason": "Always measure the reference.", "source": "caller"}]
    seen = {_json({k: v for k, v in baseline.items() if k != "id"})}
    cap = protocol["budget"]["max_generated_tokens"]
    ceiling = _preflight_cost(protocol, dev_count, test_count, chosen)
    if ceiling > cap:
        return {**failure("budget_exhausted", "Even the baseline cannot fit in the declared generation budget."),
                "generated_token_preflight_ceiling": ceiling}
    for variant, declared, reason, source in candidates:
        key = _json({k: v for k, v in variant.items() if k != "id"})
        decision = {"variant_id": variant["id"], "decision": "trial", "reason": reason, "source": source}
        if key in seen:
            decision.update(decision="excluded_duplicate", reason="Identical to the reference or an earlier candidate.")
        else:
            checked = deepcopy(protocol)
            checked["budget"]["max_trials"] = max(2, checked["budget"]["max_trials"])
            checked["variants"].append(variant)
            contract.validate(checked)
            prior = rejected.get(memory.variant_key(variant, assets.identify(variant))) if rejected else None
            cost = _preflight_cost(protocol, dev_count, test_count, [*chosen, variant])
            if prior and prior[0] in memory.SKIP:
                decision.update(decision="excluded_by_lesson", reason=prior[0], evidence_job_id=prior[1])
            elif len(chosen) >= protocol["budget"]["max_trials"] or cost > cap:
                if declared:
                    return failure("budget_exhausted", "Declared candidates exceed the trial or generation budget.")
                decision.update(decision="excluded_by_budget", reason="The trial would exceed the declared trial or token cap.")
            else:
                chosen.append(variant)
                ceiling = cost
            seen.add(key)
        decisions.append(decision)
    protocol = mission._protocol(normalized, spec, chosen)
    raw_protocol = (_json(protocol) + "\n").encode()
    contract.check(len(raw_protocol) <= controller.MAX_REQUEST_BYTES)
    plan = {"ok": True, "status": "autoplan_ready", "evaluation_scope": scope, "qualification_blockers": blockers,
            "qualification_note": ("Development measurements can guide another trial. Qualification requires explicit acceptance gates and a declared fresh final test."
                                   if scope == "development" else "The declared gates are frozen; only the reference and the selected development finalist can reach the final test."),
            "runtime_ready": runtime_fingerprint is not None, "output_dir": str(output), "defaults": defaults,
            "decisions": decisions, "excluded_recipes": list(_EXCLUSIONS), "protocol": protocol,
            "variant_ids": [v["id"] for v in chosen], "generated_token_preflight_ceiling": ceiling,
            **provenance}
    if not start:
        return plan
    protocol_path = mission._freeze(output, raw_protocol)
    result = controller._start({"action": "start", "protocol_path": protocol_path, "dataset_dir": str(dataset),
                                "output_dir": str(output / "campaign")}, settings,
                               autonomy={"source_sha256": {**data, "protocol": memory.digest(raw_protocol)},
                                         "runtime_fingerprint": runtime_fingerprint})
    if result.get("ok"):
        result.update(status="autostart_queued", plan=plan)
    return result


def dispatch(request, settings):
    try:
        action = request["action"]
        if action in ("autoplan", "autostart") and contract.fields(
            request, ("action",), ("mission", "objective", "task", "model", "output_choices", *mission._REQUIRED, *mission._OPTIONAL),
        ):
            return plan(request, settings, start=action == "autostart")
        if action == "autoreport" and contract.fields(request, ("action", "job_id")):
            return memory.report(request["job_id"], settings)
        if action == "lessons" and contract.fields(request, ("action",), ("limit",)):
            limit = request.get("limit", 20)
            contract.check(contract.number(limit, 1, 100, integer=True))
            return {"ok": True, "status": "lessons", "lessons": memory.reconcile(settings, limit)}
    except registry.RegistryError as exc:
        return failure(*exc.args)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, OverflowError, RuntimeError, sqlite3.Error):
        pass
    return failure("invalid_autonomous_mission", "Supply a valid local task, assets, dataset or autonomous job in this profile.")
