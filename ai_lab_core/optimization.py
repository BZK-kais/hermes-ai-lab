"""Build a bounded optimization plan from declared evidence without I/O."""

from copy import deepcopy
import hashlib
import json
import math

from .endpoint import failure


def _choices(*options):
    return {"type": "string", "enum": list(options)}


def _integer(minimum, maximum):
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _number(minimum, maximum):
    return {"type": "number", "minimum": minimum, "maximum": maximum}


def _text(maximum):
    return {"type": "string", "minLength": 1, "maxLength": maximum, "pattern": r"^[^\x00-\x1f]*\S[^\x00-\x1f]*$"}


_BOOLEAN = {"type": "boolean"}
_VERSION = _integer(1, 1)


def _list_of(item, maximum):
    return {"type": "array", "maxItems": maximum, "items": item}


_REF = _text(512)
_SCHEMA = {
    "task": {
        "description": _text(2000),
        "type": _choices("classification", "extraction", "retrieval_qa", "generation", "agent"),
        "language": _text(32),
        "knowledge_changes": _BOOLEAN,
    },
    "objective": {
        "primary": _choices("quality", "latency", "throughput", "memory", "cost"),
        "quality_metric": _text(100),
        "quality_min": {**_number(0, 1), "description": "Minimum acceptable quality score. Higher is better."},
    },
    "workload": {
        "mode": _choices("interactive", "batch", "agent"),
        "input_tokens_p95": _integer(1, 10_000_000),
        "output_tokens_p95": _integer(1, 10_000_000),
        "concurrency": _integer(1, 100_000),
        "shared_prefix": _BOOLEAN,
    },
    "constraints": {
        "data_policy": _choices("local_only", "approved_remote"),
        "max_memory_bytes": _integer(1, 2**60),
        "max_p95_latency_ms": _number(0.001, 86_400_000),
        "max_ttft_ms": _number(0.001, 86_400_000),
        "max_cost_per_request": {**_number(0, 1_000_000), "description": "Cap per request in the declared currency and cost scope."},
        "cost_currency": _choices("EUR", "USD"),
        "cost_scope": _choices("external_spend", "estimated_total_cost"),
    },
    "budget": {
        "max_trials": _integer(0, 1000),
        "max_wall_seconds": _integer(0, 2_592_000),
        "max_spend": {**_number(0, 1_000_000), "description": "External spend cap for trials in the declared currency."},
    },
    "evidence": {
        "baseline_ref": _REF,
        "evaluator_ref": _REF,
        "representative_inputs_ref": _REF,
        "labeled_examples": _integer(0, 100_000_000),
        "failure_modes": {
            **_list_of(_text(160), 20),
            "description": "Observed failures. Use residual_quality_gap only after simpler approaches were evaluated.",
        },
        "consumed_holdout": _BOOLEAN,
    },
}

_REQUIRED = {
    "task": ("description", "type", "language"),
    "objective": tuple(_SCHEMA["objective"]),
    "workload": ("mode", "input_tokens_p95", "output_tokens_p95", "concurrency"),
    "constraints": ("data_policy",),
    "budget": tuple(_SCHEMA["budget"]),
    "evidence": ("baseline_ref", "evaluator_ref", "representative_inputs_ref", "consumed_holdout"),
}

_QUESTIONS = (
    (("task", "objective"), "What is the task, its primary objective and its minimum acceptable quality?"),
    (("evidence",), "What baseline, evaluator and representative inputs are available, including a fresh holdout?"),
    (("workload", "constraints", "budget", "environment"),
     "What execution environment, representative load, data policy and trial limits should apply, "
     "including currency and cost scope when cost matters?"),
)

_COST_FIELDS = ("cost_currency", "cost_scope")


_REQUEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        section: {"type": "object", "additionalProperties": False, "properties": fields}
        for section, fields in _SCHEMA.items()
    },
}
_REQUEST_SCHEMA["properties"]["schema_version"] = {**_VERSION, "default": 1}


def request_schema() -> dict:
    """Return a fresh JSON Schema for Hermes and other tool clients."""
    return deepcopy(_REQUEST_SCHEMA)


def _matches(value, schema):
    kind = schema["type"]
    if kind == "object":
        fields = schema["properties"]
        return type(value) is dict and all(
            key in fields and _matches(item, fields[key]) for key, item in value.items()
        )
    if kind == "array":
        return type(value) is list and len(value) <= schema["maxItems"] and all(
            _matches(item, schema["items"]) for item in value
        )
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        if type(value) is not str:
            return False
        if "enum" in schema:
            return value in schema["enum"]
        return (schema["minLength"] <= len(value) <= schema["maxLength"] and bool(value.strip())
                and not any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value))
    types = (int,) if kind == "integer" else (int, float)
    return type(value) in types and schema["minimum"] <= value <= schema["maximum"] and math.isfinite(value)


_SIMPLE_BASELINES = {
    "classification": "Rules or a lightweight classifier with a frozen abstention threshold.",
    "extraction": "Deterministic parsing and schema validation for fields amenable to rules.",
    "retrieval_qa": "Lexical retrieval with a fixed generator and abstention policy.",
    "generation": "The simplest compatible model and fixed prompt that address the task.",
    "agent": "A deterministic workflow for the portions with a known action sequence.",
}

_PRIORITY = {
    "latency": ("runtime", "prefix_cache", "model"),
    "throughput": ("batching", "runtime", "model"),
    "memory": ("quantization", "model", "runtime"),
    "cost": ("model", "quantization", "batching"),
}


def _inventory(environment):
    if type(environment) is not dict or environment.get("ok") is not True:
        return (), ()
    hardware = environment.get("hardware")
    profiles = hardware.get("profiles") if type(hardware) is dict else None
    operations = environment.get("operation_ids")
    return tuple(
        items if _matches(items, _list_of(_text(128), 256)) else ()
        for items in (profiles, operations)
    )


def _candidate(identifier, hypothesis, measures, tradeoff, prerequisite, operations=()):
    return {
        "id": identifier,
        "hypothesis": hypothesis,
        "measures": measures,
        "tradeoff": tradeoff,
        "prerequisite": prerequisite,
        "execution": "catalog_only" if operations else "unsupported",
        "operation_ids": list(operations),
    }


def _candidates(task, objective, workload, evidence, profiles, operations):
    apple = "apple-silicon-mlx" in profiles
    cuda = any(profile.startswith("cuda-") for profile in profiles)

    def listed(*names):
        return tuple(name for name in names if name in operations)

    serving = listed("mlx.serve", "llama.serve") if apple else listed("llama.serve")
    result = []
    if task.get("type"):
        result.append(_candidate(
            "prompt", "A fixed instruction, example set or output schema may improve task fidelity.",
            ["quality", "input_tokens", "p95_ttft_ms"], "Extra examples can increase prefill and cost.",
            "Freeze prompt and generation settings using development cases.",
        ))
    if task.get("type") == "classification":
        result.append(_candidate(
            "calibration", "A calibrated threshold may improve the operating point.",
            ["quality", "coverage", "false_positive_rate", "false_negative_rate"],
            "Reducing false positives can reduce coverage or recall.",
            "Preregister error costs and select the threshold on development data.",
        ))
    if task.get("type") == "retrieval_qa" or task.get("knowledge_changes") is True:
        result.append(_candidate(
            "retrieval", "Retrieving current evidence may reduce stale or unsupported answers.",
            ["quality", "retrieval_recall", "citation_accuracy", "p95_end_to_end_ms"],
            "Retrieval adds latency and prompt tokens.",
            "Version the corpus, index, freshness policy and provenance-aware evaluator.",
        ))
    if objective in ("latency", "throughput", "memory", "cost"):
        result.append(_candidate(
            "runtime", "A compatible runtime may reduce overhead on the target workload.",
            ["p95_ttft_ms", "decode_tokens_per_second", "p95_end_to_end_ms", "peak_memory"],
            "Runtime switches may change the template, kernels or numerical output.",
            "Verify a compatible local model, runtime, output parity and actual generation.",
            serving,
        ))
        result.append(_candidate(
            "quantization", "Lower precision may improve memory fit and serving cost.",
            ["quality", "peak_memory", "p95_end_to_end_ms", "cost_per_request"],
            "Quantization can lose quality or slow execution on an unsupported kernel.",
            "Check format, model and target kernel compatibility, then remeasure quality.",
            listed("mlx.convert", "llama.quantize") if apple else listed("llama.quantize"),
        ))
    if workload.get("mode") == "batch" or workload.get("concurrency", 1) > 1:
        result.append(_candidate(
            "batching", "Batching may increase throughput under overlapping requests.",
            ["requests_per_second", "p95_ttft_ms", "p95_end_to_end_ms", "peak_memory"],
            "Queue time and memory can increase.",
            "Verify runtime batch controls and replay representative arrival rates.",
        ))
    if workload.get("shared_prefix") is True:
        result.append(_candidate(
            "prefix_cache", "Reusing a shared prefix may reduce prefill work.",
            ["p95_ttft_ms", "cache_hit_rate", "peak_memory"],
            "Cache storage and isolation matter. Decode speed needs a separate measurement.",
            "Check prefix identity, cache scope and support, then compare cold and warm runs.",
        ))
    if objective:
        result.append(_candidate(
            "model", "A different compatible model may improve the target quality-resource trade-off.",
            ["quality", "p95_end_to_end_ms", "peak_memory", "cost_per_request"],
            "Changing model can alter task behavior and serving requirements.",
            "Use an authorized model version with a compatible tokenizer, template and runtime.",
            serving,
        ))
    if evidence.get("labeled_examples", 0) > 0 and "residual_quality_gap" in evidence.get("failure_modes", []):
        training = listed("mlx.train") if apple else listed("trl.train", "unsloth.train") if cuda else ()
        result.append(_candidate(
            "training", "Adaptation may address a residual quality gap after simpler trials.",
            ["quality", "regressions_by_slice", "training_time", "peak_memory"],
            "Training adds cost and may worsen previously correct cases.",
            "Verify the residual gap and sufficient authorized labels, then qualify a compatible recipe.",
            training,
        ))
    priority = _PRIORITY.get(objective, ("retrieval", "calibration", "prompt"))
    rank = {identifier: index for index, identifier in enumerate(priority)}
    return sorted(result, key=lambda candidate: rank.get(candidate["id"], len(rank)))


def _metrics(task_type, mode, primary, constraints, budget):
    metrics = ["quality", "success_rate", "p95_end_to_end_ms", "peak_active_bytes",
               "peak_cache_bytes", "peak_rss_bytes"]
    if mode == "batch":
        metrics.extend(("goodput_requests_per_second", "batch_elapsed_seconds"))
    elif primary == "throughput" or mode == "interactive":
        metrics.append("requests_per_second")
    if mode == "agent" or task_type == "agent":
        metrics.extend(("task_success_rate", "tool_error_rate", "retries_per_task"))
    if task_type == "generation" and mode == "interactive":
        metrics.extend(("p95_ttft_ms", "p95_time_per_output_token_ms", "decode_tokens_per_second"))
    elif "max_ttft_ms" in constraints:
        metrics.append("p95_ttft_ms")
    if primary == "cost" or "max_cost_per_request" in constraints:
        metrics.append("cost_per_request")
    if budget.get("max_spend", 0) > 0:
        metrics.append("trial_external_spend")
    return metrics


def analyze(request: dict, environment: dict) -> dict:
    """Return a plan and context gaps. The request and inventory are never executed."""
    if not _matches(request, _REQUEST_SCHEMA):
        return failure("invalid_optimization_request", "Supply valid fields for optimization schema version 1.")

    profiles, operations = _inventory(environment)
    sections = {section: request.get(section, {}) for section in _SCHEMA}
    task = sections["task"]
    objective = sections["objective"]
    primary = objective.get("primary")
    workload = sections["workload"]
    constraints = sections["constraints"]
    budget = sections["budget"]
    evidence = sections["evidence"]
    needs_cost = primary == "cost" or "max_cost_per_request" in constraints or budget.get("max_spend", 0) > 0
    missing = [f"{section}.{field}" for section, fields in _SCHEMA.items()
               for field in fields if field not in sections[section] and (needs_cost or field not in _COST_FIELDS)]
    blockers = [f"{section}.{field}" for section, fields in _REQUIRED.items()
                for field in fields if field not in sections[section]]
    if needs_cost:
        blockers.extend(f"constraints.{field}" for field in _COST_FIELDS if field not in constraints)
    if not profiles:
        blockers.append("environment.hardware.profiles")
        missing.append("environment.hardware.profiles")
    if evidence.get("consumed_holdout") is True:
        blockers.append("evidence.fresh_holdout")
        missing.append("evidence.fresh_holdout")
    questions = [question for prefixes, question in _QUESTIONS
                 if any(field.split(".")[0] in prefixes for field in blockers)]

    baselines = [{"kind": "current", "reference": evidence.get("baseline_ref"), "verified": False}]
    if task.get("type") in _SIMPLE_BASELINES:
        baselines.append({"kind": "simple", "description": _SIMPLE_BASELINES[task["type"]], "verified": False})
    candidates = _candidates(task, primary, workload, evidence, profiles, operations)
    canonical = json.dumps({"schema_version": 1, **sections}, sort_keys=True, separators=(",", ":"))
    return {
        "ok": True,
        "schema_version": 1,
        "status": "needs_context" if blockers else "plan_ready",
        "mission_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "missing": missing,
        "blockers": blockers,
        "questions": questions,
        "diagnosis": {
            "status": "unmeasured",
            "hardware_profiles": list(profiles),
            "hypotheses": [candidate["id"] for candidate in candidates],
            "references_verified": False,
        },
        "baselines": baselines,
        "levers": candidates,
        "protocol": {
            "objective": objective.copy(),
            "constraints": constraints.copy(),
            "representative_load": workload.copy(),
            "references": {
                key: evidence.get(key) for key in ("evaluator_ref", "representative_inputs_ref")
            },
            "freeze": "Pin dataset splits, evaluator, thresholds, model, runtime and settings before final tests.",
            "quality": "Score paired cases with a frozen independent evaluator. Tune on development only and require a fresh reserved holdout.",
            "cache": "Measure cold and warm cache separately, with matched workload and repeated runs.",
            "metrics": _metrics(task.get("type"), workload.get("mode"), primary, constraints, budget),
            "measurement": "Record missing or inapplicable metrics explicitly, including non-streaming TTFT. "
                           "Count only successful results meeting the declared criteria in batch goodput.",
            "cost": "Verify price sources, dates and units before comparison. Use the declared currency and cost scope. "
                    "Estimated total cost includes local energy and hardware allocation. Verify these inputs separately "
                    "from external spend, which can be zero for a local run.",
            "decision_criteria": [
                "Apply mandatory constraints and the quality floor before ranking by the primary objective.",
                "A latency gain cannot offset a failed quality floor.",
                "Missing measurements required for the objective or mandatory constraints prevent qualification.",
                "Claim a speed gain only after paired measurements on the same representative workload.",
            ],
            "selection": "Use the best measured feasible candidate or report a Pareto frontier within the declared budget.",
        },
        "search": {"bounds": budget.copy(), "limits_enforced": False,
                   "executed": False, "best_measured": None},
        "next_actions": [
            "Resolve the context gaps." if blockers else "Verify the references and freeze the baseline and reserved cases.",
            "Run only explicitly authorized trials within the declared limits.",
            "Optionally use ai_lab_benchmark with an existing compatible Mac environment and authorized local inputs "
            "for sequential inference at concurrency 1. Measure concurrent service traffic separately.",
            "Qualify on the target profile before explicit activation and preserve a rollback target.",
        ],
    }
