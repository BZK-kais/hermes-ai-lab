"""Validate the local controller's finite search protocol."""

from copy import deepcopy
import math
from pathlib import Path
import re

VERSION = "controller-v1"
MAX_CASES = 256
MAX_INPUT_BYTES = 16_384
MAX_EXPECTED_BYTES = 8_192
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def check(condition):
    if not condition:
        raise ValueError("Invalid controller protocol.")


def fields(value, required, optional=()):
    return type(value) is dict and set(required) <= value.keys() <= set(required) | set(optional)


def text(value, maximum, *, empty=False):
    if type(value) is not str or not (empty or value.strip()) or len(value) > maximum or "\0" in value:
        return False
    try:
        return len(value.encode("utf-8")) <= maximum
    except UnicodeError:
        return False


def name(value):
    return type(value) is str and NAME.fullmatch(value) is not None


def sha256(value):
    return type(value) is str and SHA256.fullmatch(value) is not None


def number(value, minimum, maximum, *, integer=False):
    types = (int,) if integer else (int, float)
    return type(value) in types and minimum <= value <= maximum and math.isfinite(value)


def validate(protocol):
    """Return an independent normalized copy, or raise a static ValueError."""
    check(fields(protocol, (
        "version", "usage", "objective", "baseline_id", "grader", "constraints", "budget",
        "repetitions", "minimum_relative_gain", "quality_noninferiority_tolerance", "variants",
    ), ("holdout_candidate_id", "evaluation_scope")))
    check(protocol["version"] == VERSION)
    scope = protocol.get("evaluation_scope", "full")
    check(scope in ("development", "full"))
    check(scope == "full" or "holdout_candidate_id" not in protocol)
    usage = protocol["usage"]
    check(fields(usage, ("description", "task_type")))
    check(text(usage["description"], 2_048))
    check(usage["task_type"] in ("classification", "extraction", "generation"))
    check(protocol["objective"] in ("latency", "quality", "memory", "throughput"))
    grader = protocol["grader"]
    check(fields(grader, ("kind",)) and grader["kind"] in ("json_exact", "exact_text"))

    constraints = protocol["constraints"]
    check(fields(constraints, (
        "quality_min", "max_p95_latency_ms", "max_mlx_memory_bytes",
    ), ("minimum_slice_quality",)))
    check(number(constraints["quality_min"], 0, 1))
    check(number(constraints["max_p95_latency_ms"], 0.001, 3_600_000))
    check(number(constraints["max_mlx_memory_bytes"], 1, 2**60, integer=True))
    slices = constraints.get("minimum_slice_quality", {})
    check(type(slices) is dict and len(slices) <= 32)
    check(all(name(key) and number(value, 0, 1) for key, value in slices.items()))

    budget = protocol["budget"]
    check(fields(budget, ("max_wall_seconds", "max_trials", "max_generated_tokens")))
    check(number(budget["max_wall_seconds"], 60, 3_600, integer=True))
    check(number(budget["max_trials"], 1, 8, integer=True))
    check(number(budget["max_generated_tokens"], 1, 65_536, integer=True))
    check(number(protocol["repetitions"], 2, 5, integer=True))
    check(number(protocol["minimum_relative_gain"], 0, 1))
    check(number(protocol["quality_noninferiority_tolerance"], 0, 1))

    variants = protocol["variants"]
    check(type(variants) is list and 1 <= len(variants) <= budget["max_trials"])
    identifiers = set()
    for variant in variants:
        check(fields(variant, (
            "id", "model", "system_prompt", "input_encoding", "max_tokens", "prefill_step_size",
        ), ("adapter", "output_constraint")))
        check(name(variant["id"]) and variant["id"] not in identifiers)
        identifiers.add(variant["id"])
        for key in ("model", "adapter"):
            if key in variant:
                path = variant[key]
                check(text(path, 4_096) and Path(path).is_absolute()
                      and all(ord(character) >= 32 for character in path))
        check(text(variant["system_prompt"], MAX_INPUT_BYTES, empty=True))
        check(variant["input_encoding"] in ("plain", "json_string"))
        check(number(variant["max_tokens"], 1, 128, integer=True))
        check(type(variant["prefill_step_size"]) is int and variant["prefill_step_size"] in (64, 256, 2048))
        if "output_constraint" in variant:
            from .json_constraints import canonical_choices

            check(grader["kind"] == "json_exact")
            try:
                canonical_choices(variant["output_constraint"])
            except (TypeError, ValueError, RecursionError):
                raise ValueError("Invalid controller protocol.") from None
    check(name(protocol["baseline_id"]) and protocol["baseline_id"] in identifiers)
    if "holdout_candidate_id" in protocol:
        candidate = protocol["holdout_candidate_id"]
        check(name(candidate) and candidate in identifiers and candidate != protocol["baseline_id"])

    result = deepcopy(protocol)
    result["constraints"].setdefault("minimum_slice_quality", {})
    return result
