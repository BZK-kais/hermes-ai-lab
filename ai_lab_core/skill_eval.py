"""Paired classification evaluation with task-specific error metrics."""

import json
from pathlib import Path

from .endpoint import failure
from .skill_runtime import (
    LABELS, MAX_CASES, MAX_TEXT, PAIR_LABELS, _sha256, _valid_predictions, inputs_fingerprint,
    normalize_labels,
)


def read_cases(path, labels=LABELS):
    labels = normalize_labels(labels)
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Dataset is too large.")
    with path.open(encoding="utf-8") as stream:
        cases = [json.loads(line) for line in stream]
    max_cases = 1024 if labels == PAIR_LABELS else MAX_CASES
    if not 0 < len(cases) <= max_cases:
        raise ValueError("Dataset size is outside the supported bounds.")
    ids = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or any(not isinstance(case.get(key), str) or not case[key].strip()
                   for key in ("id", "family", "text", "label"))
            or len(case["text"]) > MAX_TEXT
            or len(case["id"]) > 160 or len(case["family"]) > 160
            or case["id"] in ids or case["label"] not in labels
        ):
            raise ValueError("Dataset contains an invalid or duplicate case.")
        ids.add(case["id"])
    return cases


def _counts(cases, predictions, labels):
    correct = 0
    ambiguous_wrong = 0
    clear_abstentions = 0
    wrong_orientation = 0
    false_same = 0
    by_label = {label: {"correct": 0, "total": 0} for label in labels}
    for case, prediction in zip(cases, predictions):
        expected, actual = case["label"], prediction["label"]
        correct += expected == actual
        by_label[expected]["total"] += 1
        by_label[expected]["correct"] += expected == actual
        ambiguous_wrong += expected == "a_clarifier" and actual != expected
        clear_abstentions += expected != "a_clarifier" and actual == "a_clarifier"
        wrong_orientation += actual != "a_clarifier" and actual != expected
        false_same += expected == "different" and actual == "same"
    metrics = {"correct": correct, "total": len(cases), "accuracy": correct / len(cases)}
    if labels == PAIR_LABELS:
        return {**metrics, "false_same": false_same, "by_label": by_label}
    return {
        **metrics,
        "ambiguous_wrong_orientation": ambiguous_wrong, "clear_abstentions": clear_abstentions,
        "wrong_orientation": wrong_orientation, "by_label": by_label,
    }


def summarize(cases, batch, labels=LABELS):
    """Score a successful batch against labels in original order."""
    return _counts(cases, batch["predictions"], normalize_labels(labels))


def compare(
    cases, base_batch, adapted_batch, *, min_correct, min_net_corrections,
    max_ambiguous_wrong_orientation_increase=0, labels=LABELS, max_false_same_increase=0,
):
    """Require real adapter load and identical base/decoder before evaluating."""
    try:
        labels = normalize_labels(labels)
    except ValueError:
        return failure("invalid_evaluation", "Unknown evaluation labels.")
    if labels == PAIR_LABELS:
        error_metric, max_increase = "false_same", max_false_same_increase
    else:
        error_metric, max_increase = "ambiguous_wrong_orientation", max_ambiguous_wrong_orientation_increase
    if (
        not cases
        or any(case["label"] not in labels for case in cases)
        or not all(type(value) is int for value in
                   (min_correct, min_net_corrections, max_increase))
        or not 0 <= min_correct <= len(cases)
        or not 0 <= max_increase <= len(cases)
        or not -len(cases) <= min_net_corrections <= len(cases)
        or not all(isinstance(batch, dict) and batch.get("ok") is True
                   for batch in (base_batch, adapted_batch))
        or not _valid_predictions(base_batch, len(cases), None, None, None, labels)
        or not _valid_predictions(adapted_batch, len(cases), True, base_batch.get("model_sha256"), None, labels)
        or not _sha256(base_batch.get("prompt_sha256"))
        or base_batch["prompt_sha256"] != adapted_batch.get("prompt_sha256")
        or base_batch.get("inputs_sha256") != inputs_fingerprint([case["text"] for case in cases])
        or base_batch["inputs_sha256"] != adapted_batch.get("inputs_sha256")
    ):
        return failure("invalid_evaluation", "Evaluation requires matching base and loaded-adapter scores.")

    base = summarize(cases, base_batch, labels)
    adapted = summarize(cases, adapted_batch, labels)
    rows = []
    corrections = 0
    regressions = 0
    for case, baseline, candidate in zip(cases, base_batch["predictions"], adapted_batch["predictions"]):
        expected = case["label"]
        base_correct = baseline["label"] == expected
        adapted_correct = candidate["label"] == expected
        corrections += not base_correct and adapted_correct
        regressions += base_correct and not adapted_correct
        rows.append({"id": case["id"], "family": case["family"], "expected": expected,
                     "base": baseline, "adapted": candidate})
    net = corrections - regressions
    error_increase = adapted[error_metric] - base[error_metric]
    increase_key = f"{error_metric}_increase"
    return {
        "ok": True,
        "passed": adapted["correct"] >= min_correct and net >= min_net_corrections
                  and error_increase <= max_increase,
        "base": {**base, "load_ms": base_batch["load_ms"]},
        "adapted": {**adapted, "load_ms": adapted_batch["load_ms"], "adapter_loaded": True},
        "decoder": base_batch["decoder"], "prompt_sha256": base_batch["prompt_sha256"],
        "model_sha256": base_batch["model_sha256"], "adapter_sha256": adapted_batch["adapter_sha256"],
        "corrections": corrections, "regressions": regressions, "net_corrections": net,
        increase_key: error_increase,
        "thresholds": {"min_correct": min_correct, "min_net_corrections": min_net_corrections,
                       f"max_{increase_key}": max_increase},
        "cases": rows,
        **({"labels": list(labels)} if labels == PAIR_LABELS else {}),
    }
