"""Reproducible producer-side selection and held-out triage evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .endpoint import failure
from .skill_eval import compare, read_cases, summarize
from .skill_runtime import (
    LABELS, MAX_PROMPT, TRAINING_SEED, TRAINING_VARIANTS, messages, run_batch, suffix, train, training_settings,
)

PRODUCER_THRESHOLDS = {
    "min_correct": 34, "min_net_corrections": 4, "max_ambiguous_wrong_orientation_increase": 0,
}
MAX_SECONDS = 1800


def _read_manifest(root):
    path = root / "manifest.sha256"
    if path.stat().st_size > 16_000:
        raise ValueError
    entries = {}
    for row in path.read_text(encoding="utf-8").splitlines():
        digest, name = row.split(maxsplit=1)
        if name in entries or Path(name).name != name or len(digest) != 64:
            raise ValueError
        entries[name] = digest
    return entries


def _verified_path(root, manifest, name):
    path = root / name
    if path.stat().st_size > 2_000_000 or path.is_symlink():
        raise ValueError
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(name):
        raise ValueError
    return path


def _disjoint(*splits):
    ids, families = set(), set()
    for split in splits:
        split_ids = {case["id"] for case in split}
        split_families = {case["family"] for case in split}
        if split_ids & ids or split_families & families:
            raise ValueError
        ids.update(split_ids)
        families.update(split_families)


def _prompts(path, training):
    entries = json.loads(path.read_text(encoding="utf-8"))["candidates"]
    if not 1 <= len(entries) <= 2:
        raise ValueError
    seen = set()
    by_id = {case["id"]: case for case in training}
    for candidate in entries:
        name = candidate["id"]
        instructions = candidate["instructions"]
        fewshots = candidate["fewshots"]
        if not isinstance(name, str) or not name or name in seen or not isinstance(instructions, str):
            raise ValueError
        if not isinstance(fewshots, list):
            raise ValueError
        seen.add(name)
        prompt = instructions
        if fewshots:
            lines = []
            for item in fewshots:
                source = by_id[item["id"]]
                if item["text"] != source["text"] or item["label"] != source["label"]:
                    raise ValueError
                lines.append(f"{messages('', item['text'])[-1]['content']} → {suffix(item['label'])}")
            prompt += "\n\nExemples annotés :\n" + "\n".join(lines)
        if not prompt.strip() or len(prompt) > MAX_PROMPT:
            raise ValueError
        yield name, prompt


def _write_json(path, result):
    staged = path.with_suffix(".tmp")
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    staged.replace(path)


def _prepare_data(directory, training, dev, prompt):
    directory.mkdir(mode=0o700)
    for name, cases in (("train", training), ("valid", dev)):
        with (directory / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for case in cases:
                chat = messages(prompt, case["text"])
                chat.append({"role": "assistant", "content": suffix(case["label"])})
                stream.write(json.dumps({"messages": chat}, ensure_ascii=False) + "\n")


def run_experiment(dataset_dir, output_dir, *, python, model, lock_path, timeout=MAX_SECONDS, progress=None):
    """Select on train/dev, freeze one candidate, then evaluate the producer holdout."""
    start = time.monotonic()
    if (
        type(timeout) not in (int, float) or not math.isfinite(timeout)
        or not 0 < timeout <= MAX_SECONDS
    ):
        return failure("invalid_input", "The experiment requires a bounded local time limit.")
    deadline = start + timeout
    try:
        root = Path(dataset_dir).resolve(strict=True)
        output = Path(output_dir).resolve()
        manifest = _read_manifest(root)
        contract = json.loads(_verified_path(root, manifest, "contract.json").read_text(encoding="utf-8"))
        protocol = json.loads(_verified_path(root, manifest, "protocol.json").read_text(encoding="utf-8"))
        producer_thresholds = {
            **protocol["thresholds"]["producer"],
            "max_ambiguous_wrong_orientation_increase":
                protocol["thresholds"]["max_ambiguous_wrong_orientation_increase"],
        }
        if (
            contract["version"] != "triage-v1" or protocol["version"] != "triage-v1"
            or tuple(contract["labels"]) != LABELS
            or contract["canonical_outputs"] != [suffix(label) for label in LABELS]
            or protocol["seeds"]["training"] != TRAINING_SEED
            or producer_thresholds != PRODUCER_THRESHOLDS
        ):
            raise ValueError
        training = read_cases(_verified_path(root, manifest, "train.jsonl"))
        dev = read_cases(_verified_path(root, manifest, "dev.jsonl"))
        if (len(training), len(dev)) != (192, 32):
            raise ValueError
        _disjoint(training, dev)
        prompts = list(_prompts(_verified_path(root, manifest, "prompt_candidates.json"), training))
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return failure("invalid_experiment", "Experiment inputs, fingerprints or output directory are invalid.")

    options = {"python": python, "model": model, "lock_path": lock_path}
    model_hash = None

    def emit(stage, **metrics):
        if progress is not None:
            progress({"stage": stage, "elapsed_seconds": round(time.monotonic() - start, 3), **metrics})

    def evaluate(cases, prompt, adapter=None, adapter_hash=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return failure("runtime_timeout", "The local experiment exceeded its time limit.")
        return run_batch(
            [case["text"] for case in cases], prompt=prompt, adapter=adapter, timeout=remaining,
            expected_model=model_hash, expected_adapter=adapter_hash, **options,
        )

    prompt_results = []
    for index, (name, prompt) in enumerate(prompts):
        batch = evaluate(dev, prompt)
        if not batch["ok"]:
            return batch
        model_hash = batch["model_sha256"]
        metrics = summarize(dev, batch)
        prompt_results.append((name, prompt, metrics))
        emit("prompt_dev_done", candidate=index, correct=metrics["correct"], total=metrics["total"])
    prompt_id, selected_prompt, _ = max(
        prompt_results, key=lambda candidate: candidate[2]["correct"]
    )

    try:
        _prepare_data(output / "data", training, dev, selected_prompt)
        _write_json(output / "training_plan.json", {
            "model_sha256": model_hash, "prompt_id": prompt_id,
            "prompt_sha256": hashlib.sha256(selected_prompt.encode()).hexdigest(),
            "train_sha256": manifest["train.jsonl"], "dev_sha256": manifest["dev.jsonl"],
            "variants": [training_settings(variant) for variant in range(len(TRAINING_VARIANTS))],
        })
    except OSError:
        return failure("artifact_write_failed", "The local experiment could not prepare training files.")
    trained = []
    for variant in range(len(TRAINING_VARIANTS)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return failure("runtime_timeout", "The local experiment exceeded its time limit.")
        adapter = output / f"adapter-{variant}"
        emit("training_start", variant=variant)
        fit = train(
            data_dir=output / "data", adapter_dir=adapter, variant=variant,
            expected_model=model_hash, timeout=remaining, **options,
        )
        if not fit["ok"]:
            return fit
        emit("training_done", variant=variant, training_seconds=fit.get("elapsed_seconds"))
        batch = evaluate(dev, selected_prompt, adapter, fit["adapter_sha256"])
        if not batch["ok"]:
            return batch
        metrics = summarize(dev, batch)
        trained.append({"variant": variant, "adapter_path": str(adapter), "fit": fit,
                        "dev": metrics, "prompt_sha256": batch["prompt_sha256"]})
        emit("adapter_dev_done", variant=variant, correct=metrics["correct"], total=metrics["total"])
    selected = max(trained, key=lambda candidate: candidate["dev"]["correct"])
    selected_adapter = selected["adapter_path"]
    adapter_hash = selected["fit"]["adapter_sha256"]
    freeze = {
        "protocol": "triage-v1", "model_sha256": model_hash, "adapter_sha256": adapter_hash,
        "adapter_path": selected_adapter, "selected_prompt": selected_prompt, "selected_prompt_id": prompt_id,
        "prompt_sha256": selected["prompt_sha256"], "selected_variant": selected["variant"],
        "producer_thresholds": PRODUCER_THRESHOLDS, "dataset_sha256": manifest,
        "prompt_dev": [{"id": name, **metrics} for name, _, metrics in prompt_results],
        "training_dev": trained,
        "selection_rule": "Higher dev correct count, then declared order.",
    }
    try:
        _write_json(output / "selected_prompt.json", {
            "prompt": selected_prompt, "prompt_sha256": selected["prompt_sha256"], "prompt_id": prompt_id,
        })
        _write_json(output / "selection.json", freeze)
        producer = read_cases(_verified_path(root, manifest, "producer_test.jsonl"))
        if len(producer) != 40:
            raise ValueError
        _disjoint(training, dev, producer)
    except (OSError, ValueError):
        return failure("invalid_holdout", "The frozen producer holdout or selection record is invalid.")

    emit("producer_holdout_start", variant=selected["variant"], total=len(producer))
    base = evaluate(producer, selected_prompt)
    if not base["ok"]:
        return base
    adapted = evaluate(producer, selected_prompt, selected_adapter, adapter_hash)
    if not adapted["ok"]:
        return adapted
    paired = compare(producer, base, adapted, **PRODUCER_THRESHOLDS)
    if not paired["ok"]:
        return paired
    report_path = output / "report.json"
    report = {**freeze, "ok": True, "producer": paired, "passed": paired["passed"],
              "elapsed_seconds": round(time.monotonic() - start, 3), "report_path": str(report_path)}
    try:
        _write_json(report_path, report)
    except OSError:
        return failure("artifact_write_failed", "The local experiment could not write its report.")
    emit("producer_holdout_done", passed=paired["passed"], correct=paired["adapted"]["correct"],
         total=len(producer), net_corrections=paired["net_corrections"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--dataset-dir", dest="dataset_dir", required=True)
    parser.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--lock", "--lock-path", dest="lock_path", required=True)
    parser.add_argument("--timeout", type=float, default=MAX_SECONDS)
    args = vars(parser.parse_args())
    result = run_experiment(**args)
    if result["ok"]:
        result = {key: result[key] for key in ("ok", "passed", "report_path", "model_sha256", "adapter_sha256")}
    json.dump(result, sys.stdout, ensure_ascii=False)
    print()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
