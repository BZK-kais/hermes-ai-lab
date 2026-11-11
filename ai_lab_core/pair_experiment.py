"""Select a PAWS paraphrase adapter on dev and evaluate one frozen candidate."""

from .process_env import runtime_env

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from .endpoint import failure
from .skill_eval import compare, read_cases, summarize
from .skill_experiment import _disjoint, _prepare_data, _read_manifest, _verified_path, _write_json
from .skill_runtime import (
    MAX_PROMPT, PAIR_LABELS, PAIR_TRAINING_PROFILE, PAIR_TRAINING_SEED, PAIR_TRAINING_VARIANTS,
    _sha256, messages, run_batch, suffix, train, training_settings,
)

VERSION = PAIR_TRAINING_PROFILE
MAX_SECONDS = 3600
CPU_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pair_baselines.py"
if not CPU_SCRIPT.is_file():
    CPU_SCRIPT = Path(__file__).resolve().with_name("assets") / "evaluate_pair_baselines.py"
SIZES = {"train": 1024, "dev": 64, "producer": 128, "receiver": 128}
SPLITS = {name: f"{name}_test.jsonl" if name in ("producer", "receiver") else f"{name}.jsonl"
          for name in SIZES}
THRESHOLDS = {
    "min_correct": 104, "min_net_corrections": 8, "max_false_same_increase": 0,
    "min_cpu_correct_gain": 4,
}


def _dev_rank(metrics):
    return metrics["correct"], -metrics["false_same"]


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _metadata(root):
    manifest = _read_manifest(root)
    contract = json.loads(_verified_path(root, manifest, "contract.json").read_text(encoding="utf-8"))
    protocol = json.loads(_verified_path(root, manifest, "protocol.json").read_text(encoding="utf-8"))
    if (
        contract["version"] != VERSION or protocol["version"] != VERSION
        or contract["language"] != "en" or tuple(contract["labels"]) != PAIR_LABELS
        or contract["canonical_outputs"] != [suffix(label) for label in PAIR_LABELS]
        or protocol["split_sizes"] != SIZES or protocol["splits"] != SPLITS
        or protocol["seeds"] != {"sampling": PAIR_TRAINING_SEED, "training": PAIR_TRAINING_SEED}
        or any(protocol["thresholds"][split] != THRESHOLDS for split in ("producer", "receiver"))
        or not all(_sha256(manifest.get(name)) for name in SPLITS.values())
    ):
        raise ValueError
    return {
        "version": VERSION, "protocol": VERSION, "labels": list(PAIR_LABELS),
        "contract_sha256": manifest["contract.json"], "protocol_sha256": manifest["protocol.json"],
        "dataset_manifest_sha256": hashlib.sha256((root / "manifest.sha256").read_bytes()).hexdigest(),
        "dataset_sha256": manifest, "thresholds": THRESHOLDS,
    }


def _cases(root, metadata, split):
    path = _verified_path(root, metadata["dataset_sha256"], SPLITS[split])
    cases = read_cases(path, labels=PAIR_LABELS)
    if len(cases) != SIZES[split]:
        raise ValueError
    return cases


def _prompts(root, metadata, training):
    path = _verified_path(root, metadata["dataset_sha256"], "prompt_candidates.json")
    candidates = json.loads(path.read_text(encoding="utf-8"))["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError
    by_id = {case["id"]: case for case in training}
    prompts = {}
    for candidate in candidates:
        name, instructions, fewshots = (candidate[key] for key in ("id", "instructions", "fewshots"))
        if (
            not isinstance(name, str) or not name.strip() or name in prompts
            or not isinstance(instructions, str) or not instructions.strip()
            or not isinstance(fewshots, list)
        ):
            raise ValueError
        lines = []
        for item in fewshots:
            source = by_id[item["id"]]
            if item["text"] != source["text"] or item["label"] != source["label"]:
                raise ValueError
            lines.append(f"{messages('', item['text'])[-1]['content']} → {suffix(item['label'])}")
        prompt = instructions + ("\n\nLabeled examples:\n" + "\n".join(lines) if lines else "")
        if len(prompt) > MAX_PROMPT:
            raise ValueError
        prompts[name] = prompt
    return prompts


def _output_path(output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError
    return output


class _Run:
    def __init__(self, *, python, model, lock_path, timeout, progress, analysis_python):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= MAX_SECONDS:
            raise ValueError
        self.start = time.monotonic()
        self.deadline = self.start + timeout
        self.options = {"python": python, "model": model, "lock_path": lock_path}
        self.progress = progress
        self.analysis_python = analysis_python
        self.model_hash = None

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def elapsed(self):
        return round(time.monotonic() - self.start, 3)

    def emit(self, stage, **metrics):
        if self.progress is not None:
            self.progress({"stage": stage, "elapsed_seconds": self.elapsed(), **metrics})

    def evaluate(self, cases, prompt, adapter=None, adapter_hash=None):
        batch = run_batch(
            [case["text"] for case in cases], prompt=prompt, adapter=adapter, timeout=self.remaining(),
            expected_model=self.model_hash, expected_adapter=adapter_hash, labels=PAIR_LABELS, **self.options,
        )
        if batch["ok"]:
            self.model_hash = batch["model_sha256"]
        return batch


def _cpu_fingerprints(path, metadata):
    if not path.is_file():
        return None
    if path.stat().st_size > 100_000:
        raise ValueError
    raw = path.read_bytes()
    record = json.loads(raw)
    model = path.with_suffix(".joblib")
    model_hash = record["baseline_model_sha256"]
    source_hash = hashlib.sha256(CPU_SCRIPT.read_bytes()).hexdigest()
    if (
        record["version"] != VERSION or not _sha256(model_hash)
        or record["source_sha256"] != source_hash
        or record["training_sha256"] != metadata["dataset_sha256"][SPLITS["train"]]
        or record["development_sha256"] != metadata["dataset_sha256"][SPLITS["dev"]]
        or model.stat().st_size > 32_000_000
        or hashlib.sha256(model.read_bytes()).hexdigest() != model_hash
    ):
        raise ValueError
    return {
        "freeze_sha256": hashlib.sha256(raw).hexdigest(), "model_sha256": model_hash,
        "source_sha256": source_hash,
    }


def _cpu_path(root, baseline_freeze):
    if baseline_freeze is not None:
        return Path(baseline_freeze).resolve()
    return root / "baseline-freeze.json"


def _cpu_result(run, root, split, cases, metadata, cpu_path):
    if run.analysis_python is None or metadata["cpu_freeze"] is None:
        return {"status": "awaiting_cpu"}
    try:
        if _cpu_fingerprints(cpu_path, metadata) != metadata["cpu_freeze"]:
            raise ValueError
        completed = subprocess.run(
            [str(run.analysis_python), str(CPU_SCRIPT), "--split", split, "--data-dir", str(root),
             "--freeze-path", str(cpu_path)],
            capture_output=True, timeout=run.remaining(), check=False, env=runtime_env(),
        )
        if completed.returncode or len(completed.stdout) > 512_000:
            raise ValueError
        result = json.loads(completed.stdout)
        rows = result["predictions"]
        if (
            result["ok"] is not True or not isinstance(rows, list)
            or [row["id"] for row in rows] != [case["id"] for case in cases]
            or not all(row["label"] in PAIR_LABELS for row in rows)
            or result["training_sha256"] != metadata["dataset_sha256"][SPLITS["train"]]
            or result["holdout_sha256"] != metadata["dataset_sha256"][SPLITS[split]]
            or result["baseline_model_sha256"] != metadata["cpu_freeze"]["model_sha256"]
            or result["source_sha256"] != metadata["cpu_freeze"]["source_sha256"]
            or result["split"] != split
            or _cpu_fingerprints(cpu_path, metadata) != metadata["cpu_freeze"]
        ):
            raise ValueError
        correct = sum(row["label"] == case["label"] for row, case in zip(rows, cases))
        false_same = sum(row["label"] == "same" and case["label"] == "different"
                         for row, case in zip(rows, cases))
        if (result["correct"], result["total"], result["false_same"]) != (correct, len(cases), false_same):
            raise ValueError
    except (AttributeError, KeyError, OSError, TypeError, ValueError, subprocess.TimeoutExpired):
        return {"status": "cpu_failed"}
    return {**result, "status": "evaluated"}


def _holdout(run, root, output, split, cases, freeze, adapter, prompt_file, cpu_path):
    prompt = freeze["prompt"]
    run.emit(f"{split}_holdout_start", variant=freeze["selected_variant"], total=len(cases))
    base = run.evaluate(cases, prompt)
    if not base["ok"]:
        return base
    _write_json(output / f"{split}_base_batch.json", base)
    adapted = run.evaluate(cases, prompt, adapter, freeze["adapter_sha256"])
    if not adapted["ok"]:
        return adapted
    _write_json(output / f"{split}_adapted_batch.json", adapted)
    comparison_thresholds = {key: value for key, value in THRESHOLDS.items() if key != "min_cpu_correct_gain"}
    paired = compare(cases, base, adapted, labels=PAIR_LABELS, **comparison_thresholds)
    if not paired["ok"]:
        return paired
    cpu = _cpu_result(run, root, split, cases, freeze, cpu_path)
    if cpu["status"] == "evaluated":
        cpu["adapted_correct_gain"] = paired["adapted"]["correct"] - cpu["correct"]
        cpu["passed"] = cpu["adapted_correct_gain"] >= THRESHOLDS["min_cpu_correct_gain"]
    passed = paired["passed"] and cpu.get("passed", False)
    report_path = output / "report.json"
    report = {
        **freeze, "ok": True, "passed": passed, split: paired, "cpu_baseline": cpu,
        "adapter_path": str(adapter), "prompt_file": str(prompt_file), "report_path": str(report_path),
        "elapsed_seconds": run.elapsed(),
    }
    _write_json(report_path, report)
    run.emit(f"{split}_holdout_done", passed=passed, correct=paired["adapted"]["correct"],
             total=len(cases), net_corrections=paired["net_corrections"])
    return report


def run_experiment(dataset_dir, output_dir, *, python, model, lock_path, timeout=MAX_SECONDS,
                   progress=None, analysis_python=None, baseline_freeze=None):
    """Select two prompts and two training presets on dev, then score the producer split."""
    try:
        run = _Run(python=python, model=model, lock_path=lock_path, timeout=timeout,
                   progress=progress, analysis_python=analysis_python)
        root = Path(dataset_dir).resolve(strict=True)
        metadata = _metadata(root)
        cpu_path = _cpu_path(root, baseline_freeze)
        metadata["cpu_freeze"] = _cpu_fingerprints(cpu_path, metadata)
        training = _cases(root, metadata, "train")
        dev = _cases(root, metadata, "dev")
        _disjoint(training, dev)
        prompts = _prompts(root, metadata, training)
        output = _output_path(output_dir)
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return failure("invalid_experiment", "Experiment inputs, fingerprints or output directory are invalid.")

    try:
        prompt_results = {}
        for index, (name, prompt) in enumerate(prompts.items()):
            batch = run.evaluate(dev, prompt)
            if not batch["ok"]:
                return batch
            raw_path = output / f"prompt-dev-{index}.json"
            _write_json(raw_path, batch)
            metrics = summarize(dev, batch, labels=PAIR_LABELS)
            prompt_results[name] = {**metrics, "batch_path": raw_path.name, "prompt_sha256": _digest(prompt)}
            run.emit("prompt_dev_done", candidate=index, correct=metrics["correct"], total=metrics["total"])
        prompt_id = max(prompt_results, key=lambda name: _dev_rank(prompt_results[name]))
        prompt = prompts[prompt_id]
        _prepare_data(output / "data", training, dev, prompt)
        settings = [training_settings(variant, profile=VERSION) for variant in range(len(PAIR_TRAINING_VARIANTS))]
        _write_json(output / "training_plan.json", {
            **metadata, "model_sha256": run.model_hash, "prompt_id": prompt_id,
            "prompt_sha256": _digest(prompt), "variants": settings, "prompt_dev": prompt_results,
        })

        fitted = []
        for variant, preset in enumerate(settings):
            adapter = output / f"adapter-{variant}"
            run.emit("training_start", variant=variant)
            fit = train(
                data_dir=output / "data", adapter_dir=adapter, variant=variant, expected_model=run.model_hash,
                timeout=run.remaining(), training_profile=VERSION, **run.options,
            )
            if not fit["ok"]:
                return fit
            _write_json(output / f"fit-{variant}.json", fit)
            run.emit("training_done", variant=variant, training_seconds=fit.get("elapsed_seconds"))
            batch = run.evaluate(dev, prompt, adapter, fit["adapter_sha256"])
            if not batch["ok"]:
                return batch
            raw_path = output / f"adapter-dev-{variant}.json"
            _write_json(raw_path, batch)
            metrics = summarize(dev, batch, labels=PAIR_LABELS)
            fitted.append({"variant": variant, "adapter_path": str(adapter), "fit": fit, "settings": preset,
                           "dev": metrics, "batch_path": raw_path.name})
            run.emit("adapter_dev_done", variant=variant, correct=metrics["correct"], total=metrics["total"])

        selected = max(fitted, key=lambda row: _dev_rank(row["dev"]))
        freeze = {
            **metadata, "model_sha256": run.model_hash, "adapter_sha256": selected["fit"]["adapter_sha256"],
            "prompt": prompt, "prompt_sha256": _digest(prompt), "prompt_id": prompt_id,
            "selected_variant": selected["variant"],
        }
        prompt_file = output / "selected_prompt.json"
        _write_json(prompt_file, freeze)
        _write_json(output / "selection.json", {
            **freeze, "adapter_path": selected["adapter_path"], "prompt_dev": prompt_results,
            "training_dev": fitted,
            "selection_rule": "Rank by dev correct count, then fewer false_same cases, then declaration order.",
        })
        producer = _cases(root, metadata, "producer")
        _disjoint(training, dev, producer)
        return _holdout(run, root, output, "producer", producer, freeze, selected["adapter_path"], prompt_file, cpu_path)
    except TimeoutError:
        return failure("runtime_timeout", "The local experiment exceeded its time limit.")
    except OSError:
        return failure("artifact_io_failed", "The local experiment could not read or write an artifact.")
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return failure("invalid_experiment", "The local evaluation or frozen holdout is invalid.")


def run_receiver(dataset_dir, output_dir, *, python, model, lock_path, adapter, prompt_file,
                 timeout=MAX_SECONDS, progress=None, analysis_python=None, baseline_freeze=None):
    """Evaluate the frozen prompt and adapter on the independently held receiver split."""
    try:
        run = _Run(python=python, model=model, lock_path=lock_path, timeout=timeout,
                   progress=progress, analysis_python=analysis_python)
        root = Path(dataset_dir).resolve(strict=True)
        metadata = _metadata(root)
        cpu_path = _cpu_path(root, baseline_freeze)
        prompt_file = Path(prompt_file).resolve(strict=True)
        if prompt_file.stat().st_size > 100_000:
            raise ValueError
        freeze = json.loads(prompt_file.read_text(encoding="utf-8"))
        prompt = freeze["prompt"]
        variant = freeze["selected_variant"]
        cpu_freeze = freeze["cpu_freeze"]
        if (
            any(freeze.get(key) != value for key, value in metadata.items())
            or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT
            or _digest(prompt) != freeze["prompt_sha256"]
            or not _sha256(freeze["model_sha256"]) or not _sha256(freeze["adapter_sha256"])
            or type(variant) is not int or not 0 <= variant < len(PAIR_TRAINING_VARIANTS)
            or cpu_freeze is not None and (
                not all(_sha256(cpu_freeze.get(key)) for key in ("freeze_sha256", "model_sha256", "source_sha256"))
            )
        ):
            raise ValueError
        if analysis_python is not None and cpu_freeze is not None and _cpu_fingerprints(cpu_path, metadata) != cpu_freeze:
            raise ValueError
        run.model_hash = freeze["model_sha256"]
        adapter = Path(adapter).resolve(strict=True)
        receiver = _cases(root, metadata, "receiver")
        output = _output_path(output_dir)
        return _holdout(run, root, output, "receiver", receiver, freeze, adapter, prompt_file, cpu_path)
    except TimeoutError:
        return failure("runtime_timeout", "The local experiment exceeded its time limit.")
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return failure("invalid_experiment", "The receiver inputs, frozen artifacts or output directory are invalid.")
