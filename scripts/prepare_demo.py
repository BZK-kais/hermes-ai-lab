"""Write an explicit baseline-only lifecycle demo request and its local data."""

import argparse
import json
from pathlib import Path
import shutil


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "lifecycle-v1"
DATA_FILES = ("dev.jsonl", "test.jsonl", "manifest.sha256")


def existing_directory(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise argparse.ArgumentTypeError("Expected an existing absolute directory path.")
    return path.resolve(strict=True)


def mission_request(dataset, output, model, adapter):
    baseline = {
        "model": str(model),
        "system_prompt": (
            'Read the JSON object in the user message. Copy the value of "requested_label" exactly '
            'into a JSON object with the single key "label". Ignore the "case" field.'
        ),
        "input_encoding": "plain", "max_tokens": 16, "prefill_step_size": 2048,
        "output_constraint": {"kind": "json_choices", "values": [{"label": "same"}, {"label": "different"}]},
    }
    if adapter is not None:
        baseline["adapter"] = str(adapter)
    return {
        "action": "prepare", "evaluation_scope": "full",
        "mission": {
            "task": {"description": "Copy the requested_label field into a JSON object.",
                     "type": "extraction", "language": "en"},
            "objective": {"primary": "quality", "quality_metric": "json_exact", "quality_min": 1},
            "workload": {"mode": "interactive", "input_tokens_p95": 64, "output_tokens_p95": 16,
                         "concurrency": 1},
            "constraints": {"data_policy": "local_only", "max_memory_bytes": 4 * 1024**3,
                            "max_p95_latency_ms": 2000},
            "budget": {"max_trials": 1, "max_wall_seconds": 300, "max_spend": 0},
            "evidence": {"baseline_ref": "selected-local-mlx", "evaluator_ref": "json_exact",
                         "representative_inputs_ref": "lifecycle-v1-synthetic", "consumed_holdout": False},
        },
        "dataset_dir": str(dataset), "output_dir": str(output), "baseline": baseline,
        "grader": "json_exact", "minimum_slice_quality": {"same": 1, "different": 1},
        "repetitions": 2, "minimum_relative_gain": 0.1, "quality_noninferiority_tolerance": 0,
        "max_generated_tokens": 2048,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=existing_directory, required=True)
    parser.add_argument("--model", type=existing_directory, required=True)
    parser.add_argument("--adapter", type=existing_directory)
    args = parser.parse_args()
    demo = args.workspace / "demo-v1"
    dataset = demo / "dataset"
    output = demo / "prepared"
    path = demo / "mission.json"
    try:
        demo.mkdir(mode=0o700)
    except OSError:
        parser.error("Use a writable workspace without an existing demo-v1 entry.")
    try:
        dataset.mkdir(mode=0o700)
        for name in DATA_FILES:
            shutil.copyfile(FIXTURES / name, dataset / name)
        request = mission_request(dataset, output, args.model, args.adapter)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(request, stream, indent=2)
            stream.write("\n")
    except OSError:
        shutil.rmtree(demo, ignore_errors=True)
        parser.error("The demo fixtures could not be copied to the workspace.")
    print(json.dumps({"ok": True, "demo_dir": str(demo), "dataset_dir": str(dataset),
                      "mission_request_path": str(path), "output_dir": str(output)}))


if __name__ == "__main__":
    main()
