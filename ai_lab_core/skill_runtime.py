"""Bounded local execution for the Hermes classification experiments."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .endpoint import failure

LABELS = ("bug", "configuration", "demande", "a_clarifier")
PAIR_LABELS = ("same", "different")
PAIR_TRAINING_PROFILE = "pair-paraphrase-v1"
PAIR_IMPROVEMENT_PROFILE = "pair-improve-v1"
TRIAGE_TRAINING_PROFILE = "triage-v1"
DEFAULT_PROMPT = (
    "Classe le signalement technique. Réponds par un objet JSON avec la seule clé label. "
    "bug : comportement existant qui dévie de l'attendu. "
    "configuration : installation, réglage ou paramétrage existant. "
    "demande : fonctionnalité nouvelle. a_clarifier : informations insuffisantes ou conflit."
)
DECODER = "chat-json-complete-suffix-logprob-v1"
MAX_CASES = 256
MAX_TEXT = 4000
MAX_PROMPT = 16000
MAX_MESSAGE_BYTES = 5_000_000
MAX_RESPONSE_BYTES = 512_000
TRAINING_SEED = 3111
TRAINING_VARIANTS = (
    {"iters": 72, "num_layers": 8, "learning_rate": 3e-4},
    {"iters": 120, "num_layers": 12, "learning_rate": 1e-4},
)


PAIR_TRAINING_VARIANTS = (
    {"iters": 128, "num_layers": 8, "learning_rate": 2e-4},
    {"iters": 256, "num_layers": 12, "learning_rate": 1e-4},
)
PAIR_TRAINING_SEED = 20261109
PAIR_IMPROVEMENT_VARIANTS = (
    {"iters": 512, "num_layers": 12, "learning_rate": 1e-4},
)


def normalize_labels(labels):
    if isinstance(labels, (tuple, list)) and tuple(labels) in (LABELS, PAIR_LABELS):
        return tuple(labels)
    raise ValueError("Unknown classification labels.")


def training_settings(variant, profile=TRIAGE_TRAINING_PROFILE):
    if profile == TRIAGE_TRAINING_PROFILE:
        variants, seed, batch, max_length, rank = TRAINING_VARIANTS, TRAINING_SEED, 2, 2048, 8
    elif profile in (PAIR_TRAINING_PROFILE, PAIR_IMPROVEMENT_PROFILE):
        variants = PAIR_TRAINING_VARIANTS if profile == PAIR_TRAINING_PROFILE else PAIR_IMPROVEMENT_VARIANTS
        seed, batch, max_length, rank = PAIR_TRAINING_SEED, 4, 1024, 16
    else:
        raise ValueError("Unknown training profile.")
    if type(variant) is not int or not 0 <= variant < len(variants):
        raise ValueError("Unknown training variant.")
    preset = variants[variant]
    return {
        "fine_tune_type": "lora", "seed": seed, "batch_size": batch, "max_seq_length": max_length,
        "mask_prompt": True, "val_batches": -1, "steps_per_eval": preset["iters"],
        "save_every": preset["iters"], "lora_parameters": {"rank": rank, "dropout": 0.0, "scale": 20.0},
        **preset,
    }


def suffix(label):
    return json.dumps({"label": label}, separators=(",", ":"))


def messages(prompt, text):
    # Keep literal role-delimiter tokens out of the user turn.
    user = json.dumps(text, ensure_ascii=False).replace("<", r"\u003c")
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


def valid_batch(texts, prompt):
    return (
        isinstance(texts, list)
        and 1 <= len(texts) <= MAX_CASES
        and all(isinstance(text, str) and 0 < len(text) <= MAX_TEXT and text.strip()
                for text in texts)
        and isinstance(prompt, str)
        and 0 < len(prompt) <= MAX_PROMPT
        and bool(prompt.strip())
    )


def _worker(payload, *, python, model, lock_path, timeout):
    try:
        interpreter = Path(python).expanduser().absolute()
        model = Path(model).expanduser().resolve(strict=True)
        lock_path = Path(lock_path).expanduser().resolve()
        if (
            not interpreter.is_file() or not model.is_dir()
            or not _number(timeout)
            or not 0 < timeout <= 7200
        ):
            raise ValueError
        payload = {**payload, "model": str(model), "lock": str(lock_path)}
        data = json.dumps(payload, allow_nan=False).encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return failure("invalid_input", "Local runtime paths or request are invalid.")

    # Only interpreter and entrypoint are passed in argv. No provider credentials
    # or Python path overrides are needed to run this local worker.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        child = subprocess.run(
            [str(interpreter), "-I", str(Path(__file__).resolve().with_name("skill_worker.py"))],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return failure("runtime_timeout", "Local MLX computation exceeded its time limit.")
    except OSError:
        return failure("runtime_unavailable", "The local MLX worker could not start.")
    try:
        if child.returncode or len(child.stdout) > MAX_RESPONSE_BYTES:
            raise ValueError
        result = json.loads(child.stdout)
        if not isinstance(result, dict) or result.get("ok") not in (True, False):
            raise ValueError
        if result["ok"] is False:
            error = result.get("error")
            allowed = {
                "compute_busy": "A local AI Lab computation already holds the GPU lock.",
                "invalid_model": "The local model or its fingerprint is invalid.",
                "invalid_adapter": "The local adapter or its fingerprint is invalid.",
                "runtime_failed": "Local MLX computation failed.",
                "runtime_unavailable": "The local MLX dependencies could not be loaded.",
            }
            code = error.get("code") if isinstance(error, dict) else None
            if not isinstance(code, str) or code not in allowed:
                raise ValueError
            return failure(code, allowed[code])
        if result["ok"] is not True:
            raise ValueError
        return result
    except (UnicodeError, ValueError):
        return failure("invalid_runtime_output", "The local MLX worker returned an invalid result.")


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _number(value):
    return (type(value) is int and abs(value) <= 10**12) or (type(value) is float and math.isfinite(value))


def _valid_predictions(result, count, adapter, expected_model, expected_adapter, labels=LABELS):
    predictions = result.get("predictions")
    if (
        result.get("decoder") != DECODER
        or result.get("labels", list(LABELS)) != list(labels)
        or not _number(result.get("load_ms")) or result["load_ms"] < 0
        or not _sha256(result.get("model_sha256"))
        or expected_model is not None and result["model_sha256"] != expected_model
        or result.get("adapter_loaded") is not (adapter is not None)
        or adapter is not None and not _sha256(result.get("adapter_sha256"))
        or adapter is None and result.get("adapter_sha256") is not None
        or expected_adapter is not None and result.get("adapter_sha256") != expected_adapter
        or not isinstance(predictions, list) or len(predictions) != count
    ):
        return False
    for prediction in predictions:
        if not isinstance(prediction, dict):
            return False
        scores = prediction.get("scores")
        latency = prediction.get("latency_ms")
        if (
            not isinstance(scores, dict) or set(scores) != set(labels)
            or any(not _number(value) or value > 0 for value in scores.values())
            or not _number(latency) or latency < 0
            or prediction.get("label") != max(labels, key=scores.__getitem__)
        ):
            return False
    return True


def run_batch(
    texts, *, python, model, lock_path, prompt=DEFAULT_PROMPT, adapter=None, timeout=120,
    expected_model=None, expected_adapter=None, labels=LABELS,
):
    """Return label, full suffix log probabilities and latency for every input."""
    try:
        labels = normalize_labels(labels)
    except ValueError:
        return failure("invalid_input", "Unknown classification labels.")
    if not valid_batch(texts, prompt):
        return failure("invalid_input", "Triage requires nonempty reports and a bounded system prompt.")
    if any(value is not None and not _sha256(value) for value in (expected_model, expected_adapter)):
        return failure("invalid_input", "Expected runtime fingerprints must be SHA-256 digests.")
    if adapter is None and expected_adapter is not None:
        return failure("invalid_input", "An adapter fingerprint requires a local adapter path.")
    try:
        adapter = str(Path(adapter).expanduser().resolve()) if adapter is not None else None
    except (OSError, TypeError, RuntimeError):
        return failure("invalid_input", "The local adapter path is invalid.")
    result = _worker(
        {"action": "classify", "texts": texts, "prompt": prompt, "adapter": adapter,
         "expected_model": expected_model, "expected_adapter": expected_adapter, "labels": labels},
        python=python, model=model, lock_path=lock_path, timeout=timeout,
    )
    if result["ok"] and not _valid_predictions(
        result, len(texts), adapter, expected_model, expected_adapter, labels,
    ):
        return failure("invalid_runtime_output", "The local MLX scores or fingerprints are invalid.")
    if result["ok"]:
        result["prompt_sha256"] = hashlib.sha256(prompt.encode(errors="surrogatepass")).hexdigest()
        result["inputs_sha256"] = inputs_fingerprint(texts)
    return result


def inputs_fingerprint(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def train(
    *, python, model, lock_path, data_dir, adapter_dir, variant, expected_model, timeout=900,
    training_profile=TRIAGE_TRAINING_PROFILE,
):
    """Fit a preset local LoRA variant on prepared train and dev data."""
    try:
        settings = training_settings(variant, profile=training_profile)
        if not _sha256(expected_model):
            raise ValueError
    except ValueError:
        return failure("invalid_input", "Training requires a preset variant and a pinned local model.")
    try:
        data_dir = str(Path(data_dir).resolve(strict=True))
        adapter_dir = str(Path(adapter_dir).resolve())
    except (OSError, RuntimeError, TypeError):
        return failure("invalid_input", "The local training paths are invalid.")
    start = time.monotonic()
    result = _worker(
        {"action": "train", "data_dir": data_dir, "adapter_dir": adapter_dir,
         "variant": variant, "expected_model": expected_model, "training_profile": training_profile},
        python=python, model=model, lock_path=lock_path, timeout=timeout,
    )
    if result["ok"]:
        if result.get("model_sha256") != expected_model or not _sha256(result.get("adapter_sha256")):
            return failure("invalid_runtime_output", "Training did not return valid artifact fingerprints.")
        if training_profile in (PAIR_TRAINING_PROFILE, PAIR_IMPROVEMENT_PROFILE) and (
            result.get("training_profile") != training_profile or result.get("settings") != settings
        ):
            return failure("invalid_runtime_output", "Training did not return the requested preset.")
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return result
