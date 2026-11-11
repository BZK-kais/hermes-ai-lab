"""Private MLX entrypoint. Its only stdout payload is one JSON object."""

from contextlib import redirect_stdout
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

# Isolated Python preserves the dedicated MLX environment and needs the
# plugin's own source root to import shared prompt and schema definitions.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_lab_core.skill_runtime import (
    DECODER, LABELS, MAX_MESSAGE_BYTES, PAIR_LABELS, TRIAGE_TRAINING_PROFILE, messages,
    normalize_labels, suffix, training_settings, valid_batch,
)


ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")
MODEL_EXTENSIONS = {".json", ".jinja", ".safetensors", ".model", ".txt", ".tiktoken", ".jsonl"}
# Variable-length LoRA batches otherwise retain many gigabytes of free Metal
# buffers. The MLX active-memory peak does not include that allocator cache.
CACHE_LIMIT_BYTES = 256 * 1024**2


def fingerprint(directory, names):
    if not names:
        raise ValueError
    manifest = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append([name, digest.hexdigest()])
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def model_fingerprint(path):
    names = {file.name for file in path.iterdir() if file.suffix in MODEL_EXTENSIONS}
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError
    return fingerprint(path, names)


def adapter_fingerprint(path):
    config_path = path / "adapter_config.json"
    if config_path.is_symlink() or config_path.stat().st_size > 100_000:
        raise ValueError
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("fine_tune_type") != "lora":
        raise ValueError
    return fingerprint(path, ADAPTER_FILES)


def _error(code):
    return {"ok": False, "error": {"code": code}}


def _score(model, tokenizer, mx, prompt, text, labels=LABELS):
    chat = messages(prompt, text)
    prefix = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    sequences = [
        tokenizer.encode(prefix + suffix(label), add_special_tokens=False)
        for label in labels
    ]
    if not prefix_ids or any(ids[:len(prefix_ids)] != prefix_ids for ids in sequences):
        raise ValueError
    length = max(map(len, sequences))
    if length > 2048 or any(len(ids) <= len(prefix_ids) for ids in sequences):
        raise ValueError
    inputs = mx.array([ids[:-1] + [0] * (length - len(ids)) for ids in sequences])
    targets = mx.array([ids[len(prefix_ids):] + [0] * (length - len(ids)) for ids in sequences])
    logits = model(inputs)[:, len(prefix_ids) - 1:, :].astype(mx.float32)
    token_scores = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    token_scores -= mx.logsumexp(logits, axis=-1)
    mask = mx.arange(targets.shape[1])[None, :] < mx.array(
        [len(ids) - len(prefix_ids) for ids in sequences]
    )[:, None]
    total_scores = mx.sum(mx.where(mask, token_scores, 0.0), axis=1).tolist()
    scores = dict(zip(labels, total_scores))
    return {"label": max(labels, key=scores.__getitem__), "scores": scores}


def _classify(payload, model_path, model_hash):
    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    labels = normalize_labels(payload.get("labels", LABELS))
    if not valid_batch(payload["texts"], payload["prompt"]):
        raise ValueError
    adapter = payload.get("adapter")
    adapter_hash = None
    if adapter is not None:
        try:
            adapter_hash = adapter_fingerprint(Path(adapter))
            expected = payload.get("expected_adapter")
            if expected is not None and adapter_hash != expected:
                return _error("invalid_adapter")
        except (AttributeError, OSError, ValueError):
            return _error("invalid_adapter")

    start = time.perf_counter()
    try:
        model, tokenizer = load(
            str(model_path), tokenizer_config={"trust_remote_code": False},
            adapter_path=adapter, trust_remote_code=False,
        )
        model.eval()
        mx.eval(model.parameters())
    except Exception:
        return _error("invalid_adapter" if adapter is not None else "runtime_failed")
    load_ms = round((time.perf_counter() - start) * 1000, 3)
    predictions = []
    for text in payload["texts"]:
        start = time.perf_counter()
        prediction = _score(model, tokenizer, mx, payload["prompt"], text, labels)
        prediction["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        predictions.append(prediction)
    return {"ok": True, "predictions": predictions, "decoder": DECODER, "load_ms": load_ms,
            "model_sha256": model_hash, "adapter_sha256": adapter_hash, "adapter_loaded": adapter is not None,
            "mlx_version": version("mlx"), "mlx_lm_version": version("mlx-lm"),
            "peak_mlx_memory_bytes": mx.get_peak_memory(), "mlx_cache_limit_bytes": CACHE_LIMIT_BYTES,
            **({"labels": list(labels)} if labels == PAIR_LABELS else {})}


def _train(payload, model_path, model_hash):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.lora import CONFIG_DEFAULTS, run

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    data = Path(payload["data_dir"])
    if {file.name for file in data.iterdir()} != {"train.jsonl", "valid.jsonl"}:
        raise ValueError
    if any(not file.is_file() or file.is_symlink() for file in data.iterdir()):
        raise ValueError
    profile = payload.get("training_profile", TRIAGE_TRAINING_PROFILE)
    settings = training_settings(payload["variant"], profile=profile)
    adapter = Path(payload["adapter_dir"])
    adapter.mkdir(mode=0o700)
    args = SimpleNamespace(**{
        **CONFIG_DEFAULTS, **settings, "model": str(model_path), "train": True, "test": False,
        "data": str(data), "adapter_path": str(adapter),
    })
    run(args)
    return {"ok": True, "model_sha256": model_hash,
            "adapter_sha256": adapter_fingerprint(adapter), "settings": settings,
            "adapter_bytes": sum((adapter / name).stat().st_size for name in ADAPTER_FILES),
            "mlx_version": version("mlx"), "mlx_lm_version": version("mlx-lm"),
            "peak_mlx_memory_bytes": mx.get_peak_memory(), "mlx_cache_limit_bytes": CACHE_LIMIT_BYTES,
            **({"training_profile": profile} if profile != TRIAGE_TRAINING_PROFILE else {})}


def _dispatch(payload):
    lock = Path(payload["lock"])
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    with os.fdopen(os.open(lock, flags, 0o600), "rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _error("compute_busy")
        model_path = Path(payload["model"])
        try:
            model_hash = model_fingerprint(model_path)
        except (OSError, ValueError):
            return _error("invalid_model")
        expected = payload.get("expected_model")
        if expected is not None and expected != model_hash:
            return _error("invalid_model")
        if payload["action"] == "classify":
            return _classify(payload, model_path, model_hash)
        if payload["action"] == "train":
            return _train(payload, model_path, model_hash)
        return _error("runtime_failed")


def main():
    try:
        # Redirect even native library output and reserve a duplicate of the
        # protocol descriptor for the final JSON.
        with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as response:
            with open(os.devnull, "w") as discard, redirect_stdout(discard):
                os.dup2(discard.fileno(), 1)
                try:
                    data = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
                    if len(data) > MAX_MESSAGE_BYTES:
                        raise ValueError
                    result = json.dumps(_dispatch(json.loads(data)), allow_nan=False)
                except ImportError:
                    result = json.dumps(_error("runtime_unavailable"))
                except Exception:
                    result = json.dumps(_error("runtime_failed"))
            response.write(result + "\n")
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
