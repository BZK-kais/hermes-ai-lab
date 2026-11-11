"""Bounded local MLX measurements and fixed graders for the controller."""
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
from time import monotonic, perf_counter

from .json_constraints import ConstraintStreamError, compile_constraint
from .skill_runtime import messages
from .skill_worker import ADAPTER_FILES, CACHE_LIMIT_BYTES, MODEL_EXTENSIONS, adapter_fingerprint, model_fingerprint

MAX_PROMPT_TOKENS = 2048
MAX_PROMPT_BYTES = 16_384
MAX_WARMUP_TOKENS = 8


class MeasureError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _constant(value):
    raise ValueError("invalid JSON constant")


def _same_value(actual, expected, *, check_values=True):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _same_value(value, expected[key], check_values=check_values) for key, value in actual.items()
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _same_value(item, other, check_values=check_values) for item, other in zip(actual, expected)
        )
    return not check_values or actual == expected


def _parse_output(output):
    return json.loads(output, object_pairs_hook=_object, parse_float=_number, parse_constant=_constant)


def grade(kind, output, expected):
    if kind == "exact_text":
        return output.strip() == expected.strip()
    if kind != "json_exact":
        raise ValueError("unsupported grader")
    try:
        return _same_value(_parse_output(output), expected)
    except (ValueError, RecursionError):
        return False


def grade_diagnostics(kind, output, expected):
    if kind == "exact_text":
        return {"output_parseable": True, "output_schema_matches": isinstance(expected, str)}
    if kind != "json_exact":
        raise ValueError("unsupported grader")
    try:
        actual = _parse_output(output)
    except (ValueError, RecursionError):
        return {"output_parseable": False, "output_schema_matches": False}
    try:
        matches = _same_value(actual, expected, check_values=False)
    except RecursionError:
        matches = False
    return {"output_parseable": True, "output_schema_matches": matches}


def _prompt_ids(tokenizer, variant, value):
    system = variant["system_prompt"]
    chat = messages(system, value) if variant["input_encoding"] == "json_string" else [
        {"role": "system", "content": system}, {"role": "user", "content": value},
    ]
    if sum(len(turn["content"].encode("utf-8")) for turn in chat) > MAX_PROMPT_BYTES:
        raise MeasureError("invalid_prompt_bytes")
    if not getattr(tokenizer, "has_chat_template", False):
        raise MeasureError("invalid_chat_template")
    try:
        ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    except Exception:
        raise MeasureError("invalid_chat_template") from None
    if (not isinstance(ids, list) or not 1 <= len(ids) <= MAX_PROMPT_TOKENS
            or any(type(token) is not int for token in ids)):
        raise MeasureError("invalid_prompt_tokens")
    return ids


def _check_deadline(deadline):
    if monotonic() >= deadline:
        raise MeasureError("budget_exhausted")


def _generate(stream_generate, model, tokenizer, ids, max_tokens, prefill_step_size, deadline, sample_memory,
              constraint=None):
    _check_deadline(deadline)
    segments = []
    first = None
    response = None
    options = {"max_tokens": max_tokens, "prefill_step_size": prefill_step_size}
    if constraint is not None:
        processor, verify = constraint.generation(len(ids))
        options["logits_processors"] = [processor]
        emitted = []
    start = perf_counter()
    for response in stream_generate(model, tokenizer, ids, **options):
        if first is None:
            first = perf_counter()
        segments.append(response.text)
        if constraint is not None:
            emitted.append(getattr(response, "token", None))
        sample_memory()
        _check_deadline(deadline)
    end = perf_counter()
    if (response is None or response.finish_reason not in {"length", "stop"}
            or response.generation_tokens != len(segments) or not 1 <= len(segments) <= max_tokens):
        raise MeasureError("invalid_generation_stream")
    output = "".join(segments)
    if constraint is not None and not verify(emitted, output, response.finish_reason):
        raise MeasureError("invalid_generation_stream")
    return output, {
        "prompt_tokens": len(ids),
        "prefill_has_effect": len(ids) > prefill_step_size + 1,
        "generated_tokens": len(segments),
        "finish_reason": response.finish_reason,
        "ttft_ms": round(1000 * (first - start), 6),
        "duration_ms": round(1000 * (end - start), 6),
    }


def _file_identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid asset file")
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _identity(path, *, adapter):
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError("invalid asset directory")
    info = path.stat()
    names = ADAPTER_FILES if adapter else sorted(
        file.name for file in path.iterdir() if file.suffix in MODEL_EXTENSIONS
    )
    return (info.st_dev, info.st_ino), tuple((name, _file_identity(path / name)) for name in names)


class PinnedAssets:
    """Hash local assets once, checking their filesystem identity around each load."""

    def __init__(self):
        self._assets = {}

    def _pin(self, value, *, adapter):
        path = Path(value)
        key = (path, adapter)
        identity = _identity(path, adapter=adapter)
        if key not in self._assets:
            digest = (adapter_fingerprint if adapter else model_fingerprint)(path)
            if _identity(path, adapter=adapter) != identity:
                raise ValueError("asset changed during fingerprinting")
            self._assets[key] = identity, digest
        original, digest = self._assets[key]
        if identity != original:
            raise ValueError("asset changed after fingerprinting")
        return digest

    def identify(self, variant):
        adapter = variant.get("adapter")
        return {
            "model_sha256": self._pin(variant["model"], adapter=False),
            "adapter_sha256": self._pin(adapter, adapter=True) if adapter is not None else None,
        }

    def verify(self, variant):
        try:
            return self.identify(variant)
        except (OSError, ValueError):
            raise MeasureError("integrity_error") from None


@dataclass
class MLXMeasure:
    mx: object
    load: object
    stream_generate: object
    assets: PinnedAssets

    def __call__(self, variant, cases, repetition, split, deadline):
        try:
            return self._measure(variant, cases, deadline)
        except (MeasureError, ConstraintStreamError) as error:
            return {"ok": False, "error": {"code": str(error)}}
        except Exception:
            return {"ok": False, "error": {"code": "measurement_failed"}}
        finally:
            gc.collect()
            self.mx.clear_cache()

    def _measure(self, variant, cases, deadline):
        _check_deadline(deadline)
        assets = self.assets.verify(variant)
        self.mx.set_cache_limit(CACHE_LIMIT_BYTES)
        self.mx.reset_peak_memory()
        cache_peak = 0

        def sample_memory():
            nonlocal cache_peak
            cache_peak = max(cache_peak, self.mx.get_cache_memory())

        start = perf_counter()
        model, tokenizer = self.load(
            variant["model"], adapter_path=variant.get("adapter"),
            tokenizer_config={"trust_remote_code": False, "local_files_only": True},
            trust_remote_code=False,
        )
        model.eval()
        self.mx.eval(model.parameters())
        load_ms = round(1000 * (perf_counter() - start), 6)
        self.assets.verify(variant)
        sample_memory()
        _check_deadline(deadline)
        prompts = [(case, _prompt_ids(tokenizer, variant, case["input"])) for case in cases]
        max_tokens = variant["max_tokens"]
        prefill_step_size = variant["prefill_step_size"]
        constraint = None
        constraint_metadata = {}
        if "output_constraint" in variant:
            start = perf_counter()
            try:
                constraint = compile_constraint(self.mx, tokenizer, variant["output_constraint"], max_tokens)
            except ValueError:
                raise MeasureError("invalid_output_constraint") from None
            constraint_metadata["decoding_setup_ms"] = round(1000 * (perf_counter() - start), 6)
        warmup_tokens = max_tokens if constraint is not None else min(max_tokens, MAX_WARMUP_TOKENS)
        _, warmup = _generate(
            self.stream_generate, model, tokenizer, prompts[0][1], warmup_tokens,
            prefill_step_size, deadline, sample_memory, constraint,
        )
        records = []
        used_tokens = warmup["generated_tokens"]
        grader = variant["grader"]["kind"]
        for case, ids in prompts:
            output, metrics = _generate(
                self.stream_generate, model, tokenizer, ids, max_tokens,
                prefill_step_size, deadline, sample_memory, constraint,
            )
            used_tokens += metrics["generated_tokens"]
            records.append({
                "case_id": case["id"],
                "correct": grade(grader, output, case["expected"]),
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                **grade_diagnostics(grader, output, case["expected"]),
                **metrics,
                **({"output_constraint_applied": True, "constraint_verified": True} if constraint is not None else {}),
            })
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "ok": True, "records": records, **assets, "load_ms": load_ms,
            **constraint_metadata,
            "used_generated_tokens": used_tokens,
            "warmup_generated_tokens": warmup["generated_tokens"],
            "peak_mlx_active_bytes": self.mx.get_peak_memory(),
            "peak_mlx_cache_sampled_bytes": cache_peak,
            "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        }
