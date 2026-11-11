"""Private MLX process for one serialized inference benchmark."""
import fcntl
import hashlib
import hmac
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
from time import perf_counter

# The worker is launched as an isolated script by the configured MLX Python.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_lab_core.inference_benchmark import (
    KIND, MAX_PROMPT_TOKENS, MAX_REQUEST_BYTES, MAX_WARMUP_TOKENS, METRICS, TIMEOUT_SECONDS,
    benchmark_failure, generation_budget, workload,
)
from ai_lab_core.profile import device_lock_path
from ai_lab_core.skill_worker import CACHE_LIMIT_BYTES, adapter_fingerprint, model_fingerprint


def _percentile(values, fraction):
    index = (len(values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return round(values[low] + (values[high] - values[low]) * (index - low), 6)


def summarize(traces):
    summary = {}
    for metric in METRICS:
        values = sorted(trace[metric] for trace in traces if trace[metric] is not None)
        summary[metric] = {
            "n": len(values),
            "min": values[0] if values else None,
            "p50": _percentile(values, 0.50) if values else None,
            "p95": _percentile(values, 0.95) if values else None,
            "max": values[-1] if values else None,
        }
    return summary


def _backend_rate(value):
    return round(value, 6) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _generate(stream_generate, model, tokenizer, ids, max_tokens, sample_memory):
    output = hashlib.sha256()
    count = 0
    first = last = None
    response = None
    start = perf_counter()
    for response in stream_generate(model, tokenizer, ids, max_tokens=max_tokens):
        last = perf_counter()
        if first is None:
            first = last
        count += 1
        output.update(response.text.encode("utf-8"))
        sample_memory()
    end = perf_counter()
    if (response is None or first is None or response.finish_reason not in {"length", "stop"}
            or response.generation_tokens != count or not 1 <= count <= max_tokens):
        raise ValueError("unexpected MLX token stream")
    decode_seconds = last - first
    return {
        "output_sha256": output.hexdigest(),
        "prompt_tokens": len(ids),
        "sampled_tokens": count,
        "finish_reason": response.finish_reason,
        "ttft_ms": round(1000 * (first - start), 6),
        "duration_ms": round(1000 * (end - start), 6),
        "tpot_ms": round(1000 * decode_seconds / (count - 1), 6) if count > 1 else None,
        "decode_tokens_per_second": round((count - 1) / decode_seconds, 6) if decode_seconds > 0 else None,
        "backend_prompt_tokens_per_second": _backend_rate(getattr(response, "prompt_tps", None)),
        "backend_decode_tokens_per_second": _backend_rate(getattr(response, "generation_tps", None)),
    }


class InvalidChatTemplate(ValueError):
    pass


class InvalidPromptTokens(ValueError):
    pass


def _prompt_ids(tokenizer, prompt, prompt_format):
    if prompt_format == "raw":
        add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        if not getattr(tokenizer, "has_chat_template", False):
            raise InvalidChatTemplate
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            )
        except Exception as error:
            raise InvalidChatTemplate from error
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise InvalidChatTemplate
    if not 1 <= len(ids) <= MAX_PROMPT_TOKENS:
        raise InvalidPromptTokens
    return ids


def _measure(payload, model_hash, adapter_hash):
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mx.reset_peak_memory()
    memory = {
        "peak_mlx_cache_sampled_bytes": 0,
        "peak_mlx_active_plus_cache_sampled_bytes": 0,
    }

    def sample_memory():
        cache = mx.get_cache_memory()
        active = mx.get_active_memory()
        memory["peak_mlx_cache_sampled_bytes"] = max(memory["peak_mlx_cache_sampled_bytes"], cache)
        memory["peak_mlx_active_plus_cache_sampled_bytes"] = max(
            memory["peak_mlx_active_plus_cache_sampled_bytes"], active + cache,
        )

    start = perf_counter()
    model, tokenizer = load(
        payload["model"], adapter_path=payload["adapter"],
        tokenizer_config={"trust_remote_code": False}, trust_remote_code=False,
    )
    model.eval()
    mx.eval(model.parameters())
    load_ms = round(1000 * (perf_counter() - start), 6)
    sample_memory()
    prompts = []
    prompt_format = payload["prompt_format"]
    for prompt in payload["prompts"]:
        try:
            ids = _prompt_ids(tokenizer, prompt, prompt_format)
        except InvalidChatTemplate:
            return benchmark_failure("invalid_prompt_format")
        except InvalidPromptTokens:
            return benchmark_failure("invalid_prompt_tokens")
        prompts.append((hashlib.sha256(prompt.encode("utf-8")).hexdigest(), ids))

    max_tokens = payload["max_tokens"]
    warmup_tokens = min(max_tokens, MAX_WARMUP_TOKENS)
    _generate(stream_generate, model, tokenizer, prompts[0][1], warmup_tokens, sample_memory)
    sample_memory()
    traces = []
    for repetition in range(1, payload["repetitions"] + 1):
        for prompt_index, (digest, ids) in enumerate(prompts, 1):
            trace = _generate(stream_generate, model, tokenizer, ids, max_tokens, sample_memory)
            traces.append({
                "prompt_index": prompt_index, "prompt_sha256": digest,
                "repetition": repetition, **trace,
            })
            sample_memory()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "kind": KIND, "ok": True, "quality_evaluated": False, "optimization_gain_verified": False,
        "model_sha256": model_hash, "adapter_sha256": adapter_hash,
        "mlx_version": version("mlx"), "mlx_lm_version": version("mlx-lm"),
        "load_ms": load_ms, "max_tokens": max_tokens, "repetitions": payload["repetitions"],
        "prompt_count": len(prompts), "sample_count": len(traces), "concurrency": 1,
        "sampler": "greedy", "prompt_format": prompt_format,
        "request_sha256": payload["request_sha256"],
        "generated_token_budget_including_warmup": generation_budget(
            len(prompts), max_tokens, payload["repetitions"],
        ),
        "warmup_runs": 1, "warmup_max_tokens": warmup_tokens, "warmup_prompt_index": 1,
        "cache_conditions": {
            "process": "new_process_then_model_loaded_once",
            "allocator": "warmed_before_measurements_and_reused",
            "kv": "fresh_per_generation",
            "os_files": "uncontrolled_and_read_during_fingerprinting",
        },
        "timeout_seconds": TIMEOUT_SECONDS,
        "mlx_cache_limit_bytes": CACHE_LIMIT_BYTES,
        "peak_mlx_active_bytes": mx.get_peak_memory(),
        **memory,
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "memory_scope": "process high-water RSS and MLX memory include load and warmup; cache sampled after load and at generation events and boundaries",
        "timing_scope": "TTFT is first emitted token event, including immediate EOS; duration ends after stream completion; input tokenization excluded",
        "token_accounting": "sampled tokens include EOS; TPOT and wall decode rate use the interval between the first and last token events",
        "backend_rates": "SDK rates taken from the final stream event; denominators differ from wall measurements",
        "quantiles": "descriptive linear interpolation over the observed sample; each metric has its own n",
        "metrics": summarize(traces), "traces": traces,
    }


def _run(payload):
    lock = device_lock_path()
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return benchmark_failure("compute_busy")
        try:
            model = Path(payload["model"])
            if model.resolve(strict=True) != model:
                raise ValueError("model path changed")
            model_hash = model_fingerprint(model)
        except (OSError, ValueError):
            return benchmark_failure("invalid_model")
        adapter_hash = None
        if payload["adapter"] is not None:
            try:
                adapter = Path(payload["adapter"])
                if adapter.resolve(strict=True) != adapter:
                    raise ValueError("adapter path changed")
                adapter_hash = adapter_fingerprint(adapter)
            except (AttributeError, OSError, ValueError):
                return benchmark_failure("invalid_adapter")
        try:
            return _measure(payload, model_hash, adapter_hash)
        except ImportError:
            return benchmark_failure("runtime_unavailable")


def _read_request(path, expected, workspace):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > MAX_REQUEST_BYTES):
            raise ValueError("request file")
        data = stream.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected):
        raise ValueError("request digest")
    payload = json.loads(data)
    if payload["workspace"] != str(workspace):
        raise ValueError("workspace")
    workload(payload["prompts"], payload["max_tokens"], payload["repetitions"], payload["prompt_format"])
    payload["request_sha256"] = expected
    return payload


def _write_report(path, report):
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    # Silence Python and native backend logs, which can contain model paths or inputs.
    with open(os.devnull, "wb") as discard:
        os.dup2(discard.fileno(), 1)
        os.dup2(discard.fileno(), 2)
    if len(sys.argv) != 3:
        return 1
    path = Path(sys.argv[1])
    workspace = Path.cwd().resolve()
    try:
        directory = path.parent
        info = directory.stat()
        if (path.name != "request.json" or workspace not in directory.parents
                or directory.resolve(strict=True) != directory or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            return 1
        report_path = directory / "report.json"
        _write_report(report_path, benchmark_failure("incomplete"))
        try:
            payload = _read_request(path, sys.argv[2], workspace)
        except (KeyError, OSError, TypeError, ValueError, RecursionError):
            report = benchmark_failure("invalid_request")
        else:
            try:
                report = _run(payload)
            except Exception:
                report = benchmark_failure("runtime_failed")
        _write_report(report_path, report)
        return 0 if report["ok"] else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
