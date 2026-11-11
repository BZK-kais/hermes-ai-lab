"""Submit bounded local MLX measurements using the profile's durable jobs."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from . import jobs
from .endpoint import failure
from .integration_config import permitted_path
from .profile import PROJECT


KIND = "ai_lab.mlx_benchmark.v1"
MAX_REPORT_BYTES = 4 * 1024**2
MAX_REQUEST_BYTES = 512 * 1024
MAX_PROMPT_BYTES = 8192
MAX_PROMPT_TOKENS = 2048
MAX_WARMUP_TOKENS = 8
MAX_GENERATED_TOKENS = 8192
PROMPT_FORMATS = ("chat", "raw")
TIMEOUT_SECONDS = 900
ERRORS = {
    "incomplete": "The benchmark did not complete. Check the durable job state.",
    "invalid_request": "The benchmark request could not be verified.",
    "compute_busy": "The local MLX device is in use.",
    "invalid_model": "The local model could not be fingerprinted.",
    "invalid_adapter": "The local LoRA adapter could not be fingerprinted.",
    "runtime_unavailable": "The configured MLX runtime is unavailable.",
    "runtime_failed": "The local MLX benchmark failed.",
    "invalid_prompt_format": "The model tokenizer could not apply its chat template.",
    "invalid_prompt_tokens": "A prompt exceeds the supported token limits or encodes to no tokens.",
}
METRICS = (
    "ttft_ms", "duration_ms", "tpot_ms", "decode_tokens_per_second",
    "backend_prompt_tokens_per_second", "backend_decode_tokens_per_second",
)


def benchmark_failure(code):
    return {
        "kind": KIND, "concurrency": 1,
        "quality_evaluated": False, "optimization_gain_verified": False,
        **failure(code, ERRORS[code]),
    }


def generation_budget(prompt_count, max_tokens, repetitions):
    return prompt_count * max_tokens * repetitions + min(max_tokens, MAX_WARMUP_TOKENS)


def workload(prompts, max_tokens, repetitions, prompt_format):
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError("prompt format")
    if (not isinstance(prompts, list) or not 1 <= len(prompts) <= 8
            or any(not isinstance(p, str) or not p.strip()
                   or len(p) > MAX_PROMPT_BYTES or len(p.encode("utf-8")) > MAX_PROMPT_BYTES
                   for p in prompts)):
        raise ValueError("prompts")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 256:
        raise ValueError("max_tokens")
    if type(repetitions) is not int or not 2 <= repetitions <= 10:
        raise ValueError("repetitions")
    if generation_budget(len(prompts), max_tokens, repetitions) > MAX_GENERATED_TOKENS:
        raise ValueError("generation budget")


def _start(request, settings):
    required = {"action", "model", "prompts", "output_dir"}
    if not required <= request.keys() or request.keys() - required - {"adapter", "max_tokens", "repetitions", "prompt_format"}:
        raise ValueError("arguments")
    prompts = request["prompts"]
    max_tokens = request.get("max_tokens", 64)
    repetitions = request.get("repetitions", 3)
    prompt_format = request.get("prompt_format", "chat")
    workload(prompts, max_tokens, repetitions, prompt_format)

    workspace = Path(settings["workspace_root"]).resolve(strict=True)
    output = permitted_path(request["output_dir"], settings, output=True, must_exist=False)
    original_output = Path(request["output_dir"]).expanduser()
    model = permitted_path(request["model"], settings)
    adapter = permitted_path(request["adapter"], settings) if "adapter" in request else None
    if (not workspace.is_dir() or not output.parent.is_dir() or output.exists()
            or original_output.is_symlink()):
        raise ValueError("output")
    for path in (model, adapter):
        if path is not None and (not path.is_dir() or output in path.parents or path in output.parents):
            raise ValueError("model or adapter")

    python = Path(settings.get("backend_paths", {}).get("mlx.python", ""))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        return failure("configuration_required", "Configure an executable as backend_paths.mlx.python.")

    state_dir = settings["state_dir"]
    payload = {
        "workspace": str(workspace), "model": str(model), "adapter": str(adapter) if adapter else None,
        "prompts": prompts, "max_tokens": max_tokens, "repetitions": repetitions, "prompt_format": prompt_format,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("request size")
    output.mkdir(mode=0o700)
    request_path = output / "request.json"
    with os.fdopen(os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data)
    report_path = output / "report.json"
    plan = {
        "operation": "inference.benchmark", "backend": "mlx", "cwd": str(workspace),
        "command": [str(python), "-I", "-B", str(Path(__file__).with_name("inference_worker.py")),
                    str(request_path), hashlib.sha256(data).hexdigest()],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "timeout_seconds": TIMEOUT_SECONDS,
        "outputs": [{"path": str(report_path), "kind": "file"}],
        "metadata": {"profile_state": state_dir, "report_path": str(report_path), "cloud": False},
    }
    return jobs.submit(plan, state_dir=state_dir)


def _report(request, settings):
    if set(request) != {"action", "report_path"}:
        raise ValueError("arguments")
    path = permitted_path(request["report_path"], settings, output=True)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > MAX_REPORT_BYTES:
            raise ValueError("report file")
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("report size")
    report = json.loads(data)
    if not isinstance(report, dict) or report.get("kind") != KIND or type(report.get("ok")) is not bool:
        raise ValueError("report kind")
    if not report["ok"]:
        return {**benchmark_failure(report["error"]["code"]), "report_path": str(path)}

    # Keep traces in the local file. Only return typed summary fields, even
    # when the supplied file has been edited since the benchmark ran.
    summary = {}
    for key in ("load_ms", "sample_count", "prompt_count", "max_tokens", "repetitions",
                "peak_mlx_active_bytes", "peak_mlx_cache_sampled_bytes",
                "peak_mlx_active_plus_cache_sampled_bytes", "peak_process_rss_bytes"):
        summary[key] = _number(report[key])
    prompt_format = report["prompt_format"]
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError("prompt format")
    summary["prompt_format"] = prompt_format
    summary["generated_token_budget_including_warmup"] = _number(report["generated_token_budget_including_warmup"])
    for key in ("model_sha256", "adapter_sha256", "request_sha256"):
        value = report[key]
        if value is not None and (not isinstance(value, str) or len(value) != 64
                                  or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("fingerprint")
        summary[key] = value
    summary["metrics"] = {}
    for metric in METRICS:
        values = report["metrics"][metric]
        summary["metrics"][metric] = {
            key: _number(values[key]) for key in ("n", "min", "p50", "p95", "max")
        }
    return {
        "ok": True, "kind": KIND, "report_path": str(path), "summary": summary,
        "concurrency": 1, "quality_evaluated": False, "optimization_gain_verified": False,
    }


def _number(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError("metric")
    return value


def dispatch(request, settings):
    try:
        if isinstance(request, dict):
            if request.get("action") == "start":
                return _start(request, settings)
            if request.get("action") == "report":
                return _report(request, settings)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    return failure("invalid_benchmark", "Supply a bounded local workload or a benchmark report inside this profile workspace.")
