"""Submit and execute one frozen, local PAWS LoRA training recipe."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

if __name__ == "__main__":
    # The configured MLX interpreter starts this script in isolated mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "ai_lab_core"

from . import jobs
from .endpoint import failure
from .integration_config import permitted_path
from .profile import device_lock_path
from .skill_experiment import _write_json
from .skill_runtime import (
    PAIR_IMPROVEMENT_PROFILE, PAIR_LABELS, _sha256, suffix, train, training_settings,
)
from .skill_worker import CACHE_LIMIT_BYTES, adapter_fingerprint, model_fingerprint

KIND = "ai_lab.lora_improve.v1"
TIMEOUT_SECONDS = 1800
MAX_REQUEST_BYTES = 32768
MAX_SPLIT_BYTES = 32 * 1024**2
SPLIT_ROWS = {"train.jsonl": 8192, "valid.jsonl": 1024}
SOURCE_FILES = ("improvement_training.py", "skill_runtime.py", "skill_worker.py", "skill_experiment.py")
VARIANT = 0


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _bounded_file(path, limit, *, private=False):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("file")
        if private and (info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1):
            raise ValueError("private file")
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError("file size")
    return data


def _data_files(directory):
    if directory.resolve(strict=True) != directory or {p.name for p in directory.iterdir()} != set(SPLIT_ROWS):
        raise ValueError("training splits")
    return {name: _bounded_file(directory / name, MAX_SPLIT_BYTES) for name in SPLIT_ROWS}


def _data_hashes(files):
    return {name: _digest(data) for name, data in files.items()}


def _source_hashes():
    directory = Path(__file__).resolve().parent
    return {name: _digest(_bounded_file(directory / name, 1024**2)) for name in SOURCE_FILES}


def _model_hash(model):
    if model.resolve(strict=True) != model:
        raise ValueError("model path")
    return model_fingerprint(model)


def start(parameters, settings):
    """Pin local inputs and submit a job through the current Hermes profile."""
    try:
        if set(parameters) != {"model", "data_dir", "output_dir"}:
            raise ValueError("arguments")
        workspace = Path(settings["workspace_root"]).resolve(strict=True)
        model = permitted_path(parameters["model"], settings)
        data = permitted_path(parameters["data_dir"], settings)
        output = permitted_path(parameters["output_dir"], settings, output=True, must_exist=False)
        original_output = Path(parameters["output_dir"]).expanduser()
        if (
            not workspace.is_dir() or not output.parent.is_dir() or output.exists()
            or original_output.is_symlink() or not model.is_dir() or not data.is_dir()
            or any(output == path or output in path.parents or path in output.parents for path in (model, data))
        ):
            raise ValueError("paths")

        python = Path(settings.get("backend_paths", {}).get("mlx.python", ""))
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            return failure("configuration_required", "Configure an executable as backend_paths.mlx.python.")

        payload = {
            "workspace": str(workspace), "model": str(model), "data_dir": str(data),
            "model_sha256": _model_hash(model), "data_sha256": _data_hashes(_data_files(data)),
            "source_sha256": _source_hashes(),
            "training_profile": PAIR_IMPROVEMENT_PROFILE,
            "settings": training_settings(VARIANT, profile=PAIR_IMPROVEMENT_PROFILE),
        }
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("request size")

        output.mkdir(mode=0o700)
        request_path = output / "request.json"
        with os.fdopen(os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(encoded)
        report_path = output / "report.json"
        adapter_dir = output / "adapter"
        request_hash = _digest(encoded)
        plan = {
            "operation": "lora.improve", "backend": "mlx", "cwd": str(workspace),
            "command": [str(python), "-I", "-B", str(Path(__file__).resolve()), str(request_path), request_hash],
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
            "timeout_seconds": TIMEOUT_SECONDS,
            "outputs": [{"path": str(report_path), "kind": "file"},
                        {"path": str(adapter_dir), "kind": "directory"}],
            "metadata": {"profile_state": settings["state_dir"], "report_path": str(report_path),
                         "adapter_path": str(adapter_dir), "request_sha256": request_hash, "cloud": False},
        }
        return jobs.submit(plan, state_dir=settings["state_dir"])
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return failure("invalid_training", "Supply a local model, prepared train and validation data, and a new output directory.")


def _read_request(path, expected, workspace):
    data = _bounded_file(path, MAX_REQUEST_BYTES, private=True)
    if not hmac.compare_digest(_digest(data), expected):
        raise ValueError("request digest")
    payload = json.loads(data)
    if (
        set(payload) != {"workspace", "model", "data_dir", "model_sha256", "data_sha256",
                         "source_sha256", "training_profile", "settings"}
        or payload["workspace"] != str(workspace)
        or payload["training_profile"] != PAIR_IMPROVEMENT_PROFILE
        or payload["settings"] != training_settings(VARIANT, profile=PAIR_IMPROVEMENT_PROFILE)
        or not _sha256(payload["model_sha256"])
        or set(payload["data_sha256"]) != set(SPLIT_ROWS)
        or not all(_sha256(value) for value in payload["data_sha256"].values())
        or set(payload["source_sha256"]) != set(SOURCE_FILES)
        or not all(_sha256(value) for value in payload["source_sha256"].values())
        or not all(Path(payload[key]).is_absolute() for key in ("model", "data_dir"))
    ):
        raise ValueError("request")
    return payload


def _load_tokenizer(model):
    # Match the tokenizer wrapper and local model defaults used by MLX-LM LoRA.
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(model, {"trust_remote_code": False, "local_files_only": True})


def _inspect_split(data, tokenizer, row_limit, settings):
    rows = data.splitlines()
    batch_size = settings["batch_size"]
    if not batch_size <= len(rows) <= row_limit or len(rows) % batch_size:
        raise ValueError("row count")
    max_tokens = 0
    completions = {suffix(label) for label in PAIR_LABELS}
    for line in rows:
        row = json.loads(line.decode("utf-8"))
        chat = row["messages"]
        if (
            set(row) != {"messages"} or type(chat) is not list or len(chat) != 3
            or any(type(message) is not dict or set(message) != {"role", "content"}
                   or type(message["content"]) is not str or not message["content"].strip()
                   for message in chat)
            or [message["role"] for message in chat] != ["system", "user", "assistant"]
            or chat[-1]["content"] not in completions
        ):
            raise ValueError("chat")
        full = tokenizer.apply_chat_template(chat, tools=None, return_dict=False)
        prompt = tokenizer.apply_chat_template(chat[:-1], tools=None, add_generation_prompt=True, return_dict=False)
        if (
            not 0 < len(prompt) < len(full) <= settings["max_seq_length"]
            or full[:len(prompt)] != prompt
        ):
            raise ValueError("chat tokens")
        max_tokens = max(max_tokens, len(full))
    return {"rows": len(rows), "max_tokens": max_tokens}


def _failed(code, message):
    return {"kind": KIND, **failure(code, message)}


def _run(payload, output, request_hash, progress):
    model = Path(payload["model"])
    data = Path(payload["data_dir"])
    try:
        files = _data_files(data)
        if (
            _source_hashes() != payload["source_sha256"]
            or _data_hashes(files) != payload["data_sha256"]
            or _model_hash(model) != payload["model_sha256"]
        ):
            raise ValueError("source changed")
    except (OSError, RuntimeError, ValueError):
        return _failed("source_changed", "The pinned code, model or data differs from the submitted inputs.")

    progress("preflight")
    tokenizer = _load_tokenizer(model)
    try:
        splits = {name: _inspect_split(files[name], tokenizer, row_limit, payload["settings"])
                  for name, row_limit in SPLIT_ROWS.items()}
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError):
        return _failed("invalid_training_data", "The prepared chat splits violate the row, prompt or token limits.")
    del tokenizer

    # The worker reads a private snapshot of the verified bytes. The original
    # sources are checked again before accepting the result.
    adapter = output / "adapter"
    with tempfile.TemporaryDirectory(prefix=".training-", dir=output) as directory:
        snapshot = Path(directory)
        for name, content in files.items():
            (snapshot / name).write_bytes(content)
        progress("training")
        result = train(
            python=sys.executable, model=model, lock_path=device_lock_path(),
            data_dir=snapshot, adapter_dir=adapter, variant=VARIANT,
            expected_model=payload["model_sha256"], timeout=TIMEOUT_SECONDS,
            training_profile=PAIR_IMPROVEMENT_PROFILE,
        )
    if not result["ok"]:
        return {"kind": KIND, **result}
    progress("verifying")
    try:
        if (
            _source_hashes() != payload["source_sha256"]
            or _data_hashes(_data_files(data)) != payload["data_sha256"]
            or _model_hash(model) != payload["model_sha256"]
        ):
            raise ValueError("source changed")
    except (OSError, RuntimeError, ValueError):
        return _failed("source_changed", "The pinned code, model or data changed during training.")
    if adapter_fingerprint(adapter) != result["adapter_sha256"] or result.get("mlx_cache_limit_bytes") != CACHE_LIMIT_BYTES:
        return _failed("invalid_training_output", "The trained adapter or cache setting could not be verified.")
    return {
        "ok": True, "kind": KIND, "request_sha256": request_hash, "data_sha256": payload["data_sha256"],
        "source_sha256": payload["source_sha256"],
        "splits": splits, "training_profile": PAIR_IMPROVEMENT_PROFILE, "settings": payload["settings"],
        "model_sha256": payload["model_sha256"], "adapter_sha256": result["adapter_sha256"],
        "adapter_bytes": result["adapter_bytes"], "elapsed_seconds": result["elapsed_seconds"],
        "mlx_version": result["mlx_version"], "mlx_lm_version": result["mlx_lm_version"],
        "peak_mlx_memory_bytes": result["peak_mlx_memory_bytes"], "mlx_cache_limit_bytes": CACHE_LIMIT_BYTES,
    }


def main():
    if len(sys.argv) != 3:
        return 1
    request_path = Path(sys.argv[1])
    output = request_path.parent
    workspace = Path.cwd().resolve()
    try:
        if (
            request_path.name != "request.json" or workspace not in output.parents
            or output.resolve(strict=True) != output
        ):
            return 1
        info = output.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            return 1
        report_path = output / "report.json"
        _write_json(report_path, _failed("incomplete", "Training did not complete. Check the durable job state."))
        try:
            payload = _read_request(request_path, sys.argv[2], workspace)
        except (AttributeError, KeyError, OSError, TypeError, ValueError, RecursionError):
            report = _failed("invalid_request", "The pinned training request could not be verified.")
        else:
            # Keep only static progress messages in the durable logs. Tokenizer
            # and native backend errors can contain raw paths and input text.
            with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as progress:
                with open(os.devnull, "w") as discard, redirect_stdout(discard), redirect_stderr(discard):
                    os.dup2(discard.fileno(), 1)
                    os.dup2(discard.fileno(), 2)
                    try:
                        report = _run(payload, output, sys.argv[2], lambda stage: print(stage, file=progress, flush=True))
                    except ImportError:
                        report = _failed("runtime_unavailable", "The configured MLX runtime is unavailable.")
                    except Exception:
                        report = _failed("runtime_failed", "Local LoRA training failed.")
        _write_json(report_path, report)
        return 0 if report["ok"] else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
