"""Environment policy for isolated runtimes."""
import os

_INHERITED = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}
_OVERRIDES = {
    "HF_HUB_OFFLINE": ("0", "1"),
    "HF_DATASETS_OFFLINE": ("0", "1"),
    "TRANSFORMERS_OFFLINE": ("0", "1"),
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": ("1",),
    "PYTHONUNBUFFERED": ("1",),
    "WANDB_MODE": ("disabled",),
}


def validate_overrides(overrides):
    if not isinstance(overrides, dict) or any(
        value not in _OVERRIDES.get(key, ()) for key, value in overrides.items()
    ):
        raise ValueError("Unsupported runtime environment override.")


def runtime_env(overrides=None):
    if overrides is None:
        overrides = {}
    validate_overrides(overrides)
    inherited = {key: os.environ[key] for key in _INHERITED if key in os.environ}
    return {**inherited, **_DEFAULTS, **overrides}
