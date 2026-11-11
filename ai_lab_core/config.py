"""Non-secret local connection settings. Never reads Hermes credentials."""

from __future__ import annotations

import json
import os
from pathlib import Path
from .profile import config_path


def load_settings() -> dict:
    defaults = {
        "model_roots": [str(Path.home() / ".cache/huggingface/hub"), str(Path.home() / "models")],
        "runtime_bin_dirs": [],
        "endpoints": [{"name": "ollama-local", "kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                      {"name": "llama-local", "kind": "llama_cpp", "base_url": "http://127.0.0.1:8080"}],
    }
    path = config_path('connections.json', 'AI_LAB_CONFIG')
    if not path.exists():
        return defaults
    if path.stat().st_size > 32768:
        raise ValueError("invalid connection settings")
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict) or set(settings) - {"model_roots", "endpoints", "runtime_bin_dirs"}:
        raise ValueError("invalid connection settings")
    roots = settings.get("model_roots", defaults["model_roots"])
    endpoints = settings.get("endpoints", defaults["endpoints"])
    bins = settings.get("runtime_bin_dirs", defaults["runtime_bin_dirs"])
    if not isinstance(roots, list) or len(roots) > 12 or any(not isinstance(p, str) or not Path(p).expanduser().is_absolute() for p in roots):
        raise ValueError("invalid model roots")
    if not isinstance(endpoints, list) or len(endpoints) > 6:
        raise ValueError("invalid endpoints")
    if not isinstance(bins, list) or len(bins) > 12 or any(not isinstance(p, str) or not Path(p).expanduser().is_absolute() for p in bins):
        raise ValueError("invalid runtime directories")
    for item in endpoints:
        if not isinstance(item, dict) or set(item) != {"name", "kind", "base_url"} or any(not isinstance(v, str) or len(v) > 2048 for v in item.values()):
            raise ValueError("invalid endpoint")
    return {"model_roots": [str(Path(p).expanduser()) for p in roots], "endpoints": endpoints,
            "runtime_bin_dirs": [str(Path(p).expanduser()) for p in bins]}
