"""Explicit integration references; credentials are resolved only by adapters."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .config import load_settings
from .profile import config_path, data_root, bound_home

PROJECT = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")


def _absolute(value):
    if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
        raise ValueError("absolute path required")
    return str(Path(value).expanduser().resolve())


def _executable(value):
    if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
        raise ValueError('absolute executable required')
    # Resolving a venv's python symlink loses its pyvenv.cfg and installed
    # packages. Keep the operator-selected executable path, as subprocess does.
    return str(Path(value).expanduser().absolute())


def load_integrations():
    settings = load_settings()
    path = config_path('integrations.json', 'AI_LAB_INTEGRATIONS')
    return parse_integrations(settings, read_integrations(path))


def read_integrations(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError("configuration too large")
    return json.loads(data.decode('utf-8'))


def default_data_roots(workspace_root):
    examples = PROJECT / 'examples'
    if not examples.is_dir():
        examples = Path(__file__).resolve().with_name('assets')
    return [str(examples), workspace_root]


def parse_integrations(settings, data):
    allowed = {"workspace_root", "data_roots", "integrations", "backend_paths", "backend_recipe_args", "state_dir", "runtime_bin_dirs", "transfer_max_bytes"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("invalid integrations settings")
    settings = dict(settings)
    root = data_root()
    if bound_home() is not None and root.is_symlink():
        raise ValueError('profile data root must not be a symbolic link')
    settings["workspace_root"] = _absolute(data.get("workspace_root", str(root / 'workspace')))
    state_default = str(root / 'state')
    if bound_home() is None:
        state_default = os.environ.get('AI_LAB_STATE_DIR', state_default)
    settings["state_dir"] = _absolute(data.get("state_dir", state_default))
    if bound_home() is not None:
        # Native plugin configuration can add read-only model/data roots but
        # cannot redirect writable state into another profile.
        for key in ('workspace_root', 'state_dir'):
            candidate = Path(settings[key])
            if candidate == root.resolve() or root.resolve() not in candidate.parents:
                raise ValueError('profile state must stay within the profile')
    roots = data.get("data_roots", default_data_roots(settings["workspace_root"]))
    if not isinstance(roots, list) or len(roots) > 32:
        raise ValueError("invalid data roots")
    settings["data_roots"] = [_absolute(p) for p in roots]
    settings["model_roots"] = list(dict.fromkeys(settings["model_roots"] + [settings["workspace_root"]]))
    bins = data.get("runtime_bin_dirs", [])
    if not isinstance(bins, list) or len(bins) > 16:
        raise ValueError("invalid runtime directories")
    settings["runtime_bin_dirs"] = list(dict.fromkeys(settings["runtime_bin_dirs"] + [_absolute(p) for p in bins]))
    paths = data.get("backend_paths", {})
    args = data.get("backend_recipe_args", {})
    if not isinstance(paths, dict) or len(paths) > 40 or not isinstance(args, dict) or len(args) > 40:
        raise ValueError("invalid backend bindings")
    settings["backend_paths"] = {k: _executable(v) for k, v in paths.items() if isinstance(k, str) and IDENTIFIER.fullmatch(k)}
    if len(settings["backend_paths"]) != len(paths):
        raise ValueError("invalid backend name")
    for key, value in args.items():
        if not IDENTIFIER.fullmatch(key) or not isinstance(value, list) or len(value) > 60 or any(not isinstance(x, str) or len(x) > 2048 for x in value):
            raise ValueError("invalid recipe args")
    settings["backend_recipe_args"] = args
    cap = data.get("transfer_max_bytes", 50 * 1024**3)
    if type(cap) is not int or not 1 <= cap <= 50 * 1024**3:
        raise ValueError("invalid transfer size limit")
    settings["transfer_max_bytes"] = cap
    connections = data.get("integrations", [])
    if not isinstance(connections, list) or len(connections) > 32:
        raise ValueError("invalid connections")
    seen = set()
    parsed_connections = []
    for connection in connections:
        if not isinstance(connection, dict) or not IDENTIFIER.fullmatch(connection.get("id", "")) or not isinstance(connection.get("type"), str):
            raise ValueError("invalid connection")
        if connection["id"] in seen or connection["id"] in {e["name"] for e in settings["endpoints"]}:
            raise ValueError("duplicate connection")
        seen.add(connection["id"])
        if any(k.lower() in {"token", "api_key", "password", "secret", "headers", "authorization"} for k in connection):
            raise ValueError("use credential references")
        connection = dict(connection)
        if root == PROJECT / '.local' and connection["type"] in {"runpod", "hf_jobs", "hf_endpoints", "hf_storage", "gcs", "s3"}:
            interpreter = PROJECT / ".venv/bin/python"
            if interpreter.is_file():
                connection.setdefault("python", str(interpreter))
        parsed_connections.append(connection)
    settings["integrations"] = parsed_connections
    return settings


def permitted_path(value, settings, *, output=False, must_exist=True):
    path = Path(_absolute(value))
    roots = [settings["workspace_root"]] if output else settings["model_roots"] + settings["data_roots"] + [settings["workspace_root"]]
    if not any(path == Path(r).resolve() or Path(r).resolve() in path.parents for r in roots):
        raise ValueError("outside configured roots")
    if output and path == Path(settings["workspace_root"]).resolve():
        raise ValueError("workspace root is not an output")
    if must_exist and not path.exists():
        raise ValueError("input missing")
    return path


def find_connection(identifier, settings=None):
    settings = settings or load_integrations()
    for connection in settings["integrations"]:
        if connection["id"] == identifier:
            return dict(connection)
    for endpoint in settings["endpoints"]:
        if endpoint["name"] == identifier:
            return {"id": identifier, "type": "inference", "kind": endpoint["kind"], "base_url": endpoint["base_url"]}
    raise ValueError("unknown connection")
