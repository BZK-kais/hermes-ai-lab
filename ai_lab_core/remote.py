"""SSH connection validation; execution awaits a supported Hermes auth route."""
from __future__ import annotations

from pathlib import Path
import re

from .providers import failure, _relative, _slug

ACTION_SCHEMAS = {
    "probe": {},
    "exec": {"plan_path": "workspace-relative JSON file", "run_id": "unique identifier"},
    "status": {"run_id": "identifier"},
    "cancel": {"run_id": "identifier"},
}
_FIELDS = {"id", "type", "host", "user", "port", "python", "workspace", "remote_workspace", "script"}
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")
_REMOTE_PATH = re.compile(r"/[A-Za-z0-9_./-]{1,512}\Z")
MAX_BYTES = 65536


def validate_connection(c: dict) -> dict:
    if not isinstance(c, dict) or set(c) - _FIELDS or c.get("type") != "ssh" or not _slug(c.get("id")):
        return failure("invalid_config", "Invalid SSH connection configuration.")
    if not isinstance(c.get("host"), str) or not _HOST.fullmatch(c["host"]) or ".." in c["host"]:
        return failure("invalid_config", "Configure a safe SSH hostname or IPv4 address.")
    if "user" in c and (not isinstance(c["user"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", c["user"])):
        return failure("invalid_config", "Invalid SSH user.")
    port = c.get("port", 22)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return failure("invalid_config", "Invalid SSH port.")
    python = c.get("python", "python3")
    if not isinstance(python, str) or not re.fullmatch(r"(?:/[A-Za-z0-9_./-]+|python3(?:\.[0-9]+)?)", python) or ".." in python:
        return failure("invalid_config", "Invalid remote Python executable.")
    work = c.get("remote_workspace", "")
    if work and (not isinstance(work, str) or not _REMOTE_PATH.fullmatch(work) or any(x in {"", ".", ".."} for x in work[1:].split("/"))):
        return failure("invalid_config", "Invalid remote workspace.")
    if "workspace" in c and (not isinstance(c["workspace"], str) or not Path(c["workspace"]).is_absolute()):
        return failure("invalid_config", "Configure an absolute local workspace.")
    if "script" in c and (not _relative(c["script"]) or not c["script"].endswith(".py")):
        return failure("invalid_config", "Configure a workspace-relative trusted Python script.")
    return {"ok": True, "id": c["id"], "type": "ssh", "state": "auth_route_unavailable", "execution_available": False, "actions": ACTION_SCHEMAS}


def probe_connection(c):
    return remote_action(c, 'probe', {})


def remote_action(c, action, params):
    checked = validate_connection(c)
    if not checked['ok']:
        return checked
    if not isinstance(action, str) or action not in ACTION_SCHEMAS or not isinstance(params, dict) or set(params) != set(ACTION_SCHEMAS[action]):
        return failure('invalid_input', 'Invalid remote action or parameters.')
    return failure('auth_route_unavailable', 'SSH execution requires a supported Hermes credential route; native SSH credentials are not accessed.')


provider_action = remote_action
probe_remote = probe_connection
validate_remote = validate_connection
