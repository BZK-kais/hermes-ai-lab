"""Resolve credentials inside Hermes or an isolated SDK child."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

MAX_SECRET_BYTES = 16384
SECRET_NAMES = frozenset({
    "AI_LAB_RUNPOD_API_KEY",
    "AI_LAB_HF_TOKEN",
    "AI_LAB_INFERENCE_API_KEY",
    "AI_LAB_MCP_API_KEY",
    "AI_LAB_HTTP_API_KEY",
    "AI_LAB_S3_ACCESS_KEY_ID",
    "AI_LAB_S3_SECRET_ACCESS_KEY",
    "AI_LAB_S3_SESSION_TOKEN",
})
_CHILD_SECRETS = ContextVar("ai_lab_child_secrets", default=None)


class CredentialError(Exception):
    """Safe static error for an invalid or unavailable credential."""


def validate_reference(reference: object, *, allow_native: bool = False) -> bool:
    return (
        isinstance(reference, str)
        and len(reference) <= 64
        and reference.startswith("hermes:")
        and reference[7:] in SECRET_NAMES
    )


def _validate_secret(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_SECRET_BYTES or not value.isprintable():
        raise CredentialError("Credential unavailable.")
    value = value.strip()
    try:
        if value and len(value.encode("utf-8")) <= MAX_SECRET_BYTES:
            return value
    except UnicodeError:
        pass
    raise CredentialError("Credential unavailable.")


@contextmanager
def secret_scope(values: Mapping[str, str]):
    """Scope a one-shot IPC payload inside a trusted child process."""
    if not isinstance(values, Mapping) or not values.keys() <= SECRET_NAMES:
        raise CredentialError("Credential reference is invalid.")
    scoped = {name: _validate_secret(value) for name, value in values.items()}
    token = _CHILD_SECRETS.set(scoped)
    try:
        yield
    finally:
        _CHILD_SECRETS.reset(token)
        scoped.clear()


def resolve_secret(reference: str) -> str:
    """Resolve an allowlisted reference using the current Hermes profile."""
    if not validate_reference(reference):
        raise CredentialError("Credential reference is invalid.")
    name = reference[7:]
    scoped = _CHILD_SECRETS.get()
    if scoped is not None:
        return _validate_secret(scoped.get(name))
    try:
        from agent.secret_scope import get_secret

        return _validate_secret(get_secret(name))
    except Exception:
        raise CredentialError("Credential unavailable.") from None


def resolve_credential(reference: str) -> str:
    """Compatibility name for trusted adapter runtimes."""
    return resolve_secret(reference)
