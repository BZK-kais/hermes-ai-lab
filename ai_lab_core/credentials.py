"""Runtime-only secret resolution. Never include resolved values in tool results."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

MAX_SECRET_BYTES = 16384
_ENV = re.compile(r"env:[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_GCP = re.compile(r"gcp:projects/[A-Za-z0-9_.:-]+/secrets/[A-Za-z0-9_-]+/versions/(?:latest|[0-9]+)\Z")


class CredentialError(Exception):
    """Safe static error; never wraps the underlying exception."""


def validate_reference(reference: object, *, allow_native: bool = True) -> bool:
    if reference == "native":
        return allow_native
    if not isinstance(reference, str) or len(reference) > 1024:
        return False
    if _ENV.fullmatch(reference) or _GCP.fullmatch(reference):
        return True
    if reference.startswith("file:"):
        path = reference[5:]
        return bool(path) and Path(path).is_absolute() and not any(ord(c) < 32 for c in path)
    return False


def _validate_secret(value: object) -> str:
    if not isinstance(value, str):
        raise CredentialError("Credential unavailable.")
    value = value.strip()
    if not value or len(value.encode()) > MAX_SECRET_BYTES or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise CredentialError("Credential unavailable.")
    return value


def resolve_secret(reference: str) -> str:
    """Internal API. Call only inside an adapter runtime, never expose as a tool."""
    if not validate_reference(reference, allow_native=False):
        raise CredentialError("Credential reference is invalid.")
    try:
        if reference.startswith("env:"):
            return _validate_secret(os.environ.get(reference[4:]))
        if reference.startswith("file:"):
            # Explicit file only: never enumerate a directory or credential store.
            with Path(reference[5:]).open("rb") as handle:
                raw = handle.read(MAX_SECRET_BYTES + 1)
            if len(raw) > MAX_SECRET_BYTES:
                raise CredentialError("Credential unavailable.")
            return _validate_secret(raw.decode("utf-8"))
        resource = reference[4:]
        project, secret, version = resource.split("/")[1::2]
        result = subprocess.run(
            ["gcloud", "secrets", "versions", "access", version, "--secret", secret,
             "--project", project, "--quiet"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=15, check=False,
        )
        if result.returncode or len(result.stdout) > MAX_SECRET_BYTES:
            raise CredentialError("Credential unavailable.")
        return _validate_secret(result.stdout.decode("utf-8"))
    except CredentialError:
        raise
    except Exception:
        raise CredentialError("Credential unavailable.") from None


def resolve_credential(reference: str) -> str:
    """Compatibility name for trusted adapter runtimes; not a public tool."""
    return resolve_secret(reference)
