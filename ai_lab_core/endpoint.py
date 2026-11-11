"""Bounded read-only probes of local inference APIs."""

from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request

MAX_BYTES = 262144


def failure(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def normalize_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 33 for c in value):
        raise ValueError("invalid URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("invalid URL")
    if parsed.query or parsed.fragment or parsed.path.rstrip("/") not in {"", "/v1"}:
        raise ValueError("invalid URL")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    if not host or not ipaddress.ip_address(host).is_loopback:
        raise ValueError("local endpoints only")
    port = parsed.port
    if port == 0:
        raise ValueError("invalid port")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", ""))


def _get_json(url: str) -> dict:
    # Ignore proxies and redirects: a local probe must remain on the selected host.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Hermes-AI-Lab/0.1"})
    try:
        with opener.open(request, timeout=3) as response:
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return failure("response_too_large", "The endpoint response exceeded the probe limit.")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return failure("invalid_response", "The endpoint did not return a JSON object.")
        return {"ok": True, "payload": payload}
    except urllib.error.HTTPError as exc:
        codes = {401: "auth_required", 403: "auth_required", 404: "not_found", 503: "not_ready"}
        code = "redirect_refused" if 300 <= exc.code < 400 else codes.get(exc.code, "http_error")
        return failure(code, "The endpoint rejected this read-only probe.")
    except (ValueError, UnicodeError):
        return failure("invalid_response", "The endpoint did not return valid JSON.")
    except (OSError, urllib.error.URLError, RuntimeError):
        return failure("unreachable", "The local endpoint could not be reached.")


def _text(value: object, limit: int = 200) -> str | None:
    if not isinstance(value, str):
        return None
    return "".join(c for c in value if c.isprintable())[:limit]


def probe(base_url: str, kind: str = "llama_cpp") -> dict:
    if kind not in {"llama_cpp", "ollama", "openai"}:
        return failure("invalid_input", "Unknown inference endpoint kind.")
    try:
        base = normalize_url(base_url)
    except (ValueError, TypeError):
        return failure("invalid_input", "Use a loopback HTTP(S) URL without credentials, query or custom path.")
    route = "/api/version" if kind == "ollama" else "/health" if kind == "llama_cpp" else "/v1/models"
    health = _get_json(base + route)
    result = {"ok": False, "kind": kind, "base_url": base, "reachable": health["ok"], "generation_verified": False}
    if not health["ok"]:
        return result | {"error": health["error"]}
    if kind == "ollama":
        version = _text(health["payload"].get("version"))
        if not version:
            return result | failure("invalid_response", "This does not appear to be an Ollama endpoint.")
        result["version"] = version
    elif kind == "llama_cpp" and health["payload"].get("status") != "ok":
        return result | failure("not_ready", "The llama.cpp endpoint is not ready.")
    models = health if kind == "openai" else _get_json(base + ("/api/tags" if kind == "ollama" else "/v1/models"))
    if not models["ok"]:
        return result | {"error": models["error"]}
    entries = models["payload"].get("models" if kind == "ollama" else "data")
    if not isinstance(entries, list):
        return result | failure("invalid_response", "The endpoint model listing is invalid.")
    selected = []
    for model in entries[:100]:
        if not isinstance(model, dict):
            continue
        identifier = _text(model.get("name") if kind == "ollama" else model.get("id"))
        if identifier:
            selected.append({"id": identifier})
    return result | {"ok": True, "state": "models_listed" if selected else "no_models", "models": selected,
                     "models_truncated": len(entries) > 100, "verification": "read_only_api_probe"}
