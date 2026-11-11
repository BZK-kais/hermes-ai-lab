"""Explicit generation against a configured inference connection."""
from __future__ import annotations

import ipaddress
import json
import urllib.parse
import urllib.request
from .endpoint import _NoRedirect, failure


def connection_url(connection, mcp=False):
    url = connection.get("base_url", "")
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 33 for c in url):
        raise ValueError("invalid URL")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("invalid URL")
    host = parsed.hostname
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme == "http" and not loopback:
        raise ValueError("HTTPS required for remote endpoints")
    if not mcp and parsed.path.rstrip("/") not in {"", "/v1"}:
        raise ValueError("invalid endpoint path")
    if parsed.port == 0:
        raise ValueError("invalid port")
    return url.rstrip("/")


def _credential(connection):
    ref = connection.get("credential_ref")
    if not ref:
        return None
    from .credentials import resolve_credential
    return resolve_credential(ref)


def infer(connection, model, prompt, max_tokens=128):
    if (not isinstance(model, str) or not 1 <= len(model) <= 256 or not isinstance(prompt, str)
            or not 1 <= len(prompt) <= 100000 or type(max_tokens) is not int or not 1 <= max_tokens <= 2048):
        return failure("invalid_input", "Invalid model, prompt or token limit.")
    try:
        base = connection_url(connection)
        kind = connection.get("kind", "openai")
        if kind not in {"openai", "llama_cpp", "ollama", "mlx"}:
            return failure("invalid_input", "Unsupported inference protocol.")
        token = _credential(connection)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if kind == "ollama":
            route = base.removesuffix("/v1") + "/api/chat"
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"num_predict": max_tokens}}
        else:
            route = base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False}
        request = urllib.request.Request(route, data=json.dumps(body).encode(), headers=headers, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=120) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return failure("response_too_large", "Generation response exceeded the limit.")
        payload = json.loads(raw)
        message = payload.get("message") if kind == "ollama" else payload["choices"][0]["message"]
        content = message.get("content")
        if not isinstance(content, str):
            return failure("invalid_response", "Endpoint did not return textual content.")
        if token:
            content = content.replace(token, "[REDACTED]")
        usage = payload.get("usage", {})
        usage = {k: v for k, v in usage.items() if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(v) is int} if isinstance(usage, dict) else {}
        return {"ok": True, "connection_id": connection["id"], "model": model, "content": content,
                "usage": usage, "generation_verified": True, "artifact_identity_verified": False}
    except ImportError:
        return failure("dependency_missing", "The credential adapter is unavailable.")
    except Exception:
        return failure("generation_failed", "The configured endpoint could not complete generation; inspect its connection status.")


def probe_inference(connection):
    try:
        base = connection_url(connection)
        kind = connection.get("kind", "openai")
        token = _credential(connection)
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        route = base.removesuffix("/v1") + "/api/tags" if kind == "ollama" else base + ("" if base.endswith("/v1") else "/v1") + "/models"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(urllib.request.Request(route, headers=headers), timeout=5) as response:
            raw = response.read(262145)
        if len(raw) > 262144:
            raise ValueError
        payload = json.loads(raw)
        entries = payload.get("models" if kind == "ollama" else "data")
        if not isinstance(entries, list):
            raise ValueError
        models = []
        for entry in entries[:100]:
            value = entry.get("name" if kind == "ollama" else "id") if isinstance(entry, dict) else None
            if isinstance(value, str):
                models.append(value[:256].replace(token, "[REDACTED]") if token else value[:256])
        return {"ok": True, "connection_id": connection["id"], "state": "models_listed" if models else "no_models", "models": models, "generation_verified": False}
    except Exception:
        return failure("connection_unverified", "The configured inference endpoint could not be verified.")
