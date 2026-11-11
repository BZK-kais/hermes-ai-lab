"""Explicit provider operations. No account discovery or writes during import/probe."""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from .credentials import CredentialError, resolve_secret, secret_scope
from .process_env import runtime_env

MAX_BYTES = 2 * 1024 * 1024
TIMEOUT = 10
_TYPES = {"runpod", "hf_jobs", "hf_endpoints", "hf_storage", "gcs", "s3"}
_FIELDS = {"id", "type", "base_url", "project", "bucket", "credential_ref", "python", "workspace", "region", "repo_id"}
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_BASES = {"runpod": "https://api.runpod.io/v2", "hf_jobs": "https://huggingface.co",
          "hf_endpoints": "https://huggingface.co", "hf_storage": "https://huggingface.co"}
_KEY_NAMES = {"runpod": "AI_LAB_RUNPOD_API_KEY", "hf_jobs": "AI_LAB_HF_TOKEN",
              "hf_endpoints": "AI_LAB_HF_TOKEN", "hf_storage": "AI_LAB_HF_TOKEN",
              "s3": "AI_LAB_S3_ACCESS_KEY_ID"}
_S3_SECRET = "AI_LAB_S3_SECRET_ACCESS_KEY"
_S3_SESSION = "AI_LAB_S3_SESSION_TOKEN"

# Public discovery schema. Each action accepts exactly the documented fields.
ACTION_SCHEMAS = {
    "runpod": {"probe": {}, "list": {}, "get": {"resource_id": "string"},
               "create": {"config": "object", "budget_usd": "positive number", "timeout_seconds": "1..86400", "idempotency_key": "string"},
               "stop": {"resource_id": "string"}, "delete": {"resource_id": "string"},
               "logs": {"resource_id": "string", "limit": "1..100"}},
    "hf_jobs": {"probe": {}, "list": {}, "get": {"resource_id": "string"},
                "create": {"config": "object", "budget_usd": "positive number", "timeout_seconds": "1..86400", "idempotency_key": "string"},
                "cancel": {"resource_id": "string"}, "logs": {"resource_id": "string", "limit": "1..100"}},
    "hf_endpoints": {"probe": {}, "list": {}, "get": {"resource_id": "string"},
                     "create": {"config": "object", "budget_usd": "positive number", "timeout_seconds": "1..86400", "idempotency_key": "string"},
                     "stop": {"resource_id": "string"}, "delete": {"resource_id": "string"}},
}
for _kind in ("hf_storage", "gcs", "s3"):
    ACTION_SCHEMAS[_kind] = {"probe": {}, "list": {"prefix": "string", "limit": "1..100"},
                             "upload": {"local_path": "workspace-relative file", "key": "relative object path"},
                             "get": {"key": "relative object path"}}
del ACTION_SCHEMAS["gcs"]["upload"]


def failure(code, message):
    return {"ok": False, "error": {"code": code, "message": message}}


def _slug(value):
    return isinstance(value, str) and bool(_SLUG.fullmatch(value)) and ".." not in value


def _relative(value):
    return (isinstance(value, str) and 0 < len(value) <= 1024 and not value.startswith(("/", "-"))
            and "\\" not in value and all(ord(c) >= 32 and ord(c) != 127 for c in value)
            and all(part not in ("", ".", "..") for part in value.split("/")))


def _repo(value):
    return isinstance(value, str) and len(value.split("/")) in (1, 2) and all(_slug(p) for p in value.split("/"))


def validate_connection(c: dict) -> dict:
    if not isinstance(c, dict) or set(c) - _FIELDS or not _slug(c.get("id")) or not isinstance(c.get("type"), str) or c.get("type") not in _TYPES:
        return failure("invalid_config", "Invalid provider connection configuration.")
    kind = c["type"]
    if any(not isinstance(v, str) or len(v) > 2048 or any(ord(x) < 32 for x in v) for k, v in c.items()):
        return failure("invalid_config", "Provider configuration fields must be bounded strings.")
    if c.get("base_url") and c["base_url"].rstrip("/") != _BASES.get(kind):
        return failure("invalid_config", "Provider URL is not an allowed service endpoint.")
    if kind == "gcs":
        if "credential_ref" in c:
            return failure("invalid_config", "Google Cloud Storage uses the plugin's Hermes Google sign-in.")
    else:
        expected_ref = "hermes:" + _KEY_NAMES[kind]
        if "credential_ref" not in c:
            return failure("configuration_required", f"Configure the provider credential reference {expected_ref}.")
        if c["credential_ref"] != expected_ref:
            return failure("invalid_config", f"Use the provider credential reference {expected_ref}.")
    if kind in {"hf_jobs", "hf_endpoints", "gcs"} and not _slug(c.get("project")):
        return failure("configuration_required", "Configure the provider namespace or project.")
    if kind in {"gcs", "s3"} and not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", c.get("bucket", "")):
        return failure("configuration_required", "Configure an existing bucket.")
    if kind == "s3" and not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-[0-9]", c.get("region", "")):
        return failure("configuration_required", "Configure an AWS region.")
    if kind == "hf_storage" and not _repo(c.get("repo_id")):
        return failure("configuration_required", "Configure an existing HF model repository.")
    for field in ("workspace", "python"):
        if field in c and not Path(c[field]).is_absolute():
            return failure("invalid_config", "Workspace and Python paths must be absolute.")
    return {"ok": True, "id": c["id"], "type": kind, "actions": ACTION_SCHEMAS[kind], "state": "configured"}


class _AdapterError(Exception):
    def __init__(self, code):
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_request(method, url, token, body=None, *, stream=False, limit=100):
    """Runpod-only authenticated transport. Never follows redirects or retries."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "api.runpod.io" or not parsed.path.startswith("/v2/"):
        raise _AdapterError("invalid_config")
    request = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + token, "Accept": "text/event-stream" if stream else "application/json", "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            if stream:
                # SSE is open-ended: consume only bounded events; return metadata, never raw log text.
                entries, consumed = [], 0
                deadline = time.monotonic() + TIMEOUT
                for _ in range(limit * 8):
                    if time.monotonic() >= deadline:
                        break
                    line = response.readline(8193)
                    consumed += len(line)
                    if consumed > MAX_BYTES or len(line) > 8192:
                        break
                    if not line:
                        break
                    if line.startswith(b"data:"):
                        entry = json.loads(line[5:])
                        entries.append({"source": entry.get("source") if entry.get("source") in {"container", "system"} else "unknown",
                                        "message_withheld": True})
                        if len(entries) >= limit:
                            break
                return {"entries": entries, "content_withheld": True, "is_truncated": len(entries) >= limit}
            raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise _AdapterError("response_too_large")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        status = exc.code
        with contextlib.suppress(Exception):
            exc.close()
        raise _AdapterError("redirect_refused" if 300 <= status < 400 else
                            {401: "auth_required", 403: "forbidden", 404: "not_found", 429: "rate_limited"}.get(status, "provider_error")) from None


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _selected(obj, fields):
    result = {}
    for name in fields:
        value = _value(obj, name)
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, str):
            # Metadata is bounded; URL-looking strings are never returned.
            result[name] = value if len(value) <= 200 and "://" not in value and "?" not in value else "[withheld]"
        elif value is None or isinstance(value, (bool, int)):
            result[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = value
    return result


def _safe_output(value, secrets=()):
    """Defense in depth: resolved secrets cannot survive even in selected metadata."""
    if isinstance(value, dict):
        return {k: _safe_output(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_output(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    return value


def _artifact(c, relative):
    if not c.get("workspace") or not _relative(relative):
        raise _AdapterError("invalid_input")
    root = Path(c["workspace"]).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 50 * 1024**3:
        raise _AdapterError("invalid_input")
    return str(path)


def _intent(c, params):
    """Exclusive intent file prevents repeat POSTs, including after ambiguous failure."""
    key = params.get("idempotency_key")
    budget, duration = params.get("budget_usd"), params.get("timeout_seconds")
    if (not _slug(key) or not isinstance(budget, (int, float)) or isinstance(budget, bool)
            or not math.isfinite(budget) or budget <= 0 or budget > 100000
            or not isinstance(duration, int) or isinstance(duration, bool) or not 1 <= duration <= 86400
            or not c.get("workspace")):
        raise _AdapterError("budget_required")
    root = Path(c["workspace"]).resolve(strict=True)
    journal = root / ".ai-lab" / "intents"
    if not journal.resolve().is_relative_to(root):
        raise _AdapterError("invalid_config")
    journal.mkdir(parents=True, exist_ok=True)
    if not journal.resolve().is_relative_to(root):
        raise _AdapterError("invalid_config")
    digest = hashlib.sha256((c["id"] + ":" + key).encode()).hexdigest()
    path = journal / (digest + ".json")
    try:
        with path.open("x") as handle:
            json.dump({"connection_id": c["id"], "state": "intent_recorded", "budget_usd": budget,
                       "timeout_seconds": duration, "budget_enforced": False}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise _AdapterError("reconciliation_required") from None
    return path


def _finish_intent(path, resource_id):
    # Only selected identifier/state, never config, env, credentials or provider body.
    with path.open("w") as handle:
        json.dump({"state": "submitted", "resource_id": resource_id}, handle)


def _create_config(kind, config, duration):
    if not isinstance(config, dict):
        raise _AdapterError("invalid_input")
    allowed = {
        "runpod": {"name", "image", "gpu", "disk", "dataCenterIds", "templateId"},
        "hf_jobs": {"image", "command", "flavor"},
        "hf_endpoints": {"name", "repository", "framework", "accelerator", "instance_size", "instance_type", "region", "vendor", "revision", "task", "min_replica", "max_replica", "scale_to_zero_timeout"},
    }[kind]
    if set(config) - allowed or len(json.dumps(config)) > 16384:
        raise _AdapterError("invalid_input")
    if kind == "runpod":
        if not _slug(config.get("name")) or not isinstance(config.get("gpu"), dict):
            raise _AdapterError("invalid_input")
        gpu = config["gpu"]
        if set(gpu) != {"id", "count"} or not isinstance(gpu.get("id"), str) or not isinstance(gpu.get("count"), int) or isinstance(gpu.get("count"), bool) or not 1 <= gpu["count"] <= 8:
            raise _AdapterError("invalid_input")
        if not isinstance(config.get("image", config.get("templateId")), str):
            raise _AdapterError("invalid_input")
        for key in ("image", "templateId"):
            if key in config and (not config[key] or len(config[key]) > 256 or any(ord(x) < 33 for x in config[key])):
                raise _AdapterError("invalid_input")
    elif kind == "hf_jobs":
        if (not isinstance(config.get("image"), str) or not isinstance(config.get("flavor"), str)
                or not isinstance(config.get("command"), list) or not 1 <= len(config["command"]) <= 32
                or any(not isinstance(x, str) or len(x) > 2048 or "\x00" in x for x in config["command"])):
            raise _AdapterError("invalid_input")
        # Cloud command is an explicitly supplied container configuration, never local shell.
        config = config | {"timeout": duration, "attempts": 1}
    else:
        required = {"name", "repository", "framework", "accelerator", "instance_size", "instance_type", "region", "vendor"}
        if not required <= set(config) or any(not isinstance(config[k], str) or not config[k] or len(config[k]) > 256 for k in required):
            raise _AdapterError("invalid_input")
        if not _slug(config["name"]) or not _repo(config["repository"]):
            raise _AdapterError("invalid_input")
        config = config | {"type": "authenticated"}
    return config


def _runpod(c, action, p):
    token = resolve_secret(c["credential_ref"])
    base = _BASES["runpod"] + "/pods"
    if action in {"probe", "list"}:
        data = _http_request("GET", base + "?limit=100", token)
        if not isinstance(data, dict) or not isinstance(data.get("pods"), list):
            raise _AdapterError("invalid_response")
        result = {"ok": True, "resources": [_selected(x, ("id", "name", "status", "cost")) for x in data["pods"][:100]],
                  "is_truncated": bool(_value(data.get("pagination", {}), "hasNextPage", False))}
    elif action == "create":
        payload = _create_config("runpod", p.get("config"), p.get("timeout_seconds"))
        intent = _intent(c, p)
        data = _http_request("POST", base, token, payload)
        if not isinstance(data, dict) or not _slug(data.get("id")):
            raise _AdapterError("invalid_response")
        _finish_intent(intent, _safe_output(data["id"], (token,)))
        result = {"ok": True, "resource": _selected(data, ("id", "name", "status", "cost")),
                  "budget": {"enforced": False, "requested_usd": p["budget_usd"]},
                  "timeout": {"enforced": False, "requested_seconds": p["timeout_seconds"]}}
    else:
        resource = p.get("resource_id")
        if not _slug(resource):
            raise _AdapterError("invalid_input")
        url = base + "/" + urllib.parse.quote(resource, safe="")
        if action == "logs":
            data = _http_request("GET", url + "/logs?tail=" + str(p.get("limit", 100)), token, stream=True, limit=p.get("limit", 100))
            result = {"ok": True, "logs": data}
        else:
            method = "GET" if action == "get" else "DELETE" if action == "delete" else "POST"
            data = _http_request(method, url + ("/action" if action == "stop" else ""), token,
                                 {"action": "stop"} if action == "stop" else None)
            result = {"ok": True, "resource": _selected(data, ("id", "name", "status", "cost")),
                      "action": action, "storage_deleted": False if action == "stop" else None}
    return _safe_output(result, (token,))


def _guard_hf_http(kind, create=False, transfer=False):
    """Constrain the isolated SDK client and remove bearer headers from transfer hosts."""
    import httpx
    import huggingface_hub
    if not hasattr(huggingface_hub, "set_client_factory"):
        raise _AdapterError("dependency_version")
    writes = 0
    def guard(request):
        nonlocal writes
        host = request.url.host
        allowed = host in {"huggingface.co", "api.endpoints.huggingface.cloud"}
        if kind == "hf_storage":
            allowed = allowed or host.endswith(".hf.co") or host.endswith(".huggingface.co")
        if request.url.scheme != "https" or request.url.port not in (None, 443) or request.url.userinfo or not allowed:
            raise _AdapterError("destination_refused")
        if kind == "hf_storage" and host != "huggingface.co":
            request.headers.pop("Authorization", None)
        if create and request.method == "POST":
            writes += 1
            if writes > 1:
                raise _AdapterError("reconciliation_required")
    def reject_redirect(response):
        if not transfer and 300 <= response.status_code < 400:
            raise _AdapterError("redirect_refused")
    huggingface_hub.set_client_factory(lambda: httpx.Client(timeout=TIMEOUT, follow_redirects=False, trust_env=False,
                                                             event_hooks={"request": [guard], "response": [reject_redirect]}))


def _hf_operation(c, action, p):
    from huggingface_hub import HfApi
    kind = c["type"]
    token = resolve_secret(c["credential_ref"])
    _guard_hf_http(kind, action == "create")
    api = HfApi(endpoint="https://huggingface.co", token=token)
    namespace = c.get("project")
    resource = p.get("resource_id")
    if kind == "hf_jobs":
        if action in {"probe", "list"}:
            items = list(itertools.islice(api.list_jobs(namespace=namespace), 101))
            result = {"ok": True, "resources": [_job(x) for x in items[:100]], "is_truncated": len(items) > 100}
        elif action == "create":
            item = api.run_job(namespace=namespace, **p["config"])
            result = {"ok": True, "resource": _job(item)}
        elif action == "get":
            result = {"ok": True, "resource": _job(api.inspect_job(job_id=resource, namespace=namespace))}
        elif action == "cancel":
            api.cancel_job(job_id=resource, namespace=namespace)
            result = {"ok": True, "resource_id": resource, "action": "cancel"}
        else:
            items = list(itertools.islice(api.fetch_job_logs(job_id=resource, namespace=namespace), p.get("limit", 100)))
            result = {"ok": True, "logs": {"events_observed": len(items), "content_withheld": True,
                                            "is_truncated": len(items) == p.get("limit", 100)}}
    elif kind == "hf_endpoints":
        if action in {"probe", "list"}:
            items = list(itertools.islice(api.list_inference_endpoints(namespace=namespace), 101))
            result = {"ok": True, "resources": [_selected(x, ("name", "namespace", "status", "repository", "revision")) for x in items[:100]], "is_truncated": len(items) > 100}
        elif action == "create":
            item = api.create_inference_endpoint(namespace=namespace, **p["config"])
            result = {"ok": True, "resource": _selected(item, ("name", "namespace", "status", "repository", "revision"))}
        elif action == "get":
            result = {"ok": True, "resource": _selected(api.get_inference_endpoint(resource, namespace=namespace), ("name", "namespace", "status", "repository", "revision"))}
        else:
            function = api.pause_inference_endpoint if action == "stop" else api.delete_inference_endpoint
            function(resource, namespace=namespace)
            result = {"ok": True, "resource_id": resource, "action": action}
    else:
        repo = c["repo_id"]
        if action == "upload":
            info = api.upload_file(path_or_fileobj=p["local_path"], path_in_repo=p["key"], repo_id=repo, repo_type="model", commit_message="AI Lab artifact")
            result = {"ok": True, "commit": _selected(info, ("oid",)), "uploaded": True}
        elif action in {"probe", "get"}:
            info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=action == "get")
            result = {"ok": True, "repository": _selected(info, ("id", "sha", "private"))}
            if action == "get":
                matches = [x for x in info.siblings if x.rfilename == p["key"]]
                if not matches:
                    raise _AdapterError("not_found")
                result["object"] = _selected(matches[0], ("rfilename", "size"))
        else:
            entries = api.list_repo_tree(repo_id=repo, path_in_repo=p.get("prefix") or None, recursive=False, repo_type="model")
            items = list(itertools.islice(entries, p.get("limit", 100) + 1))
            result = {"ok": True, "objects": [_selected(x, ("path", "size")) for x in items[:p.get("limit", 100)]], "is_truncated": len(items) > p.get("limit", 100)}
    return _safe_output(result, (token,))


def _job(item):
    result = _selected(item, ("id", "docker_image", "flavor"))
    result["status"] = _selected(_value(item, "status", {}), ("stage",)).get("stage")
    return result


def _gcs_operation(c, action, p):
    from hermes_cli.plugin_provider_requests import ProviderNotSignedIn, credentialed_provider_request

    url = "https://storage.googleapis.com/storage/v1/b/" + urllib.parse.quote(c["bucket"], safe="")
    object_fields = "name,size,generation"
    limit = p.get("limit", 100)
    if action == "list":
        url += "/o?" + urllib.parse.urlencode({"prefix": p.get("prefix", ""), "maxResults": limit,
                                             "fields": "items(" + object_fields + "),nextPageToken"})
    elif action == "get":
        url += "/o/" + urllib.parse.quote(p["key"], safe="") + "?" + urllib.parse.urlencode({"fields": object_fields})
    else:
        url += "?fields=name"
    try:
        response = credentialed_provider_request("google", "GET", url, timeout=TIMEOUT)
    except (ProviderNotSignedIn, PermissionError):
        raise CredentialError("Hermes Google sign-in is unavailable.") from None
    if response.status != 200:
        code = {401: "auth_required", 403: "forbidden", 404: "not_found", 429: "rate_limited"}
        raise _AdapterError("redirect_refused" if 300 <= response.status < 400 else code.get(response.status, "provider_error"))
    if len(response.body) > MAX_BYTES:
        raise _AdapterError("response_too_large")
    data = response.json()
    if not isinstance(data, dict):
        raise _AdapterError("invalid_response")
    if action == "probe":
        if data.get("name") != c["bucket"]:
            raise _AdapterError("invalid_response")
        return {"ok": True, "bucket_verified": True}
    if action == "list":
        items = data.get("items", [])
        if not isinstance(items, list):
            raise _AdapterError("invalid_response")
        return {"ok": True, "objects": [_gcs_object(item) for item in items[:limit]],
                "is_truncated": bool(data.get("nextPageToken")) or len(items) > limit}
    return {"ok": True, "object": _gcs_object(data), "uploaded": False}


def _gcs_object(item):
    if not isinstance(item, dict):
        raise _AdapterError("invalid_response")
    result = _selected(item, ("name", "size", "generation"))
    for name in ("size", "generation"):
        value = result.get(name)
        if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
            result[name] = int(value)
    return result


def _provider_secrets(c):
    name = _KEY_NAMES[c["type"]]
    values = {name: resolve_secret("hermes:" + name)}
    if c["type"] == "s3":
        values[_S3_SECRET] = resolve_secret("hermes:" + _S3_SECRET)
        try:
            values[_S3_SESSION] = resolve_secret("hermes:" + _S3_SESSION)
        except CredentialError:
            pass
    return values


def _s3_client(c):
    import boto3
    from botocore.config import Config
    keys = _provider_secrets(c)
    bucket_name = c["bucket"]
    client = boto3.client("s3", region_name=c["region"], endpoint_url=f"https://s3.{c['region']}.amazonaws.com",
                          aws_access_key_id=keys[_KEY_NAMES["s3"]], aws_secret_access_key=keys[_S3_SECRET],
                          aws_session_token=keys.get(_S3_SESSION),
                          config=Config(connect_timeout=TIMEOUT, read_timeout=TIMEOUT, retries={"total_max_attempts": 1}, proxies={}))
    def guard(request, **kwargs):
        parsed = urllib.parse.urlsplit(request.url)
        regional = f"s3.{c['region']}.amazonaws.com"
        if parsed.scheme != "https" or parsed.netloc not in {regional, f"{bucket_name}.{regional}"}:
            raise _AdapterError("destination_refused")
    client.meta.events.register("before-send.s3", guard)
    return client


def _storage_operation(c, action, p):
    client = _s3_client(c)
    bucket_name = c["bucket"]
    if action == "probe":
        client.head_bucket(Bucket=bucket_name)
        return {"ok": True, "bucket_verified": True}
    if action == "list":
        limit = p.get("limit", 100)
        data = client.list_objects_v2(Bucket=bucket_name, Prefix=p.get("prefix", ""), MaxKeys=limit)
        return {"ok": True, "objects": [_selected(x, ("Key", "Size")) for x in data.get("Contents", [])[:limit]], "is_truncated": bool(data.get("IsTruncated"))}
    if action == "upload":
        # Single PUT bounded by the service's 5 GiB limit. Never supply raw credentials in config.
        if Path(p["local_path"]).stat().st_size > 5 * 1024**3:
            raise _AdapterError("artifact_too_large")
        with open(p["local_path"], "rb") as handle:
            client.put_object(Bucket=bucket_name, Key=p["key"], Body=handle, IfNoneMatch="*")
        return {"ok": True, "uploaded": True}
    data = client.head_object(Bucket=bucket_name, Key=p["key"])
    return {"ok": True, "object": _selected(data, ("ContentLength", "VersionId"))}


def _sdk_worker(payload):
    """Child entry: exceptions/logs never leave this boundary."""
    try:
        c, action, p = payload["connection"], payload["action"], payload["params"]
        values = payload["secrets"]
        with secret_scope(values):
            if action in {"storage.download", "storage.upload"}:
                from .transfer_task import execute_transfer
                result = execute_transfer(action, c, **p)
            elif c["type"].startswith("hf_"):
                result = _hf_operation(c, action, p)
            elif c["type"] == "s3":
                result = _storage_operation(c, action, p)
            else:
                raise _AdapterError("unsupported_transport")
            return _safe_output(result, values.values())
    except ImportError:
        return failure("dependency_missing", "Install the optional provider SDK in the configured Python environment.")
    except CredentialError:
        return failure("credential_unavailable", "The configured credential could not be used.")
    except _AdapterError as exc:
        return failure(exc.code, "The provider operation could not be completed.")
    except Exception:
        return failure("provider_error", "The provider rejected or could not complete this operation.")


_SDK_CODE = '''import json,os,sys
output=os.fdopen(os.dup(sys.stdout.fileno()),"w",encoding="utf-8")
with open(os.devnull,"wb") as sink:
    os.dup2(sink.fileno(),sys.stdout.fileno())
    os.dup2(sink.fileno(),sys.stderr.fileno())
sys.path.insert(0,sys.argv[1])
from ai_lab_core.providers import MAX_BYTES,_sdk_worker
raw=sys.stdin.buffer.read(MAX_BYTES+1)
if len(raw)>MAX_BYTES:
    sys.exit(2)
result=json.dumps(_sdk_worker(json.loads(raw)),allow_nan=False)
if len(result.encode("utf-8"))>MAX_BYTES:
    sys.exit(2)
with output:
    output.write(result)
'''


def _sdk_call(c, action, p, *, values=None, timeout=None):
    # Use an isolated SDK process so client policies do not mutate Hermes/global SDK state.
    if values is None:
        values = _provider_secrets(c)
    payload = json.dumps({"connection": c, "action": action, "params": p, "secrets": values}).encode()
    if len(payload) > MAX_BYTES:
        raise _AdapterError("invalid_input")
    root = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory(prefix="ai-lab-sdk-") as home:
        env = runtime_env({"HF_HUB_OFFLINE": "0"})
        env.update(HOME=home, AWS_EC2_METADATA_DISABLED="true")
        result = subprocess.run([c.get("python", sys.executable), "-I", "-c", _SDK_CODE, root],
                                input=payload, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=timeout or (300 if action == "upload" else 45), cwd=root,
                                env=env, check=False)
    if result.returncode or len(result.stdout) > MAX_BYTES:
        raise _AdapterError("provider_error")
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
        raise _AdapterError("invalid_response")
    return _safe_output(data, values.values())


def probe_connection(c: dict) -> dict:
    return provider_action(c, "probe", {})


def provider_action(c: dict, action: str, params: dict) -> dict:
    checked = validate_connection(c)
    if not checked["ok"]:
        return checked
    kind = c["type"]
    if kind == "gcs" and action == "upload":
        return failure("unsupported_transport", "Hermes Google sign-in supports bounded metadata requests in AI Lab.")
    if not isinstance(action, str) or action not in ACTION_SCHEMAS[kind] or not isinstance(params, dict):
        return failure("invalid_input", "Unknown provider action or invalid parameters.")
    schema = ACTION_SCHEMAS[kind][action]
    if set(params) - set(schema):
        return failure("invalid_input", "Unexpected provider action parameters.")
    try:
        p = dict(params)
        limit = p.get("limit", 100)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise _AdapterError("invalid_input")
        if action in {"get", "stop", "delete", "cancel", "logs"} and kind not in {"hf_storage", "gcs", "s3"} and not _slug(p.get("resource_id")):
            raise _AdapterError("invalid_input")
        if kind in {"hf_storage", "gcs", "s3"}:
            if action in {"get", "upload"} and not _relative(p.get("key")):
                raise _AdapterError("invalid_input")
            if p.get("prefix") and not _relative(p["prefix"]):
                raise _AdapterError("invalid_input")
            if action == "upload":
                p["local_path"] = _artifact(c, p.get("local_path"))
        if kind == "runpod":
            result = _runpod(c, action, p)
        elif kind == "gcs":
            result = _gcs_operation(c, action, p)
        else:
            intent = None
            if action == "create":
                config = _create_config(kind, p.get("config"), p.get("timeout_seconds"))
                values = _provider_secrets(c)
                intent = _intent(c, p)
                result = _sdk_call(c, action, {"config": config}, values=values)
            else:
                result = _sdk_call(c, action, p)
            if intent and result.get("ok"):
                resource = result.get("resource", {})
                _finish_intent(intent, resource.get("id", resource.get("name")))
                result["budget"] = {"enforced": False, "requested_usd": params["budget_usd"]}
                result["timeout"] = {"enforced": kind == "hf_jobs", "requested_seconds": params["timeout_seconds"]}
        return result | {"connection_id": c["id"], "type": kind, "actions": ACTION_SCHEMAS[kind]}
    except CredentialError:
        return failure("credential_unavailable", "The configured credential could not be used.")
    except ImportError:
        return failure("dependency_missing", "This provider requires a compatible Hermes version or optional SDK.")
    except _AdapterError as exc:
        return failure(exc.code, "The provider operation could not be completed; inspect selected resource state before retrying a creation.")
    except subprocess.TimeoutExpired:
        return failure("timeout", "Provider request timed out; reconcile resource state before retrying a creation.")
    except Exception:
        return failure("provider_error", "The provider operation failed; no raw provider details are exposed.")
