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

from .credentials import CredentialError, resolve_secret, validate_reference

MAX_BYTES = 2 * 1024 * 1024
TIMEOUT = 10
_TYPES = {"runpod", "hf_jobs", "hf_endpoints", "hf_storage", "gcs", "s3"}
_FIELDS = {"id", "type", "base_url", "project", "bucket", "credential_ref", "python", "workspace", "region", "repo_id"}
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_BASES = {"runpod": "https://api.runpod.io/v2", "hf_jobs": "https://huggingface.co",
          "hf_endpoints": "https://huggingface.co", "hf_storage": "https://huggingface.co"}

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
    if "credential_ref" in c and not validate_reference(c["credential_ref"], allow_native=kind != "runpod"):
        return failure("invalid_config", "Invalid credential reference.")
    if kind in {"gcs", "s3"} and c.get("credential_ref", "native") != "native":
        return failure("invalid_config", "Object storage uses native SDK credentials in this adapter.")
    if kind.startswith("hf_") and not c.get("credential_ref"):
        return failure("configuration_required", "Configure an explicit credential reference or native HF authentication.")
    if kind == "runpod" and not c.get("credential_ref"):
        return failure("configuration_required", "Configure a Runpod credential reference.")
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
            result[name] = value[:200] if "://" not in value and "?" not in value else "[withheld]"
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


def _guard_hf_http(kind, create=False):
    """SDK v1+ clients confined to this child process, without redirects/proxy/retries."""
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
        if create and request.method == "POST":
            writes += 1
            if writes > 1:
                raise _AdapterError("reconciliation_required")
    def reject_redirect(response):
        if 300 <= response.status_code < 400:
            raise _AdapterError("redirect_refused")
    huggingface_hub.set_client_factory(lambda: httpx.Client(timeout=TIMEOUT, follow_redirects=False, trust_env=False,
                                                             event_hooks={"request": [guard], "response": [reject_redirect]}))


def _hf_operation(c, action, p):
    from huggingface_hub import HfApi, get_token
    kind = c["type"]
    token = get_token() if c.get("credential_ref") == "native" else resolve_secret(c["credential_ref"])
    if not token:
        raise CredentialError("Credential unavailable.")
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


def _storage_operation(c, action, p):
    kind = c["type"]
    bucket_name = c["bucket"]
    if kind == "gcs":
        from google.cloud import storage
        client = storage.Client(project=c["project"])
        # Restrict the authenticated storage session; SDK-native credential refresh remains SDK-managed.
        request = client._http.request
        def bounded(method, url, **kwargs):
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com":
                raise _AdapterError("destination_refused")
            kwargs.update(timeout=TIMEOUT, allow_redirects=False)
            return request(method, url, **kwargs)
        client._http.request = bounded
        bucket = client.bucket(bucket_name)
        if action == "probe":
            bucket.reload(timeout=TIMEOUT, retry=None)
            return {"ok": True, "bucket_verified": True}
        if action == "list":
            items = list(client.list_blobs(bucket_name, prefix=p.get("prefix", ""), max_results=p.get("limit", 100) + 1, timeout=TIMEOUT, retry=None))
            return {"ok": True, "objects": [_selected(x, ("name", "size", "generation")) for x in items[:p.get("limit", 100)]], "is_truncated": len(items) > p.get("limit", 100)}
        blob = bucket.blob(p["key"])
        if action == "upload":
            blob.upload_from_filename(p["local_path"], timeout=TIMEOUT, retry=None, if_generation_match=0)
        else:
            blob.reload(timeout=TIMEOUT, retry=None)
        return {"ok": True, "object": _selected(blob, ("name", "size", "generation")), "uploaded": action == "upload"}
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", region_name=c["region"], endpoint_url=f"https://s3.{c['region']}.amazonaws.com",
                          config=Config(connect_timeout=TIMEOUT, read_timeout=TIMEOUT, retries={"total_max_attempts": 1}, proxies={}))
    def guard(request, **kwargs):
        parsed = urllib.parse.urlsplit(request.url)
        if parsed.scheme != "https" or not (parsed.hostname == f"s3.{c['region']}.amazonaws.com" or parsed.hostname == f"{bucket_name}.s3.{c['region']}.amazonaws.com"):
            raise _AdapterError("destination_refused")
    client.meta.events.register("before-send.s3", guard)
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
    c, action, p = payload["connection"], payload["action"], payload["params"]
    try:
        if c["type"].startswith("hf_"):
            return _hf_operation(c, action, p)
        return _storage_operation(c, action, p)
    except ImportError:
        return failure("dependency_missing", "Install the optional provider SDK in the configured Python environment.")
    except CredentialError:
        return failure("credential_unavailable", "The configured credential could not be used.")
    except _AdapterError as exc:
        return failure(exc.code, "The provider operation could not be completed.")
    except Exception:
        return failure("provider_error", "The provider rejected or could not complete this operation.")


_SDK_CODE = '''import contextlib,json,os,sys
sys.path.insert(0,sys.argv[1])
from ai_lab_core.providers import _sdk_worker
payload=json.load(sys.stdin)
with open(os.devnull,"w") as sink, contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
    result=_sdk_worker(payload)
print(json.dumps(result,allow_nan=False))
'''


def _sdk_call(c, action, p):
    # Use an isolated SDK process so client policies do not mutate Hermes/global SDK state.
    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "HF_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "STORAGE_EMULATOR_HOST", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    with tempfile.TemporaryFile() as output:
        result = subprocess.run([c.get("python", sys.executable), "-I", "-c", _SDK_CODE, str(Path(__file__).resolve().parents[1])],
                                input=json.dumps({"connection": c, "action": action, "params": p}).encode(),
                                stdout=output, stderr=subprocess.DEVNULL, timeout=45 if action != "upload" else 300,
                                cwd=str(Path(__file__).resolve().parents[1]), env=env, check=False)
        if result.returncode or output.tell() > MAX_BYTES:
            raise _AdapterError("provider_error")
        output.seek(0)
        data = json.load(output)
    if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
        raise _AdapterError("invalid_response")
    return data


def probe_connection(c: dict) -> dict:
    return provider_action(c, "probe", {})


def provider_action(c: dict, action: str, params: dict) -> dict:
    checked = validate_connection(c)
    if not checked["ok"]:
        return checked
    kind = c["type"]
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
        else:
            intent = None
            if action == "create":
                config = _create_config(kind, p.get("config"), p.get("timeout_seconds"))
                intent = _intent(c, p)
                p = {"config": config}
            result = _sdk_call(c, action, p)
            if intent and result.get("ok"):
                resource = result.get("resource", {})
                _finish_intent(intent, resource.get("id", resource.get("name")))
                result["budget"] = {"enforced": False, "requested_usd": params["budget_usd"]}
                result["timeout"] = {"enforced": kind == "hf_jobs", "requested_seconds": params["timeout_seconds"]}
        return result | {"connection_id": c["id"], "type": kind, "actions": ACTION_SCHEMAS[kind]}
    except CredentialError:
        return failure("credential_unavailable", "The configured credential could not be used.")
    except _AdapterError as exc:
        return failure(exc.code, "The provider operation could not be completed; inspect selected resource state before retrying a creation.")
    except subprocess.TimeoutExpired:
        return failure("timeout", "Provider request timed out; reconcile resource state before retrying a creation.")
    except Exception:
        return failure("provider_error", "The provider operation failed; no raw provider details are exposed.")
