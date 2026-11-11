"""Bounded, anonymous access to public Hugging Face model metadata."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_TIMEOUT_SECONDS = 10
_SLUG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}\Z")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _ResponseTooLarge(Exception):
    pass


def _fetch_metadata(url: str) -> bytes:
    """Transport boundary: no auth, environment proxies, redirects or downloads."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
        data = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(data) > _MAX_RESPONSE_BYTES:
        raise _ResponseTooLarge
    return data


def _failure(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _valid_repo(repo_id) -> bool:
    if not isinstance(repo_id, str):
        return False
    parts = repo_id.split("/")
    return len(parts) in (1, 2) and all(
        _SLUG.fullmatch(part) and not part.endswith((".", "-"))
        and ".." not in part and "--" not in part for part in parts
    )


def _valid_revision(revision) -> bool:
    return (isinstance(revision, str) and 1 <= len(revision) <= 128
            and revision == revision.strip()
            and all(32 < ord(char) < 127 for char in revision)
            and "\\" not in revision
            and all(part not in ("", ".", "..") for part in revision.split("/")))


def _size(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _text(value):
    return value if isinstance(value, str) else None


def _summarize(data: dict, repo_id: str, file_limit: int) -> dict:
    siblings = data.get("siblings", [])
    if not isinstance(siblings, list):
        raise ValueError
    files = []
    for sibling in siblings:
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            raise ValueError
        size = _size(sibling.get("size"))
        if size is None and isinstance(sibling.get("lfs"), dict):
            size = _size(sibling["lfs"].get("size"))
        files.append({"name": sibling["rfilename"], "size": size})
    card = data.get("cardData")
    safe = data.get("safetensors")
    parameters = safe.get("parameters") if isinstance(safe, dict) else None
    parameters = ({key: value for key, value in parameters.items()
                   if isinstance(key, str) and _size(value) is not None}
                  if isinstance(parameters, dict) else {})
    gated = data.get("gated")
    return {
        "ok": True,
        "repo_id": repo_id,
        "sha": _text(data.get("sha")),
        "private": data.get("private") if isinstance(data.get("private"), bool) else None,
        "gated": gated if isinstance(gated, bool) or gated in ("auto", "manual") else None,
        "pipeline_tag": _text(data.get("pipeline_tag")),
        "library_name": _text(data.get("library_name")),
        "license": _text(card.get("license")) if isinstance(card, dict) else None,
        "safetensors": {"parameters": parameters},
        "total_size": sum(item["size"] for item in files) if files and all(
            item["size"] is not None for item in files) else None,
        "files": files[:file_limit],
        "is_truncated": len(files) > file_limit,
    }


def get_model_info(repo_id: str, revision: str = "main", file_limit: int = 30) -> dict:
    """Return selected public metadata; never authenticate or fetch model weights."""
    if (not _valid_repo(repo_id) or not _valid_revision(revision)
            or not isinstance(file_limit, int) or isinstance(file_limit, bool)
            or not 1 <= file_limit <= 100):
        return _failure("invalid_input", "Invalid repository, revision or file limit.")
    url = ("https://huggingface.co/api/models/"
           + urllib.parse.quote(repo_id, safe="/") + "/revision/"
           + urllib.parse.quote(revision, safe="") + "?blobs=true")
    try:
        raw = _fetch_metadata(url)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise _ResponseTooLarge
    except _ResponseTooLarge:
        return _failure("response_too_large", "Hub metadata exceeds the response limit.")
    except urllib.error.HTTPError as exc:
        code, message = {
            401: ("auth_required", "This repository requires authentication."),
            403: ("gated", "Access to this repository is restricted."),
            404: ("not_found", "Repository or revision was not found or is not public."),
            429: ("rate_limited", "Hub request limit reached; try again later."),
        }.get(exc.code, ("http_error", "Hub metadata request was rejected."))
        exc.close()
        return _failure(code, message)
    except Exception:
        return _failure("network_error", "Hub metadata could not be reached.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or "error" in data:
            raise ValueError
        return _summarize(data, repo_id, file_limit)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return _failure("invalid_json", "Hub returned invalid model metadata.")
