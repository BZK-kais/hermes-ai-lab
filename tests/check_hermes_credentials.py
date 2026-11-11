"""Exercise an AI Lab source checkout with the real Hermes registry and mocked wire I/O."""

from contextlib import ExitStack, contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx
from agent.secret_scope import (
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from hermes_cli import plugin_oauth
from hermes_cli.plugins import PluginManager
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.registry import registry

RUNPOD_KEY = "AI_LAB_RUNPOD_API_KEY"
HF_KEY = "AI_LAB_HF_TOKEN"
HF_FIXTURES = ("fixture-hf-profile-a", "fixture-hf-profile-b")
AMBIENT = "fixture-ambient-must-not-be-used"
A_KEY = "fixture-runpod-profile-a"
B_KEY = "fixture-runpod-profile-b"
GOOGLE = "fixture-google-oauth"
TOOL = "ai_lab_provider_action"


@contextmanager
def scope(home, secrets):
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(secrets, profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def load(source, home, bucket):
    installed = home / "plugins" / "ai-lab"
    installed.mkdir(parents=True)
    for filename in ("__init__.py", "plugin.yaml"):
        shutil.copy2(source / filename, installed / filename)
    for directory in ("ai_lab_core", "skills"):
        shutil.copytree(
            source / directory,
            installed / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    data = home / "ai-lab"
    data.mkdir()
    (data / "integrations.json").write_text(json.dumps({"integrations": [
        {"id": "runpod-test", "type": "runpod", "credential_ref": "hermes:" + RUNPOD_KEY},
        {"id": "hf-test", "type": "hf_jobs", "project": "fixture-project",
         "credential_ref": "hermes:" + HF_KEY},
        {"id": "gcs-test", "type": "gcs", "project": "fixture-project", "bucket": bucket},
    ]}), encoding="utf-8")
    with scope(home, {}):
        manager = PluginManager()
        manifests = manager._scan_directory(installed.parent, source="user")
        assert len(manifests) == 1
        manifest = manifests[0]
        manager._load_plugin(manifest)
        plugin = manager._plugins[manifest.key or manifest.name]
        assert plugin.enabled and not plugin.error, "native registration failed"
        assert TOOL in plugin.tools_registered
    return manager, installed


def dispatch(manager, connection, action):
    result = registry.dispatch(TOOL, {
        "connection_id": connection,
        "action": action,
        "parameters": {},
    }, scope=manager.scope_key)
    serialized = result if isinstance(result, str) else json.dumps(result)
    for fixture in (A_KEY, B_KEY, AMBIENT, GOOGLE, *HF_FIXTURES):
        assert fixture not in serialized, "synthetic secret appeared in tool output"
    return json.loads(serialized)


def deny_network(*args, **kwargs):
    raise AssertionError("Unexpected socket I/O")


def main(source):
    urllib_calls = []
    oauth_calls = []
    hf_calls = []
    caller_paths = []
    client_type = httpx.Client

    class MockOpener:
        def open(self, request, *, timeout):
            assert urlsplit(request.full_url).netloc == "api.runpod.io"
            assert request.get_method() == "GET" and timeout > 0
            auth = request.get_header("Authorization")
            urllib_calls.append((get_hermes_home(), auth))
            return io.BytesIO(json.dumps({"pods": [{"name": auth}]}).encode())

    def fake_child(argv, *, input, stdout, stderr, timeout, cwd, env, check):
        payload = json.loads(input)
        assert payload["connection"]["type"] == "hf_jobs"
        assert payload["action"] == "list"
        assert payload["secrets"].keys() == {HF_KEY}
        token = payload["secrets"][HF_KEY]
        assert token in HF_FIXTURES
        assert HF_KEY not in env and RUNPOD_KEY not in env
        assert all(token not in value for value in [*argv, *env.values()])
        assert argv[1:3] == ["-I", "-c"]
        assert stdout == subprocess.PIPE and stderr == subprocess.DEVNULL and not check
        hf_calls.append((get_hermes_home(), token))
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "ok": True, "resources": [{"name": token}],
        }).encode())

    def respond(request):
        assert request.url.scheme == "https" and request.url.host == "storage.googleapis.com"
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer " + GOOGLE
        oauth_calls.append((get_hermes_home(), request.url.path))
        return httpx.Response(200, json={"name": request.url.path.rsplit("/", 1)[1]})

    def mocked_client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    def fake_bearer(plugin_dir, client):
        assert client.id == "google"
        assert client.hosts == ("storage.googleapis.com",)
        caller_paths.append(plugin_dir)
        return GOOGLE

    with tempfile.TemporaryDirectory(prefix="ai-lab-native-credentials-") as root, ExitStack() as stack:
        base = Path(root).resolve()
        a, b = base / "profile-a", base / "profile-b"
        stack.enter_context(patch.dict(os.environ, {RUNPOD_KEY: AMBIENT, HF_KEY: AMBIENT}))
        stack.enter_context(patch.object(socket, "create_connection", deny_network))
        stack.enter_context(patch.object(socket.socket, "connect", deny_network))
        stack.enter_context(patch.object(socket.socket, "connect_ex", deny_network))
        stack.enter_context(patch("urllib.request.build_opener", return_value=MockOpener()))
        stack.enter_context(patch.object(httpx, "Client", side_effect=mocked_client))
        stack.enter_context(patch.object(plugin_oauth, "_bearer", side_effect=fake_bearer))
        multiplex_token = set_multiplex_context(True)
        stack.callback(reset_multiplex_context, multiplex_token)

        manager_a, installed_a = load(source, a, "fixture-bucket-a")
        stack.callback(manager_a.unload)
        manager_b, installed_b = load(source, b, "fixture-bucket-b")
        stack.callback(manager_b.unload)

        with scope(a, {RUNPOD_KEY: A_KEY}):
            assert dispatch(manager_a, "runpod-test", "list")["ok"]
            assert dispatch(manager_a, "gcs-test", "probe")["bucket_verified"]
        with scope(b, {RUNPOD_KEY: B_KEY}):
            assert dispatch(manager_b, "runpod-test", "list")["ok"]
            assert dispatch(manager_b, "gcs-test", "probe")["bucket_verified"]
        with scope(b, {}):
            missing = dispatch(manager_b, "runpod-test", "list")
            assert missing.get("ok") is False
            assert missing["error"]["code"] == "credential_unavailable"
        with scope(a, {RUNPOD_KEY: A_KEY}):
            assert dispatch(manager_a, "runpod-test", "list")["ok"]

        with patch.object(subprocess, "run", side_effect=fake_child):
            with scope(a, {HF_KEY: HF_FIXTURES[0]}):
                assert dispatch(manager_a, "hf-test", "list")["ok"]
            with scope(b, {HF_KEY: HF_FIXTURES[1]}):
                assert dispatch(manager_b, "hf-test", "list")["ok"]
            with scope(b, {}):
                missing = dispatch(manager_b, "hf-test", "list")
                assert missing.get("ok") is False
                assert missing["error"]["code"] == "credential_unavailable"

        assert hf_calls == [(a, HF_FIXTURES[0]), (b, HF_FIXTURES[1])]
        assert urllib_calls == [(a, "Bearer " + A_KEY), (b, "Bearer " + B_KEY), (a, "Bearer " + A_KEY)]
        assert caller_paths == [installed_a, installed_b]
        assert oauth_calls == [(a, "/storage/v1/b/fixture-bucket-a"), (b, "/storage/v1/b/fixture-bucket-b")]

    print("PASS: real Hermes plugin discovery/registry, two profile-specific Runpod headers,")
    print("      missing credential blocked before HTTP despite ambient key,")
    print("      Hugging Face credentials handed to the SDK child over stdin only,")
    print("      and real Hermes OAuth caller + origin checks on both installed plugin copies.")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
