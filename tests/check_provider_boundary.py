"""Check native Hermes dispatch with synthetic credentials and simulated providers.

Run with the Hermes runtime Python and its SDK checkout on PYTHONPATH.
The only scratch files live below this checkout's .local/review directory.
"""

import argparse
from contextlib import ExitStack, asynccontextmanager, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit


EXPECTED_TOOLS = {
    'ai_lab_' + name for name in (
        'status', 'list_local_models', 'hf_model_info', 'probe_endpoint', 'connections',
        'connection_probe', 'operations', 'plan', 'run', 'job', 'provider_action',
        'dataset_validate', 'artifact_inspect', 'generate', 'profile', 'experiment',
        'analyze', 'benchmark', 'optimize', 'onboarding', 'mission', 'registry',
    )
}
RUNPOD_KEY = 'AI_LAB_RUNPOD_API_KEY'
HF_KEY = 'AI_LAB_HF_TOKEN'
MCP_KEY = 'AI_LAB_MCP_API_KEY'
SECRETS = {
    RUNPOD_KEY: 'fixture-native-runpod-key',
    HF_KEY: 'fixture-native-hf-key',
    MCP_KEY: 'fixture-native-mcp-key',
}
PROVIDER = 'ai_lab_provider_action'
RUN = 'ai_lab_run'


def stage(source, home):
    installed = home / 'plugins' / 'ai-lab'
    installed.mkdir(parents=True)
    for name in ('__init__.py', 'plugin.yaml'):
        shutil.copy2(source / name, installed / name)
    for name in ('ai_lab_core', 'skills'):
        shutil.copytree(source / name, installed / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (home / 'config.yaml').write_text('plugins:\n  enabled: [ai-lab]\n', encoding='utf-8')
    data = home / 'ai-lab'
    workspace = data / 'workspace'
    workspace.mkdir(parents=True)
    artifact = workspace / 'fixture.bin'
    artifact.write_bytes(b'fixture artifact')
    (data / 'integrations.json').write_text(json.dumps({'integrations': [
        {'id': 'fixture-runpod', 'type': 'runpod', 'workspace': str(workspace),
         'credential_ref': 'hermes:' + RUNPOD_KEY},
        {'id': 'fixture-hf', 'type': 'hf_storage', 'workspace': str(workspace),
         'repo_id': 'fixture-owner/fixture-repo', 'credential_ref': 'hermes:' + HF_KEY},
        {'id': 'fixture-mcp', 'type': 'mcp', 'base_url': 'https://fixture.invalid/mcp',
         'credential_ref': 'hermes:' + MCP_KEY},
    ]}), encoding='utf-8')
    return artifact


def arguments(connection, action, **parameters):
    return {'connection_id': connection, 'action': action, 'parameters': parameters}


def result(raw):
    assert isinstance(raw, str), 'Hermes must return a JSON string'
    assert all(secret not in raw for secret in SECRETS.values()), 'Synthetic credential leaked'
    return json.loads(raw)


def assert_denied(raw, name, action):
    data = result(raw)
    assert data.get('ok') is False, (name, action, 'unexpected success')
    assert isinstance(data.get('error'), dict), (name, action, 'not an AI Lab error')
    assert data['error'].get('code') == 'operator_required', (name, action, data['error'].get('code'))


def check_dispatches(stack, home, artifact):
    import mcp
    from mcp.client import streamable_http
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli.main import _attach_plugin_cli_command
    from hermes_cli.plugins import get_plugin_manager
    from model_tools import handle_function_call
    from tools.registry import registry

    manager = get_plugin_manager()
    assert Path(manager.scope_key) == home
    stack.callback(manager.unload)
    manager.discover_and_load()
    plugin = manager._plugins.get('ai-lab')
    assert plugin and plugin.enabled and not plugin.error, 'Native registration failed'
    assert set(plugin.tools_registered) == EXPECTED_TOOLS
    assert all(registry.get_entry(name, scope=manager.scope_key) for name in EXPECTED_TOOLS)

    calls = []
    credentials = importlib.import_module('.ai_lab_core.credentials', plugin.module.__name__)
    providers = importlib.import_module('.ai_lab_core.providers', plugin.module.__name__)
    resolve_secret = credentials.resolve_secret

    def record_secret(ref):
        calls.append(('credential', ref))
        return resolve_secret(ref)

    class MockOpener:
        def open(self, request, *, timeout):
            assert urlsplit(request.full_url).netloc == 'api.runpod.io'
            assert request.get_header('Authorization') == 'Bearer ' + SECRETS[RUNPOD_KEY]
            assert timeout > 0
            method = request.get_method()
            calls.append(('runpod', method))
            if method == 'GET':
                return io.BytesIO(b'{"pods": [{"id": "fixture-pod"}]}')
            assert method == 'POST' and json.loads(request.data)['name'] == 'fixture-pod'
            return io.BytesIO(b'{"id": "fixture-pod"}')

    def fake_sdk_child(argv, **kwargs):
        if 'importlib.util.find_spec' in argv[3]:
            assert argv[-1] == 'huggingface_hub'
            calls.append(('sdk_dependency', argv[-1]))
            return subprocess.CompletedProcess(argv, 0)
        payload = json.loads(kwargs['input'])
        assert payload['connection']['id'] == 'fixture-hf'
        assert payload['secrets'] == {HF_KEY: SECRETS[HF_KEY]}
        assert payload['params']['local_path'] == str(artifact)
        assert payload['params']['key'] == 'fixture.bin'
        assert payload['action'] in ('upload', 'storage.upload')
        calls.append(('sdk_child', payload['action']))
        return subprocess.CompletedProcess(argv, 0, b'{"ok": true, "uploaded": true}')

    @asynccontextmanager
    async def fake_mcp_transport(url, *, headers, **kwargs):
        assert url == 'https://fixture.invalid/mcp'
        assert headers == {'Authorization': 'Bearer ' + SECRETS[MCP_KEY]}
        calls.append(('mcp_transport', 'open'))
        yield None, None, None

    class FakeMcpSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def initialize(self):
            calls.append(('mcp_session', 'initialize'))

        async def list_tools(self):
            calls.append(('mcp_session', 'tools/list'))
            return SimpleNamespace(tools=[], nextCursor=None)

        async def call_tool(self, tool, args):
            calls.append(('mcp_session', 'tools/call'))
            assert tool == 'list' and args == {}
            return SimpleNamespace(isError=False, content=[])

    secret_token = set_secret_scope(SECRETS, profile_home=str(home))
    stack.callback(reset_secret_scope, secret_token)
    stack.enter_context(patch.object(credentials, 'resolve_secret', side_effect=record_secret))
    stack.enter_context(patch.object(providers, 'resolve_secret', side_effect=record_secret))
    stack.enter_context(patch('urllib.request.build_opener', return_value=MockOpener()))
    stack.enter_context(patch.object(subprocess, 'run', side_effect=fake_sdk_child))

    create = arguments('fixture-runpod', 'create',
                       config={'name': 'fixture-pod', 'image': 'fixture-image:1',
                               'gpu': {'id': 'fixture-gpu', 'count': 1}},
                       budget_usd=1, timeout_seconds=30, idempotency_key='fixture-create')
    provider_upload = arguments('fixture-hf', 'upload', key='fixture.bin', local_path=artifact.name)
    storage_upload = {'operation': 'storage.upload',
                      'parameters': {'connection_id': 'fixture-hf', 'key': 'fixture.bin',
                                     'local_path': str(artifact)}}
    mcp_list = arguments('fixture-mcp', 'list')
    blocked = (
        (PROVIDER, 'create', create), (PROVIDER, 'upload', provider_upload),
        (PROVIDER, 'list', mcp_list), (PROVIDER, 'probe', arguments('fixture-mcp', 'probe')),
        (RUN, 'storage.upload', storage_upload),
    )

    def raw_registry(name, params):
        return registry.dispatch(name, params, scope=manager.scope_key)

    for dispatch in (raw_registry, handle_function_call):
        for name, action, params in blocked:
            assert_denied(dispatch(name, params), name, action)
        assert not calls, 'A denied call touched credential lookup or a transport'
        assert not (artifact.parent / '.ai-lab').exists(), 'A denied creation recorded an intent'
        assert not (home / 'ai-lab' / 'state').exists(), 'A denied transfer created job state'
    print('PASS: real Hermes SDK registered 22 tools and denied writes in registry and model dispatch')

    # AI Lab targets the v1 client entry point, which newer Hermes MCP
    # installations may omit. Simulate it only for the allowed dispatches.
    stack.enter_context(patch.object(streamable_http, 'streamablehttp_client', fake_mcp_transport, create=True))
    stack.enter_context(patch.object(mcp, 'ClientSession', FakeMcpSession))
    for action in ('list', 'probe'):
        assert result(raw_registry(PROVIDER, arguments('fixture-runpod', action)))['ok']
    mcp_probe = result(raw_registry('ai_lab_connection_probe', {'connection_id': 'fixture-mcp'}))
    assert mcp_probe.get('state') == 'tools_listed'
    assert calls == [
        ('credential', 'hermes:' + RUNPOD_KEY), ('runpod', 'GET'),
        ('credential', 'hermes:' + RUNPOD_KEY), ('runpod', 'GET'),
        ('credential', 'hermes:' + MCP_KEY), ('mcp_transport', 'open'),
        ('mcp_session', 'initialize'), ('mcp_session', 'tools/list'),
    ]
    print('PASS: Runpod metadata and upstream MCP tools/list reached simulated transports')
    calls.clear()

    parser = argparse.ArgumentParser()
    _attach_plugin_cli_command(parser.add_subparsers(required=True), manager._cli_commands['ai-lab'])

    def cli(command, params):
        args_file = home / 'operator-arguments.json'
        args_file.write_text(json.dumps(params), encoding='utf-8')
        args = parser.parse_args(['ai-lab', command, '--args-file', str(args_file)])
        output = io.StringIO()
        with redirect_stdout(output):
            code = args.func(args)
        data = result(output.getvalue())
        assert code == 0 and data.get('ok') is True, (command, 'native CLI failed')
        return data

    assert cli('provider-action', create)['resource']['id'] == 'fixture-pod'
    intents = list((artifact.parent / '.ai-lab' / 'intents').glob('*.json'))
    assert len(intents) == 1
    assert json.loads(intents[0].read_text())['state'] == 'submitted'
    assert cli('provider-action', provider_upload)['uploaded']
    assert cli('provider-action', mcp_list)['tool'] == 'list'
    assert cli('run', storage_upload)['job']['state'] == 'succeeded'
    assert calls == [
        ('credential', 'hermes:' + RUNPOD_KEY), ('runpod', 'POST'),
        ('credential', 'hermes:' + HF_KEY), ('sdk_child', 'upload'),
        ('credential', 'hermes:' + MCP_KEY), ('mcp_transport', 'open'),
        ('mcp_session', 'initialize'), ('mcp_session', 'tools/call'),
        ('sdk_dependency', 'huggingface_hub'),
        ('credential', 'hermes:' + HF_KEY), ('sdk_child', 'storage.upload'),
    ]
    calls.clear()
    for name, action, params in blocked:
        assert_denied(raw_registry(name, params), name, action)
    assert not calls, 'Named CLI authority leaked into later registry dispatch'
    print('PASS: native named CLI reached simulated writes and released operator authority')


def main():
    source = Path(__file__).resolve().parents[1]
    review = source / '.local' / 'review'
    review.mkdir(parents=True, exist_ok=True)
    network_attempts = []

    def deny_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError('Unexpected network I/O')

    with tempfile.TemporaryDirectory(prefix='native-provider-', dir=review) as tmp, ExitStack() as stack:
        root = Path(tmp)
        home = root / 'hermes'
        bundled = root / 'bundled-plugins'
        bundled.mkdir()
        artifact = stage(source, home)
        stack.enter_context(patch.dict(os.environ, {
            'HOME': str(root), 'HERMES_HOME': str(home), 'TMPDIR': str(root),
            'HERMES_BUNDLED_PLUGINS': str(bundled), 'PATH': os.defpath,
        }, clear=True))
        stack.enter_context(patch.object(tempfile, 'tempdir', str(root)))
        for name in ('create_connection', 'getaddrinfo'):
            stack.enter_context(patch.object(socket, name, deny_network))
        for name in ('connect', 'connect_ex', 'sendto'):
            stack.enter_context(patch.object(socket.socket, name, deny_network))
        check_dispatches(stack, home, artifact)
        assert not network_attempts, 'A real network operation was attempted'


if __name__ == '__main__':
    main()
