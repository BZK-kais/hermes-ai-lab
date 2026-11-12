"""JSON CLI over the same tool contracts used by Hermes, MCP and HTTP."""
import argparse
import json
from pathlib import Path
from . import tool_api
from .operator_policy import operator_scope


def _params(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--params', help='JSON object of named operation parameters')
    group.add_argument('--params-file', help='Exact local JSON file (bounded to 256 KiB)')


def _json_object(raw=None, filename=None):
    if filename is not None:
        with Path(filename).open('rb') as stream:
            data = stream.read(tool_api.MAX_JSON_BYTES + 1)
        if len(data) > tool_api.MAX_JSON_BYTES:
            raise ValueError('size')
        raw = data.decode('utf-8')
    raw = raw if raw is not None else '{}'
    if len(raw.encode('utf-8')) > tool_api.MAX_JSON_BYTES:
        raise ValueError('size')
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError('object')
    return value


def main():
    parser = argparse.ArgumentParser(description='AI Lab shared connection and execution tools')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    sub.add_parser('profile')
    setup = sub.add_parser('setup', help='Initialize or inspect this profile without downloading models')
    setup.add_argument('--check', action='store_true')
    for flag in ('model-root', 'data-root', 'mlx-python'):
        setup.add_argument('--' + flag)
    for verb in ('analyze', 'benchmark', 'optimize', 'mission', 'registry', 'onboarding', 'experiment'):
        p = sub.add_parser(verb)
        p.add_argument('--args-file', required=True, help='Bounded JSON tool request')
    local = sub.add_parser('models'); local.add_argument('--limit', type=int, default=50)
    info = sub.add_parser('model-info'); info.add_argument('repo_id'); info.add_argument('--revision', default='main'); info.add_argument('--file-limit', type=int, default=30)
    probe = sub.add_parser('probe'); probe.add_argument('base_url'); probe.add_argument('--kind', choices=['llama_cpp', 'ollama', 'openai'], default='llama_cpp')
    sub.add_parser('connections')
    cp = sub.add_parser('connection-probe'); cp.add_argument('connection_id')
    sub.add_parser('operations')
    for verb in ('plan', 'run'):
        p = sub.add_parser(verb); p.add_argument('operation'); _params(p)
    job = sub.add_parser('jobs'); job.add_argument('action', nargs='?', default='list', choices=['list', 'get', 'status', 'logs', 'cancel', 'reconcile']); job.add_argument('job_id', nargs='?'); job.add_argument('--limit', type=int, default=20)
    provider = sub.add_parser('provider-action'); provider.add_argument('connection_id'); provider.add_argument('action'); _params(provider)
    dataset = sub.add_parser('dataset-validate'); dataset.add_argument('path')
    artifact = sub.add_parser('artifact-inspect'); artifact.add_argument('path'); artifact.add_argument('--hash-files', action='store_true')
    generate = sub.add_parser('generate'); generate.add_argument('connection_id'); generate.add_argument('model'); generate.add_argument('prompt'); generate.add_argument('--max-tokens', type=int, default=128)
    generic = sub.add_parser('call'); generic.add_argument('name')
    group = generic.add_mutually_exclusive_group()
    group.add_argument('--args'); group.add_argument('--args-file')
    sub.add_parser('mcp')
    api = sub.add_parser('api'); api.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.command == 'mcp':
        from .mcp_server import main as serve
        return serve()
    if args.command == 'api':
        from .http_api import main as serve
        from .credentials import CredentialError
        try:
            return serve(args.port)
        except (OSError, ValueError, CredentialError):
            print(tool_api.encode(tool_api.failure('api_start_failed', 'The local API could not be started.')))
            raise SystemExit(1)
    values = vars(args).copy(); command = values.pop('command')
    name = {'models': 'list_local_models', 'model-info': 'hf_model_info', 'probe': 'probe_endpoint', 'jobs': 'job'}.get(command, command.replace('-', '_'))
    try:
        if command == 'setup':
            from .onboarding import dispatch as configure
            request = {'action': 'check' if values.pop('check') else 'setup'}
            request.update({key: value for key, value in values.items() if value is not None})
            result = configure(request)
            print(tool_api.encode(result))
            raise SystemExit(0 if result.get('ok') else 1)
        if command in ('plan', 'run', 'provider-action'):
            values['parameters'] = _json_object(values.pop('params'), values.pop('params_file'))
        if command == 'jobs' and values['job_id'] is None:
            del values['job_id']
        if command == 'call':
            name = values['name']
            values = _json_object(values['args'], values['args_file'])
        if command in ('analyze', 'benchmark', 'optimize', 'mission', 'registry', 'onboarding', 'experiment'):
            values = _json_object(filename=values['args_file'])
            if command != 'experiment':
                values = {'request': values}
        if not name.startswith('ai_lab_'):
            name = 'ai_lab_' + name
        if command in ('provider-action', 'run'):
            with operator_scope(command):
                result = tool_api.dispatch(name, values)
        else:
            result = tool_api.dispatch(name, values)
    except (ValueError, OSError, UnicodeError, RecursionError):
        result = tool_api.failure('invalid_arguments', 'Supply a bounded JSON object or readable parameter file.')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result.get('ok') else 1)


if __name__ == '__main__':
    main()
