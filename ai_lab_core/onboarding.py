"""Prepare one AI Lab workspace and inspect local prerequisites offline."""

import fcntl
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import stat
import tempfile

from .config import load_settings
from .datasets import validate_dataset
from .endpoint import failure
from .integration_config import load_integrations, parse_integrations, read_integrations
from .profile import config_path, data_root


def _installed(distribution):
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


def _check(settings):
    workspace = Path(settings['workspace_root'])
    state = Path(settings['state_dir'])
    interpreter = settings['backend_paths'].get('mlx.python')
    sample = Path(__file__).resolve().with_name('assets') / 'smoke-data'
    dataset = validate_dataset(str(sample), {**settings, 'data_roots': [str(sample)]})
    if not dataset['ok']:
        return failure('installation_incomplete', 'The bundled smoke dataset could not be validated.')
    return {
        'ok': True,
        'smoke_dataset': {'valid': True, 'splits': len(dataset['splits'])},
        'workspace': {'path': str(workspace), 'ready': workspace.is_dir() and os.access(workspace, os.W_OK)},
        'state': {'path': str(state), 'ready': state.is_dir() and os.access(state, os.W_OK)},
        'model_roots': {'configured': len(settings['model_roots']),
                        'present': sum(Path(path).is_dir() for path in settings['model_roots'])},
        'mlx': {'state': ('executable_present' if Path(interpreter).is_file() and os.access(interpreter, os.X_OK)
                          else 'executable_missing') if interpreter else 'unconfigured',
                'generation_verified': False},
        'optional_packages_in_current_python': {
            name: _installed(name) for name in ('mcp', 'huggingface-hub', 'boto3', 'google-cloud-storage')
        },
        'configured_connections': len(settings['integrations']) + len(settings['endpoints']),
        'network_probed': False,
    }


def _directory(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('directory')
    return str(path.resolve())


def _interpreter(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('interpreter')
    return str(path.absolute())


def _append_root(roots, new_path, limit):
    for path in roots:
        if Path(path).expanduser().resolve() == Path(new_path):
            return roots
    if len(roots) >= limit:
        raise ValueError('root limit')
    return [*roots, new_path]


def _config_bytes(value, size_limit):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > size_limit:
        raise ValueError('configuration size')
    return data


def _write_config(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)


def _update(request):
    connections_path = config_path('connections.json', 'AI_LAB_CONFIG')
    integrations_path = config_path('integrations.json', 'AI_LAB_INTEGRATIONS')
    for path in (connections_path, integrations_path):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('configuration path')
    if connections_path.resolve() == integrations_path.resolve():
        raise ValueError('configuration collision')

    connections = load_settings()
    original_integrations = read_integrations(integrations_path)
    settings = parse_integrations(connections, original_integrations)
    integrations = dict(original_integrations)
    changes = {}
    if 'model_root' in request:
        raw_connections = json.loads(connections_path.read_text(encoding='utf-8')) if connections_path.exists() else {}
        original_roots = raw_connections.get('model_roots', connections['model_roots'])
        roots = _append_root(original_roots, _directory(request['model_root']), 12)
        if roots != original_roots:
            connections['model_roots'] = [str(Path(path).expanduser()) for path in roots]
            changes[connections_path] = _config_bytes({**raw_connections, 'model_roots': roots}, 32768)
    if 'data_root' in request:
        original_roots = integrations.get('data_roots', settings['data_roots'])
        roots = _append_root(original_roots, _directory(request['data_root']), 32)
        if roots != original_roots:
            integrations['data_roots'] = roots
    if 'mlx_python' in request:
        interpreter = _interpreter(request['mlx_python'])
        if interpreter != settings['backend_paths'].get('mlx.python'):
            integrations['backend_paths'] = {**integrations.get('backend_paths', {}), 'mlx.python': interpreter}
    if integrations != original_integrations:
        changes[integrations_path] = _config_bytes(integrations, 65536)
    settings = parse_integrations(connections, integrations)

    for key in ('workspace_root', 'state_dir'):
        Path(settings[key]).mkdir(mode=0o700, parents=True, exist_ok=True)
    for path, data in changes.items():
        _write_config(path, data)
    return {**_check(settings), 'configuration_updated': bool(changes)}


def _setup(request):
    root = data_root()
    if root.is_symlink():
        raise ValueError('data root')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(root / '.setup.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'rb') as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError('setup lock')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _update(request)


def dispatch(request):
    if type(request) is not dict:
        return failure('invalid_arguments', 'Select the check or setup action.')
    action = request.get('action', 'check')
    if action == 'check' and not set(request) - {'action'}:
        try:
            return _check(load_integrations())
        except (OSError, ValueError, TypeError, RecursionError):
            return failure('invalid_config', 'AI Lab configuration could not be read.')
    options = {'model_root', 'data_root', 'mlx_python'}
    if (action != 'setup' or set(request) - options - {'action'}
            or any(type(request[key]) is not str or not 0 < len(request[key]) <= 4096
                   for key in options.intersection(request))):
        return failure('invalid_arguments', 'Supply setup with existing absolute model, data, or Python paths.')
    try:
        return _setup(request)
    except (OSError, ValueError, TypeError, RecursionError):
        return failure('setup_failed', 'Check the supplied paths and the AI Lab profile configuration.')
