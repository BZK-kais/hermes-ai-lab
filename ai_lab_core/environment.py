"""Collect a bounded, passive snapshot for optimization planning."""

from datetime import datetime, timezone
from pathlib import Path

from . import backends, hardware_profiles, local
from .integration_config import IDENTIFIER
from .profile import bound_home, profile_home


_MODEL_LIMIT = 8
_OPERATION_IDS = tuple(sorted((*backends.OPS, 'storage.download', 'storage.upload')))
_CONNECTION_TYPES = {'ssh', 'mcp', 'inference', 'runpod', 'hf_jobs', 'hf_endpoints',
                     'hf_storage', 'gcs', 's3'}
_INFERENCE_KINDS = {'llama_cpp', 'ollama', 'openai', 'mlx'}
_LIMITATIONS = (
    'Available memory is unknown. Installed memory does not estimate capacity for a workload.',
    'Kernel compatibility, topology, training and model generation were not verified.',
    'Runtime versions are local observations. Bindings are checked by path only.',
    'Model inventory is bounded and only describes files and metadata.',
    'Configured connections were not probed and credentials were not resolved.',
    'Operation IDs are declarations. Dependencies and execution were not verified.',
)


def _profile():
    scope = 'standalone'
    if bound_home() is not None:
        scope = 'bound'
    elif profile_home() is not None:
        scope = 'ambient'
    return {'scope': scope}


def _observed(result):
    if result.get('ok') is not True:
        raise ValueError('Inventory source failed')
    return result


def _hardware():
    result = _observed(hardware_profiles.detect_profile())
    host = result.get('hardware', {})
    devices = [
        {**{key: device.get(key) for key in ('name', 'profile', 'compute_capability',
                                            'memory_mib', 'memory_kind', 'driver_version')},
         'available_memory': None}
        for device in result['devices']
    ]
    return {
        'system': result['system'], 'cpu_architecture': result['cpu_architecture'],
        'profiles': result['profiles'], 'devices': devices, 'chip': host.get('chip'),
        'memory_kind': result.get('memory_kind'), 'installed_memory': host.get('memory'),
        'available_memory': None, 'kernel_validated': False, 'topology_verified': False,
        'training_validated': False, 'generation_verified': False,
    }


def _runtimes(settings):
    result = _observed(local.runtime_inventory(settings.get('runtime_bin_dirs', []),
                                               inspect_mlx_package=False))
    return [
        {'name': item['name'], 'installed': item['installed'], 'version': item.get('version'),
         'state': 'installed' if item['installed'] else 'not_found',
         'service_reachable': None, 'generation_verified': False}
        for item in result['runtimes']
    ]


def _bindings(settings):
    paths = settings.get('backend_paths', {})
    names = {'python', *(tool for tool, _ in backends.OPS.values()), *paths}
    bindings = []
    for name in sorted(names):
        path = backends._tool(name, settings)
        bindings.append({'name': name, 'configured': name in paths,
                         'executable_present': path is not None and Path(path).is_file(),
                         'verification': 'path_only_not_imported'})
    return bindings


def _models(settings):
    result = _observed(local.inventory_models(settings.get('model_roots', []), limit=_MODEL_LIMIT))
    entries = result['models'][:_MODEL_LIMIT]
    return {
        'models': [{key: item[key] for key in ('format', 'size_bytes', 'files', 'complete', 'verification')}
                   for item in entries],
        'unique_size_bytes': result['unique_size_bytes'],
        'truncated': result['truncated'] or len(entries) == _MODEL_LIMIT,
    }


def _connection(identifier, connection_type, kind=None):
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise ValueError('Invalid connection ID')
    entry = {'id': identifier, 'type': connection_type if connection_type in _CONNECTION_TYPES else 'unknown',
             'state': 'configured_not_probed'}
    if connection_type == 'inference':
        entry['kind'] = kind if kind in _INFERENCE_KINDS else 'unknown'
    return entry


def _connections(settings):
    connections = [_connection(item['id'], item['type'], item.get('kind'))
                   for item in settings.get('integrations', [])]
    connections.extend(_connection(item['name'], 'inference', item['kind'])
                       for item in settings.get('endpoints', []))
    return connections


def collect(settings: dict) -> dict:
    """Project loaded settings and local observations without probing connections."""
    result = {
        'ok': True, 'observed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'operation_ids': list(_OPERATION_IDS), 'limitations': list(_LIMITATIONS),
    }
    sources = {
        'profile': _profile, 'hardware': _hardware, 'runtimes': lambda: _runtimes(settings),
        'bindings': lambda: _bindings(settings), 'models': lambda: _models(settings),
        'connections': lambda: _connections(settings),
    }
    errors = {}
    for name, observe in sources.items():
        try:
            result[name] = observe()
        except Exception:
            result[name] = None
            errors[name] = {'code': 'inventory_source_unavailable',
                            'message': 'This inventory source could not be read.'}
    if errors:
        result['errors'] = errors
    return result
