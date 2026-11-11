"""Profile-scoped lifecycle for configurations qualified by a local controller job."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from zipfile import BadZipFile

from . import controller, jobs
from .controller_contract import MAX_INPUT_BYTES, fields, name, number, sha256, text, validate
from .controller_engine import _atomic_write, _json as canonical_json
from .controller_measure import PinnedAssets
from .endpoint import failure
from .integration_config import permitted_path
from .registry_archive import KIND as PACKAGE_KIND, export_bundle, read_bundle

KIND = 'ai-lab-registry-v1'
GENERATION_KIND = 'ai-lab-active-generation-v1'
QUALIFIED = {'qualified_candidate', 'retained_baseline'}
RUNTIME_FIELDS = ('mlx_version', 'mlx_lm_version', 'python_version')
RUNTIME_SOURCES = ('controller_measure.py', 'json_constraints.py', 'skill_runtime.py', 'skill_worker.py')
MAX_STATE_BYTES = 8 * 1024**2
MAX_ENTRIES = 128
VARIANT_FIELDS = {'id', 'system_prompt', 'input_encoding', 'max_tokens', 'prefill_step_size'}
HOLDOUT_METRICS = ('quality', 'p95_latency_ms', 'mlx_active_plus_cache_bound_bytes', 'requests_per_second')
PROOF_CONTRACT_FIELDS = ('usage', 'objective', 'grader', 'constraints', 'repetitions',
                         'minimum_relative_gain', 'quality_noninferiority_tolerance')
ACTIONS = {
    'capture': ('job_id',), 'list': (), 'inspect': ('entry_id',),
    'export': ('entry_id', 'output_path'), 'import': ('package_path', 'model'),
    'activate': ('entry_id',), 'rollback': (), 'status': (),
    'generate': ('input', 'output_dir'), 'result': ('job_id',),
}


class RegistryError(ValueError):
    pass


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value, settings, *, output=False, must_exist=True):
    path = permitted_path(value, settings, output=output, must_exist=must_exist)
    if Path(value) != path:
        raise ValueError('path must be absolute and canonical')
    return path


def _receipt(job, operation, settings, *, states=('succeeded',)):
    metadata = job['metadata']
    if (job['state'] not in states or job['operation'] != operation or job['backend'] != 'mlx'
            or metadata.get('profile_state') != settings['state_dir']
            or not sha256(metadata.get('request_sha256')) or not sha256(metadata.get('report_sha256'))):
        raise RegistryError('job_unverified', 'A succeeded job with a report receipt in this profile is required.')
    path = _path(metadata['report_path'], settings, output=True)
    if path.name != 'report.json':
        raise ValueError('report name')
    raw = controller._read(path, controller.MAX_REPORT_BYTES, owned=True)
    if _digest(raw) != metadata['report_sha256']:
        raise RegistryError('integrity_error', 'The job report differs from its completion receipt.')
    return path, raw


def _job(job_id, settings):
    result = jobs.get_job(job_id, state_dir=settings['state_dir'])
    if not result.get('ok'):
        raise RegistryError('job_not_found', 'The job is unavailable in this profile.')
    return result['job']


def _identity(value):
    if (not fields(value, ('model_sha256', 'adapter_sha256')) or not sha256(value['model_sha256'])
            or not (value['adapter_sha256'] is None or sha256(value['adapter_sha256']))):
        raise ValueError('asset identity')
    return value


def _runtime(hardware):
    versions = {key: hardware[key] for key in RUNTIME_FIELDS}
    sources = {key: hardware['source_sha256'][key] for key in RUNTIME_SOURCES}
    if (hardware.get('metal_available') is not True or 'apple-silicon-mlx' not in hardware.get('profiles', [])
            or not all(text(value, 120) for value in versions.values())
            or not all(sha256(value) for value in sources.values())):
        raise ValueError('runtime provenance')
    return {'versions': versions, 'sources': sources}


def _verify_assets(variant, identity, settings):
    try:
        for key in ('model', 'adapter'):
            if key in variant and not _path(variant[key], settings).is_dir():
                raise ValueError('asset directory')
        matches = PinnedAssets().identify(variant) == identity
    except (OSError, ValueError, RuntimeError):
        matches = False
    if not matches:
        raise RegistryError('asset_changed', 'The local assets differ from the recorded configuration.')


def _capture(job_id, settings):
    job = _job(job_id, settings)
    path, raw = _receipt(job, 'optimization.search', settings)
    report = controller._json(raw)
    summary = controller._report({'action': 'report', 'report_path': str(path)}, settings)
    if summary.get('report_sha256') != _digest(raw):
        raise RegistryError('integrity_error', 'The report changed during capture.')
    if summary.get('status') not in QUALIFIED:
        raise RegistryError('not_qualified', 'The controller must accept a configuration after a full evaluation.')
    selected = summary['accepted_variant_id']
    if not all(controller._eligible(summary[split], selected) for split in ('dev', 'test')):
        raise RegistryError('not_qualified', 'The accepted configuration must pass the development and holdout gates.')

    request_raw = controller._read(path.with_name('request.json'), controller.MAX_REQUEST_BYTES, owned=True)
    request_digest = _digest(request_raw)
    request = controller._json(request_raw)
    provenance = report['provenance']
    if (request_digest != job['metadata']['request_sha256']
            or request_digest != provenance['hardware']['request_sha256']
            or request['output_dir'] != str(path.parent)
            or request['workspace'] != str(Path(settings['workspace_root']).resolve())):
        raise RegistryError('integrity_error', 'The frozen request differs from the submitted controller job.')
    protocol = validate(request['protocol'])
    data = request['source_sha256']
    if (_digest(canonical_json(protocol).encode()) != provenance['protocol_sha256']
            or data['protocol'] != provenance['hardware']['protocol_source_sha256']
            or {key: data[key] for key in ('manifest', 'dev', 'test')} != provenance['hardware']['source_data_sha256']
            or data['manifest'] != provenance['dataset_manifest_sha256']
            or {f'{key}.jsonl': data[key] for key in ('dev', 'test')} != provenance['dataset_sha256']
            or protocol['baseline_id'] != summary['baseline_variant_id']
            or protocol.get('holdout_candidate_id') != summary.get('holdout_candidate_id')
            or protocol.get('evaluation_scope', 'full') != 'full'):
        raise RegistryError('integrity_error', 'The report provenance differs from the frozen protocol or dataset hashes.')
    frozen_raw = controller._read(path.with_name('selection.json'), controller.MAX_REPORT_BYTES, owned=True)
    frozen = controller._json(frozen_raw)
    if (_digest(frozen_raw) != summary['selection_sha256']
            or any(frozen.get(key) != report.get(key) for key in (
                'kind', 'baseline_variant_id', 'selected_variant_id', 'selection_basis', 'holdout_candidate_id',
                'evaluation_scope', 'provenance', 'dev',
            ))):
        raise RegistryError('integrity_error', 'The frozen selection differs from the controller report.')
    variant = next((variant for variant in protocol['variants'] if variant['id'] == selected), None)
    if variant is None:
        raise ValueError('accepted variant missing')
    identity = _identity(provenance['identities'][selected])
    runtime = _runtime(provenance['hardware'])
    _verify_assets(variant, identity, settings)
    return {
        'entry_id': job_id, 'qualification': 'local_controller', 'variant': variant,
        'identity': identity, 'runtime': runtime,
        'proof': {'status': summary['status'], 'report_sha256': summary['report_sha256'],
                  'selection_sha256': summary['selection_sha256'], 'protocol_sha256': provenance['protocol_sha256'],
                  'dataset_manifest_sha256': provenance['dataset_manifest_sha256'],
                  'evaluated_adapter_sha256': identity['adapter_sha256'],
                  'optimization_gain_verified': summary['optimization_gain_verified'],
                  'contract': {key: protocol[key] for key in PROOF_CONTRACT_FIELDS},
                  'holdout': {key: summary['test'][selected][key] for key in HOLDOUT_METRICS}},
    }


@contextmanager
def _locked(settings):
    root = jobs._root(settings['state_dir'])
    with os.fdopen(jobs._owned_file(root / 'model-registry.lock'), 'r+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'model-registry.json'
        if os.path.lexists(path):
            state = controller._json(controller._read(path, MAX_STATE_BYTES, owned=True))
            if (not fields(state, ('kind', 'entries', 'active', 'previous', 'has_rollback'))
                    or state['kind'] != KIND or type(state['entries']) is not dict
                    or len(state['entries']) > MAX_ENTRIES or type(state['has_rollback']) is not bool
                    or any(pointer is not None and pointer not in state['entries']
                           for pointer in (state['active'], state['previous']))):
                raise ValueError('registry state')
        else:
            state = {'kind': KIND, 'entries': {}, 'active': None, 'previous': None, 'has_rollback': False}
        yield path, state


def _save(path, state):
    if len(canonical_json(state).encode()) + 1 > MAX_STATE_BYTES:
        raise RegistryError('registry_limit', 'The configuration registry is full.')
    _atomic_write(path, state)


def _entry(state, entry_id):
    if not name(entry_id) or entry_id not in state['entries']:
        raise RegistryError('entry_not_found', 'The configuration is unavailable in this profile.')
    return state['entries'][entry_id]


def _add(path, state, entry):
    identifier = entry['entry_id']
    existing = state['entries'].get(identifier)
    if existing == entry:
        return
    if existing is not None:
        legacy = {**entry, 'proof': {key: value for key, value in entry['proof'].items() if key != 'contract'}}
        if existing != legacy:
            raise RegistryError('registry_conflict', 'This configuration identifier already exists in the profile.')
    elif len(state['entries']) >= MAX_ENTRIES:
        raise RegistryError('registry_limit', 'The configuration registry is full or already contains this identifier.')
    state['entries'][identifier] = entry
    _save(path, state)


def _public(entry, state, *, detail=False):
    variant = entry['variant']
    output = {'entry_id': entry['entry_id'], 'variant_id': variant['id'],
              'qualification': entry['qualification'], 'active': state['active'] == entry['entry_id'],
              'identity': entry['identity'], 'proof': entry['proof'], 'remote_execution_verified': False}
    if detail:
        output['configuration'] = variant
        output['runtime'] = entry['runtime']
    return output


def _portable(entry):
    return {'variant': {key: value for key, value in entry['variant'].items() if key not in ('model', 'adapter')},
            'identity': entry['identity'], 'proof': entry['proof']}


def _validate_bundle(manifest):
    if not fields(manifest, ('kind', 'variant', 'identity', 'proof', 'files')) or manifest['kind'] != PACKAGE_KIND:
        raise ValueError('package manifest')
    variant = manifest['variant']
    if (not fields(variant, VARIANT_FIELDS, ('output_constraint',)) or not name(variant['id'])
            or not text(variant['system_prompt'], MAX_INPUT_BYTES, empty=True)
            or variant['input_encoding'] not in ('plain', 'json_string')
            or not number(variant['max_tokens'], 1, 128, integer=True)
            or type(variant['prefill_step_size']) is not int or variant['prefill_step_size'] not in (64, 256, 2048)):
        raise ValueError('package variant')
    if 'output_constraint' in variant:
        from .json_constraints import canonical_choices

        canonical_choices(variant['output_constraint'])
    _identity(manifest['identity'])
    proof = manifest['proof']
    hashes = ('report_sha256', 'selection_sha256', 'protocol_sha256', 'dataset_manifest_sha256')
    if (not fields(proof, ('status', 'holdout', 'optimization_gain_verified', 'evaluated_adapter_sha256', *hashes),
                   ('contract',))
            or proof['status'] not in QUALIFIED or type(proof['optimization_gain_verified']) is not bool
            or (proof['evaluated_adapter_sha256'] is not None and not sha256(proof['evaluated_adapter_sha256']))
            or (proof['evaluated_adapter_sha256'] is None) != (manifest['identity']['adapter_sha256'] is None)
            or not all(sha256(proof[key]) for key in hashes)
            or not fields(proof['holdout'], HOLDOUT_METRICS)
            or not all(number(proof['holdout'][key], 0, 2**60) for key in HOLDOUT_METRICS)
            or proof['holdout']['quality'] > 1):
        raise ValueError('package proof')
    if 'contract' in proof:
        contract = proof['contract']
        if not fields(contract, PROOF_CONTRACT_FIELDS):
            raise ValueError('package evaluation contract')
        # Validate the portable contract using the same rules as the original
        # controller. Runtime paths and execution budgets are bound locally.
        validate({
            'version': controller.VERSION, **contract, 'baseline_id': variant['id'],
            'variants': [{**variant, 'model': '/portable-model'}],
            'budget': {'max_wall_seconds': 60, 'max_trials': 1, 'max_generated_tokens': 65_536},
        })


def _import_bundle(request, settings):
    package = _path(request['package_path'], settings)
    model = _path(request['model'], settings)
    workspace = Path(settings['workspace_root']).resolve(strict=True)
    imports = workspace / 'model-registry-imports'
    imports.mkdir(mode=0o700, exist_ok=True)
    if imports.is_symlink() or not imports.is_dir() or imports.stat().st_uid != os.getuid():
        raise ValueError('unsafe import directory')
    with tempfile.TemporaryDirectory(prefix='.staged-', dir=imports) as stage_name:
        stage = Path(stage_name)
        manifest = read_bundle(package, stage, _validate_bundle)
        variant = {**manifest['variant'], 'model': str(model)}
        if manifest['files']:
            variant['adapter'] = str(stage)
        _verify_assets(variant, manifest['identity'], settings)
        identifier = str(uuid.uuid4())
        destination = imports / identifier
        with _locked(settings) as (path, state):
            entry = {'entry_id': identifier, 'qualification': 'imported_unverified', 'variant': variant,
                     'identity': manifest['identity'], 'runtime': None, 'proof': manifest['proof']}
            moved = False
            try:
                if manifest['files']:
                    destination.mkdir(mode=0o700)
                    moved = True
                    for item in stage.iterdir():
                        item.replace(destination / item.name)
                    variant['adapter'] = str(destination)
                _add(path, state, entry)
            except BaseException:
                if moved:
                    shutil.rmtree(destination)
                raise
            return _public(entry, state, detail=True)


def _qualified(entry, settings):
    if entry['qualification'] != 'local_controller':
        raise RegistryError('not_qualified', 'Run a local controller evaluation and capture its accepted configuration.')
    _verify_assets(entry['variant'], entry['identity'], settings)


def _status(state):
    return {'active_entry_id': state['active'], 'previous_entry_id': state['previous'],
            'rollback_available': state['has_rollback'], 'remote_execution_verified': False}


def _submit_generation(entry, request, settings):
    if not text(request['input'], MAX_INPUT_BYTES):
        raise ValueError('input size')
    output = _path(request['output_dir'], settings, output=True, must_exist=False)
    if not output.parent.is_dir() or os.path.lexists(output):
        raise RegistryError('output_exists', 'Generation requires a new output directory under the profile workspace.')
    python = Path(settings.get('backend_paths', {}).get('mlx.python', ''))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RegistryError('configuration_required', 'Configure an executable as backend_paths.mlx.python.')
    workspace = Path(settings['workspace_root']).resolve(strict=True)
    payload = {'kind': GENERATION_KIND, 'entry_id': entry['entry_id'], 'variant': entry['variant'],
               'identity': entry['identity'], 'runtime': entry['runtime'], 'input': request['input'],
               'workspace': str(workspace), 'output_dir': str(output)}
    raw = canonical_json(payload).encode()
    digest = _digest(raw)
    output.mkdir(mode=0o700)
    request_path = output / 'request.json'
    with os.fdopen(os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
        stream.write(raw)
    report = output / 'report.json'
    return jobs.submit({
        'operation': 'registry.generate', 'backend': 'mlx', 'cwd': str(workspace),
        'command': [str(python), '-I', '-B', str(Path(__file__).with_name('registry_worker.py')), str(request_path), digest],
        'env': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}, 'timeout_seconds': 120,
        'outputs': [{'path': str(report), 'kind': 'file'}], 'report_receipt': True,
        'metadata': {'profile_state': settings['state_dir'], 'report_path': str(report),
                     'request_sha256': digest, 'entry_id': entry['entry_id'], 'cloud': False},
    }, state_dir=settings['state_dir'])


def _result(job_id, settings):
    job = _job(job_id, settings)
    if job['operation'] != 'registry.generate':
        raise RegistryError('job_not_found', 'The generation job is unavailable in this profile.')
    if job['state'] != 'succeeded' and not sha256(job['metadata'].get('report_sha256')):
        return {'ok': True, 'job_id': job_id, 'state': job['state'], 'generation_verified': False}
    _, raw = _receipt(job, 'registry.generate', settings, states=('succeeded', 'failed'))
    result = controller._json(raw)
    if (result['kind'] != GENERATION_KIND or result['ok'] is not (job['state'] == 'succeeded')
            or result['request_sha256'] != job['metadata']['request_sha256']
            or result['entry_id'] != job['metadata']['entry_id']):
        raise ValueError('generation result')
    if not result['ok']:
        code = result['error']['code']
        if code not in {'compute_busy', 'runtime_unavailable', 'runtime_changed', 'integrity_error',
                        'invalid_input', 'generation_failed'}:
            raise ValueError('generation error')
        return {'job_id': job_id, 'state': 'failed', 'generation_verified': False,
                **failure(code, 'The active configuration could not complete local generation.')}
    if (not text(result['content'], 128 * 1024, empty=True) or result['artifact_identity_verified'] is not True
            or result['generation_verified'] is not True):
        raise ValueError('generation result')
    with _locked(settings) as (_, state):
        entry = _entry(state, result['entry_id'])
        if _identity(result['identity']) != entry['identity'] or result['variant_id'] != entry['variant']['id']:
            raise ValueError('generation identity')
    return {'ok': True, 'job_id': job_id, 'state': 'succeeded', 'entry_id': result['entry_id'],
            'variant_id': result['variant_id'], 'content': result['content'], 'identity': result['identity'],
            'generation_verified': True, 'artifact_identity_verified': True, 'remote_execution_verified': False}


def dispatch(request, settings):
    try:
        if type(request) is not dict or not isinstance(request.get('action'), str):
            raise ValueError('action')
        action = request['action']
        if action not in ACTIONS or not fields(request, ('action', *ACTIONS[action])):
            raise ValueError('arguments')
        if action == 'capture':
            entry = _capture(request['job_id'], settings)
            with _locked(settings) as (path, state):
                _add(path, state, entry)
                return {'ok': True, **_public(entry, state)}
        if action == 'import':
            return {'ok': True, **_import_bundle(request, settings)}
        if action == 'result':
            return _result(request['job_id'], settings)
        with _locked(settings) as (path, state):
            if action == 'list':
                return {'ok': True, **_status(state), 'entries': [_public(entry, state) for entry in state['entries'].values()]}
            if action == 'status':
                return {'ok': True, **_status(state)}
            if action == 'rollback':
                if not state['has_rollback']:
                    raise RegistryError('no_rollback', 'There is no previous activation to restore.')
                previous = state['previous']
                if previous is not None:
                    _qualified(_entry(state, previous), settings)
                state.update(active=previous, previous=None, has_rollback=False)
                _save(path, state)
                return {'ok': True, **_status(state)}
            if action == 'generate':
                if state['active'] is None:
                    raise RegistryError('no_active_configuration', 'Activate a qualified configuration before generating.')
                return _submit_generation(_entry(state, state['active']), request, settings)
            entry = _entry(state, request['entry_id'])
            if action == 'inspect':
                return {'ok': True, **_public(entry, state, detail=True)}
            if action == 'activate':
                _qualified(entry, settings)
                if state['active'] != entry['entry_id']:
                    state.update(active=entry['entry_id'], previous=state['active'], has_rollback=True)
                    _save(path, state)
                return {'ok': True, **_status(state)}
            if action == 'export':
                _verify_assets(entry['variant'], entry['identity'], settings)
                output = _path(request['output_path'], settings, output=True, must_exist=False)
                if not output.parent.is_dir() or os.path.lexists(output):
                    raise RegistryError('output_exists', 'Export requires a new file under the profile workspace.')
                export_bundle(output, _portable(entry), entry['variant'].get('adapter'))
                return {'ok': True, 'entry_id': entry['entry_id'], 'output_path': str(output),
                        'remote_execution_verified': False}
    except RegistryError as error:
        return failure(*error.args)
    except (AttributeError, BadZipFile, EOFError, KeyError, OSError, OverflowError, RecursionError,
            RuntimeError, TypeError, ValueError, sqlite3.Error):
        pass
    return failure('invalid_registry_request', 'The registry request, configuration, package or local files could not be verified.')
