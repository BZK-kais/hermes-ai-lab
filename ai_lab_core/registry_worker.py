"""Execute the exact frozen generation settings of an active configuration."""
import fcntl
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import monotonic

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_lab_core import controller, jobs
from ai_lab_core.controller_contract import MAX_INPUT_BYTES, fields, name, sha256, text
from ai_lab_core.controller_measure import MeasureError, PinnedAssets, _generate, _prompt_ids
from ai_lab_core.controller_worker import _private_directory
from ai_lab_core.controller_engine import _json as canonical_json
from ai_lab_core.json_constraints import compile_constraint
from ai_lab_core.profile import device_lock_path
from ai_lab_core.registry import GENERATION_KIND, RUNTIME_FIELDS, RUNTIME_SOURCES, VARIANT_FIELDS, _digest, _identity
from ai_lab_core.skill_worker import CACHE_LIMIT_BYTES

MAX_CONTENT = 128 * 1024


def _read_request(path, expected, workspace):
    if path.name != 'request.json' or not _private_directory(path.parent, workspace) or not sha256(expected):
        raise ValueError('request path')
    raw = controller._read(path, controller.MAX_REQUEST_BYTES, owned=True)
    if _digest(raw) != expected:
        raise ValueError('request digest')
    payload = controller._json(raw)
    if (not fields(payload, ('kind', 'entry_id', 'variant', 'identity', 'runtime', 'input', 'workspace', 'output_dir'))
            or payload['kind'] != GENERATION_KIND or not name(payload['entry_id'])
            or payload['workspace'] != str(workspace) or payload['output_dir'] != str(path.parent)
            or not text(payload['input'], MAX_INPUT_BYTES)):
        raise ValueError('request payload')
    variant = payload['variant']
    if (not fields(variant, (*VARIANT_FIELDS, 'model'), ('adapter', 'output_constraint'))
            or not name(variant['id']) or not text(variant['system_prompt'], MAX_INPUT_BYTES, empty=True)
            or variant['input_encoding'] not in ('plain', 'json_string')
            or type(variant['max_tokens']) is not int or not 1 <= variant['max_tokens'] <= 128
            or type(variant['prefill_step_size']) is not int or variant['prefill_step_size'] not in (64, 256, 2048)):
        raise ValueError('variant')
    _identity(payload['identity'])
    runtime = payload['runtime']
    if (not fields(runtime, ('sources', 'versions')) or not fields(runtime['sources'], RUNTIME_SOURCES)
            or not fields(runtime['versions'], RUNTIME_FIELDS)
            or not all(sha256(digest) for digest in runtime['sources'].values())
            or not all(text(value, 120) for value in runtime['versions'].values())):
        raise ValueError('runtime provenance')
    return payload


def _verified_versions(runtime):
    actual = {'python_version': '.'.join(map(str, sys.version_info[:3])),
              'mlx_version': version('mlx'), 'mlx_lm_version': version('mlx-lm')}
    source_dir = Path(__file__).resolve().parent
    return (actual == runtime['versions']
            and all(_digest(controller._read(source_dir / filename, controller.MAX_REQUEST_BYTES)) == digest
                    for filename, digest in runtime['sources'].items()))


def _execute(payload, mx, load, stream_generate):
    variant = payload['variant']
    identity = payload['identity']
    assets = PinnedAssets()
    if assets.verify(variant) != identity:
        raise MeasureError('integrity_error')
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    model, tokenizer = load(variant['model'], adapter_path=variant.get('adapter'),
                            tokenizer_config={'trust_remote_code': False, 'local_files_only': True},
                            trust_remote_code=False)
    model.eval()
    mx.eval(model.parameters())
    if assets.verify(variant) != identity:
        raise MeasureError('integrity_error')
    try:
        ids = _prompt_ids(tokenizer, variant, payload['input'])
    except MeasureError:
        raise MeasureError('invalid_input') from None
    constraint = None
    if 'output_constraint' in variant:
        constraint = compile_constraint(mx, tokenizer, variant['output_constraint'], variant['max_tokens'])
    output, _ = _generate(stream_generate, model, tokenizer, ids, variant['max_tokens'],
                          variant['prefill_step_size'], monotonic() + 90, lambda: None, constraint)
    if assets.verify(variant) != identity:
        raise MeasureError('integrity_error')
    if not text(output, MAX_CONTENT, empty=True):
        raise MeasureError('generation_failed')
    return {'ok': True, 'variant_id': variant['id'], 'identity': identity, 'content': output,
            'generation_verified': True, 'artifact_identity_verified': True}


def _error(code):
    return {'ok': False, 'error': {'code': code}}


def _publish(path, result):
    fd, temporary = tempfile.mkstemp(prefix='.generation-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical_json(result).encode() + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _run(payload):
    if sys.platform != 'darwin' or platform.machine().lower() not in ('arm64', 'aarch64'):
        return _error('runtime_unavailable')
    lock = device_lock_path()
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(jobs._owned_file(lock), 'r+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _error('compute_busy')
        try:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            if not _verified_versions(payload['runtime']):
                return _error('runtime_changed')
            import mlx.core as mx
            from mlx_lm import load, stream_generate

            if not mx.metal.is_available():
                return _error('runtime_unavailable')
            mx.set_default_device(mx.gpu)
        except (ImportError, OSError, RuntimeError):
            return _error('runtime_unavailable')
        try:
            return _execute(payload, mx, load, stream_generate)
        except MeasureError as error:
            code = str(error)
            return _error(code if code in ('integrity_error', 'invalid_input') else 'generation_failed')
        except Exception:
            return _error('generation_failed')


def main():
    with open(os.devnull, 'wb') as discard:
        os.dup2(discard.fileno(), 1)
        os.dup2(discard.fileno(), 2)
    if len(sys.argv) != 3:
        return 1
    try:
        path = Path(sys.argv[1])
        expected = sys.argv[2]
        payload = _read_request(path, expected, Path.cwd().resolve())
        report = path.with_name('report.json')
        if os.path.lexists(report):
            return 1
        try:
            result = _run(payload)
        except Exception:
            result = _error('generation_failed')
        _publish(report, {'kind': GENERATION_KIND, 'entry_id': payload['entry_id'],
                          'request_sha256': expected, **result})
        return 0 if result['ok'] else 1
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError, OverflowError, RecursionError):
        return 1


if __name__ == '__main__':
    sys.exit(main())
