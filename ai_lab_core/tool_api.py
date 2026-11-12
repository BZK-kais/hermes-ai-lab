"""One bounded tool contract for Hermes, CLI, MCP and local HTTP clients."""
import copy
import importlib
import json
import math
from .optimization import request_schema
from . import __version__

MAX_JSON_BYTES = 256 * 1024
API_VERSION = __version__


def failure(code='invalid_arguments', message='Tool arguments do not match the schema.'):
    return {'ok': False, 'error': {'code': code, 'message': message}}


def _string(maximum=4096):
    return {'type': 'string', 'minLength': 1, 'maxLength': maximum}


def _integer(default, maximum, minimum=1):
    return {'type': 'integer', 'minimum': minimum, 'maximum': maximum, 'default': default}


_JSON_OBJECT = {'type': 'object', 'additionalProperties': True}
_BASELINE = {'type': 'object', 'additionalProperties': False, 'properties': {
    'model': _string(), 'adapter': _string(), 'system_prompt': {'type': 'string', 'maxLength': 16384},
    'input_encoding': {'type': 'string', 'enum': ['plain', 'json_string']},
    'max_tokens': {'type': 'integer', 'minimum': 1, 'maximum': 128}, 'prefill_step_size': {'type': 'integer', 'enum': [64, 256, 2048]},
    'output_constraint': _JSON_OBJECT,
}}
_DEFS = (
    ('status', 'Inspect AI Lab capabilities and connection evidence.', {}, ()),
    ('list_local_models', 'List configured local model files without downloading weights.', {'limit': _integer(50, 200)}, ()),
    ('hf_model_info', 'Read anonymous public Hugging Face model metadata without downloading weights.', {'repo_id': _string(256), 'revision': {**_string(256), 'default': 'main'}, 'file_limit': _integer(30, 100)}, ('repo_id',)),
    ('probe_endpoint', 'Read bounded loopback endpoint metadata without generating text.', {'base_url': _string(2048), 'kind': {'type': 'string', 'enum': ['llama_cpp', 'ollama', 'openai'], 'default': 'llama_cpp'}}, ('base_url',)),
    ('connections', 'List configured AI Lab connections and their capabilities.', {}, ()),
    ('connection_probe', 'Probe one configured connection; discovery does not provision resources.', {'connection_id': _string(128)}, ('connection_id',)),
    ('operations', 'List named operations and their parameter contracts.', {}, ()),
    ('plan', 'Validate and preview a named operation without executing it.', {'operation': _string(128), 'parameters': _JSON_OBJECT}, ('operation', 'parameters')),
    ('run', 'Execute a named, validated operation and return its durable job identifier. Remote storage uploads require the named operator CLI run command.', {'operation': _string(128), 'parameters': _JSON_OBJECT}, ('operation', 'parameters')),
    ('job', 'List, inspect, read logs, cancel, or reconcile durable AI Lab jobs.', {'action': {'type': 'string', 'enum': ['list', 'get', 'status', 'logs', 'cancel', 'reconcile'], 'default': 'list'}, 'job_id': _string(64), 'limit': {**_integer(20, 64000), 'description': 'List limit is at most 200; log character limit is at most 64000.'}}, ()),
    ('provider_action', 'Read metadata or logs from a supported provider connection. Remote changes and all upstream MCP tool calls require the named operator CLI provider-action command and may incur charges.', {'connection_id': _string(128), 'action': _string(128), 'parameters': _JSON_OBJECT}, ('connection_id', 'action', 'parameters')),
    ('dataset_validate', 'Validate a dataset at an operator-selected local path without training.', {'path': _string(4096)}, ('path',)),
    ('artifact_inspect', 'Inspect an operator-selected local model artifact; hashing is optional.', {'path': _string(4096), 'hash_files': {'type': 'boolean', 'default': False}}, ('path',)),
    ('generate', 'Generate bounded text using an explicitly selected inference connection and model. Remote inference may incur charges.', {'connection_id': _string(128), 'model': _string(256), 'prompt': _string(65536), 'max_tokens': _integer(128, 2048)}, ('connection_id', 'model', 'prompt')),
    ('profile', 'Inspect the AI Lab data directories bound to this Hermes profile without reading credentials.', {}, ()),
    ('onboarding', 'Check local setup or initialize this profile workspace and state. Model/data roots and executable bindings are configured through the operator setup CLI. Does not download, train, provision or activate models.', {
        'request': {'type': 'object', 'additionalProperties': False, 'properties': {
            'action': {'type': 'string', 'enum': ['check', 'setup']},
        }, 'required': ['action']},
    }, ('request',)),
    ('mission', 'Prepare a declared local comparison or automatically plan and run compatible trials from a task, objective, local model and dataset. autoplan previews without model inference and reconciles local lessons; autostart queues bounded local MLX evaluation; autoreport learns context-specific lessons from a completed local job; lessons reads that memory. Missing quality criteria keep shorthand runs in development. Never downloads, trains, activates or provisions resources.', {
        'request': {'type': 'object', 'additionalProperties': False, 'properties': {
            'action': {'type': 'string', 'enum': ['prepare', 'autoplan', 'autostart', 'autoreport', 'lessons']}, 'mission': request_schema(),
            'objective': {'type': 'string', 'enum': ['quality', 'latency', 'memory', 'throughput']},
            'task': {'anyOf': [_string(2000), {'type': 'object', 'additionalProperties': False,
                'properties': {'description': _string(2000),
                    'type': {'type': 'string', 'enum': ['classification', 'extraction', 'generation']},
                    'language': _string(64)}, 'required': ['description', 'type']}]},
            'model': _string(), 'job_id': _string(64),
            'output_choices': {'type': 'array', 'items': _JSON_OBJECT, 'minItems': 2, 'maxItems': 32},
            'limit': _integer(20, 100),
            'dataset_dir': _string(), 'output_dir': _string(), 'baseline': _BASELINE,
            'grader': {'type': 'string', 'enum': ['json_exact', 'exact_text']},
            'repetitions': {'type': 'integer', 'minimum': 2, 'maximum': 5},
            'minimum_relative_gain': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'quality_noninferiority_tolerance': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'max_generated_tokens': {'type': 'integer', 'minimum': 1, 'maximum': 65536},
            'minimum_slice_quality': _JSON_OBJECT,
            'prompt_candidates': {'type': 'array', 'items': {'type': 'string', 'maxLength': 16384}, 'maxItems': 7},
            'model_candidates': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'model': _string(), 'adapter': _string()}, 'required': ['model'], 'additionalProperties': False}, 'maxItems': 7},
            'adapter_candidates': {'type': 'array', 'items': {'anyOf': [_string(), {'type': 'null'}]}, 'maxItems': 7},
            'prefill_candidates': {'type': 'array', 'items': {'type': 'integer', 'enum': [64, 256, 2048]}, 'maxItems': 3},
            'evaluation_scope': {'type': 'string', 'enum': ['full', 'development']},
        }, 'required': ['action']},
    }, ('request',)),
    ('registry', 'Capture a locally qualified configuration from a completed job; inspect, export or import a portable package; explicitly activate, generate with, or roll back a configuration in this profile. Imports remain inactive and require local qualification. Activation rechecks evidence and asset identities.', {
        'request': {'type': 'object', 'additionalProperties': False, 'properties': {
            'action': {'type': 'string', 'enum': ['capture', 'list', 'inspect', 'export', 'import', 'activate', 'rollback', 'status', 'generate', 'result']},
            'job_id': _string(64), 'entry_id': _string(128), 'output_path': _string(),
            'package_path': _string(), 'model': _string(), 'input': _string(8192), 'output_dir': _string(),
        }, 'required': ['action']},
    }, ('request',)),
    ('analyze', 'Analyze the task, objective, workload and local environment to propose bounded optimization trials. Read-only inventory; no network probe, model load, training or provisioning. An empty request returns the missing context.', {
        'request': request_schema(),
    }, ('request',)),
    ('benchmark', 'Start a bounded local MLX inference measurement job or read its report. Measures one request at a time; does not evaluate task quality or claim an optimization gain.', {
        'request': {'type': 'object', 'additionalProperties': False, 'properties': {
            'action': {'type': 'string', 'enum': ['start', 'report']},
            'model': _string(), 'adapter': _string(), 'output_dir': _string(), 'report_path': _string(),
            'prompts': {'type': 'array', 'items': {**_string(8192), 'description': 'At most 8192 UTF-8 bytes and 2048 tokens after formatting.'}, 'minItems': 1, 'maxItems': 8},
            'prompt_format': {'type': 'string', 'enum': ['chat', 'raw'], 'default': 'chat'},
            'max_tokens': _integer(64, 256), 'repetitions': _integer(3, 10, 2),
        }, 'required': ['action']},
    }, ('request',)),
    ('optimize', 'Compare local MLX configurations under a frozen quality, memory and performance contract, with optional finite JSON-constrained decoding. Select on development data, or predeclare a fixed baseline/candidate pair for held-out comparison. Qualification thresholds still apply. Returns a durable job or report; never activates a model.', {
        'request': {'type': 'object', 'additionalProperties': False, 'properties': {
            'action': {'type': 'string', 'enum': ['start', 'report']},
            'protocol_path': _string(), 'dataset_dir': _string(),
            'output_dir': _string(), 'report_path': _string(),
        }, 'required': ['action']},
    }, ('request',)),
    ('experiment', 'Start a bounded local MLX learning experiment, evaluate a frozen candidate, or read its report. Start and evaluate return a durable job; no cloud calls.', {
        'action': {'type': 'string', 'enum': ['start', 'evaluate', 'train_lora', 'report']},
        'model': _string(), 'dataset_dir': _string(), 'data_dir': _string(), 'output_dir': _string(),
        'adapter': _string(), 'prompt_file': _string(), 'report_path': _string(),
    }, ('action',)),
)
_SCHEMAS = {
    'ai_lab_' + name: {'name': 'ai_lab_' + name, 'description': description,
                      'parameters': {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}}
    for name, description, properties, required in _DEFS
}


def schemas():
    return list(copy.deepcopy(_SCHEMAS).values())


def _bounded_json(value, depth=0):
    if depth > 16:
        return False
    if type(value) is str:
        return len(value) <= MAX_JSON_BYTES
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        if len(value) > 65536:
            return False
        return all(_bounded_json(item, depth + 1) for item in value)
    if type(value) is dict:
        if len(value) > 65536:
            return False
        return all(type(key) is str and _bounded_json(item, depth + 1) for key, item in value.items())
    return False


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _matches(value, spec):
    kind = spec['type']
    if kind == 'string':
        return type(value) is str and spec.get('minLength', 0) <= len(value) <= spec.get('maxLength', MAX_JSON_BYTES) and '\x00' not in value and ('enum' not in spec or value in spec['enum'])
    if kind == 'integer':
        return type(value) is int and spec['minimum'] <= value <= spec['maximum']
    if kind == 'boolean':
        return type(value) is bool
    return kind == 'object' and type(value) is dict


def dispatch(name, arguments):
    """Return a safe JSON object; exceptions never expose provider details."""
    if type(name) is not str or name not in _SCHEMAS:
        return failure('unknown_tool', 'The requested AI Lab tool does not exist.')
    if type(arguments) is not dict or not _bounded_json(arguments):
        return failure()
    try:
        if len(encode(arguments).encode('utf-8')) > MAX_JSON_BYTES:
            return failure('request_too_large', 'Tool arguments exceed the size limit.')
        spec = _SCHEMAS[name]['parameters']
        if set(arguments) - set(spec['properties']) or not set(spec['required']).issubset(arguments):
            return failure()
        if not all(_matches(value, spec['properties'][key]) for key, value in arguments.items()):
            return failure()
        if name == 'ai_lab_job':
            action = arguments.get('action', 'list')
            if action not in ('list', 'reconcile') and not arguments.get('job_id'):
                return failure()
            if action == 'list' and arguments.get('limit', 20) > 200:
                return failure()
        service = importlib.import_module('.service', package=__package__)
        result = getattr(service, name[len('ai_lab_'):])(**arguments)
        if type(result) is not dict or type(result.get('ok')) is not bool or not _bounded_json(result):
            return failure('service_error', 'The AI Lab service returned an invalid result.')
        if len(encode(result).encode('utf-8')) > MAX_JSON_BYTES:
            return failure('response_too_large', 'The AI Lab response exceeds the size limit.')
        return result
    except Exception:
        return failure('service_error', 'The AI Lab service could not complete the request.')
