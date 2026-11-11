"""CLI planning adapters. Planning performs no download, conversion or training."""
from __future__ import annotations

from .process_env import runtime_env

import os
import re
import shutil
import sys
import subprocess
from pathlib import Path

# Schemas are also the allowlist: callers cannot append arbitrary CLI arguments.
def _schema(required=(), **properties):
    return {'type': 'object', 'properties': properties, 'required': list(required),
            'additionalProperties': False}

S = {'type': 'string', 'minLength': 1}
P = {'type': 'string', 'format': 'absolute-path'}
I = {'type': 'integer', 'minimum': 1, 'maximum': 1000000}
O = {'output_dir': P}
_MLX_MIXED_BITS = {'mixed_2_6': (2, 6), 'mixed_3_4': (3, 4), 'mixed_3_6': (3, 6), 'mixed_4_6': (4, 6)}
_MLX_AFFINE_GROUP_SIZE = 64
OPS = {
    'hf.download': ('hf', _schema(('repo_id', 'output_dir'), repo_id=S, revision=S, allow_patterns={'type':'array','items':S,'minItems':1,'maxItems':32}, **O)),
    'mlx.convert': ('mlx_lm.convert', _schema(('model', 'output_dir'), model=P,
        bits={'type':'integer','enum':[2,3,4,6,8], 'description':'Uniform affine quantization. Mutually exclusive with quant_predicate.'},
        quant_predicate={'type':'string','enum':list(_MLX_MIXED_BITS), 'description':'Native MLX-LM mixed-bit affine recipe. Requires a compatible model.'},
        group_size={'type':'integer','enum':[32,64,128], 'description':'Affine group size. Requires bits or quant_predicate. Defaults to 64.'},
        **O)),
    'mlx.train': ('mlx_lm.lora', _schema(('model','data','output_dir'), model=P, data=P, iters=I,
                                       batch_size=I, steps_per_report=I, steps_per_eval=I, save_every=I, **O)),
    'mlx.fuse': ('mlx_lm.fuse', _schema(('model','adapter','output_dir'), model=P, adapter=P, **O)),
    'mlx.generate': ('mlx_lm.generate', _schema(('model','prompt'), model=P,
                      prompt={'type':'string','format':'text','minLength':1},
                      max_tokens={'type':'integer','minimum':1,'maximum':2048}, adapter=P)),
    'hermes.export': ('hermes', _schema(('session_id','output_file'),
                       session_id={'type':'string','pattern':r'[A-Za-z0-9][A-Za-z0-9_-]{5,127}'},output_file=P)),
    'mlx.serve': ('mlx_lm.server', _schema(('model',), model=P, port={'type':'integer','minimum':1024,'maximum':65535})),
    'llama.convert': ('llama.convert', _schema(('model','output_file'), model=P, output_file=P,
                           dtype={'type':'string','enum':['f16','bf16','f32']})),
    'llama.quantize': ('llama-quantize', _schema(('model','output_file','quant'), model=P, output_file=P,
                         quant={'type':'string','enum':['Q4_K_M','Q5_K_M','Q6_K','Q8_0','Q4_0','Q5_0','IQ4_XS']}, imatrix=P)),
    'llama.imatrix': ('llama-imatrix', _schema(('model','data','output_file'), model=P, data=P, output_file=P, chunks=I)),
    'llama.serve': ('llama-server', _schema(('model',), model=P, port={'type':'integer','minimum':1024,'maximum':65535})),
    'unsloth.train': ('unsloth', _schema(('config','output_dir'), config=P, **O)),
    'trl.train': ('trl.train', _schema(('config','output_dir'), config=P, **O)),
    'llamafactory.train': ('llamafactory-cli', _schema(('config','output_dir'), config=P, **O)),
    'llamafactory.export': ('llamafactory-cli', _schema(('config','output_dir'), config=P, **O)),
    'nemo.automodel.train': ('automodel', _schema(('config','output_dir'), config=P, **O)),
    'nemo.bridge.train': ('nemo.bridge.train', _schema(('config','output_dir'), config=P, **O)),
    'nemo.rl.train': ('nemo.rl.train', _schema(('config','output_dir'), config=P, **O)),
    'lm_eval.evaluate': ('lm_eval', _schema(('model','tasks','output_dir'), model=P,
                        tasks={'type':'array','items':S,'minItems':1,'maxItems':32}, limit=I, **O)),
}
RECIPE_OPS = {'nemo.bridge.train', 'nemo.rl.train'}


def _error(code, message):
    return {'ok': False, 'error': {'code': code, 'message': message}}


def _tool(name, settings):
    configured = settings.get('backend_paths', {}).get(name)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute() or not candidate.is_file():
            return None
        if candidate.suffix == '.py' or os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return None
    if name == 'trl.train':
        return str(Path(__file__).parent / 'recipes' / 'trl_sft.py')
    if name in {'llama.convert'} | RECIPE_OPS:
        return None
    mlx_python = settings.get('backend_paths', {}).get('mlx.python')
    if name.startswith('mlx_lm.') and mlx_python:
        interpreter = Path(mlx_python)
        candidate = interpreter.parent / name
        if (interpreter.is_absolute() and interpreter.is_file() and os.access(interpreter, os.X_OK)
                and candidate.is_file() and os.access(candidate, os.X_OK)):
            return str(candidate)
    for value in [str(Path(d).expanduser() / name) for d in settings.get('runtime_bin_dirs', [])[:32]] + [
            shutil.which(name), str(Path('/opt/homebrew/bin') / name)]:
        if value and Path(value).is_file() and os.access(value, os.X_OK):
            return str(Path(value).resolve())
    return None


def _recipe_args(operation, settings):
    args = settings.get('backend_recipe_args', {}).get(operation)
    if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
        return None
    # Templates are installed by the operator, never supplied as operation params.
    if not any('{config}' in a for a in args) or not any('{output_dir}' in a for a in args):
        return None
    for value in args:
        if re.search(r'\{(?!config\}|output_dir\})', value) or '\x00' in value:
            return None
    return args


def _python_has(python, modules):
    try:
        result = subprocess.run([python, '-I', '-c',
            'import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(x) is not None for x in sys.argv[1:]) else 1)',
            *modules], capture_output=True, timeout=3, env=runtime_env())
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _requirements(operation, settings):
    tool = OPS[operation][0]
    missing = []
    executable = _tool(tool, settings)
    if not executable:
        missing.append({'code':'configuration_required' if tool in RECIPE_OPS or tool == 'llama.convert'
                        else 'dependency_missing','requirement':tool})
    if operation in RECIPE_OPS and not _recipe_args(operation, settings):
        missing.append({'code':'configuration_required','requirement':'backend_recipe_args.' + operation})
    if operation == 'trl.train':
        python = _tool('python', settings) or sys.executable
        if not _python_has(python,['transformers','trl','datasets']):
            missing.append({'code':'dependency_missing','requirement':'Python packages: transformers, trl, datasets'})
    return missing


def catalog(settings: dict) -> dict:
    operations = []
    for operation, (tool, schema) in OPS.items():
        missing = _requirements(operation, settings)
        operations.append({'operation':operation,'backend':operation.split('.')[0],
                           'params_schema':schema,'available':not missing,'missing_requirements':missing,
                           'requirements':[tool] + (['backend_recipe_args.' + operation] if operation in RECIPE_OPS else []),
                           'status':'ready_to_plan' if not missing else 'unavailable',
                           'execution_verified':False})
    return {'ok':True,'operations':operations}


class _Invalid(Exception):
    pass


def _validate(params, schema):
    if not isinstance(params, dict) or set(params) - set(schema['properties']):
        raise _Invalid()
    if set(schema['required']) - set(params):
        raise _Invalid()
    for key, value in params.items():
        spec = schema['properties'][key]
        kind = spec['type']
        if kind == 'string' and (not isinstance(value, str) or not value or len(value) > 4096 or
                                  any(ord(c) < 32 and not (spec.get('format') == 'text' and c in '\n\r\t') for c in value)):
            raise _Invalid()
        if kind == 'integer' and (isinstance(value, bool) or not isinstance(value, int) or
                not spec.get('minimum', 1) <= value <= spec.get('maximum', 1000000)):
            raise _Invalid()
        if 'pattern' in spec and not re.fullmatch(spec['pattern'],value):
            raise _Invalid()
        if 'enum' in spec and value not in spec['enum']:
            raise _Invalid()
        if kind == 'array' and (not isinstance(value, list) or not 1 <= len(value) <= 32 or
                any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_*?.\[\]/-]{1,200}' if key == 'allow_patterns' else r'[A-Za-z0-9_./-]{1,100}', v) for v in value)):
            raise _Invalid()


def plan_operation(operation: str, params: dict, settings: dict) -> dict:
    if not isinstance(operation,str) or operation not in OPS:
        return _error('unknown_operation', 'Unknown backend operation.')
    tool, schema = OPS[operation]
    try:
        _validate(params, schema)
        if operation == 'mlx.convert':
            if 'bits' in params and 'quant_predicate' in params:
                return _error('invalid_parameters', 'Choose bits or quant_predicate for MLX quantization.')
            if 'group_size' in params and not {'bits', 'quant_predicate'} & params.keys():
                return _error('invalid_parameters', 'MLX group_size requires bits or quant_predicate.')
        workspace = Path(settings.get('workspace_root', ''))
        if not workspace.is_absolute() or not workspace.is_dir():
            return _error('configuration_required', 'An existing absolute workspace root is required.')
        workspace = workspace.resolve()
        roots = [workspace]
        for key in ('model_roots','data_roots'):
            roots += [Path(p).expanduser().resolve() for p in settings.get(key, []) if Path(p).expanduser().is_absolute()]
        paths = {}
        for key in ('model','data','adapter','config','imatrix'):
            if key not in params:
                continue
            path = Path(params[key])
            if not path.is_absolute():
                raise _Invalid()
            path = path.resolve(strict=True)
            if not any(path.is_relative_to(root) for root in roots):
                return _error('path_not_allowed', 'Input path is outside configured roots.')
            if key in ('config','imatrix') and not path.is_file():
                raise _Invalid()
            paths[key] = str(path)
        directory_model = operation.startswith('mlx.') or operation in {'llama.convert','lm_eval.evaluate'}
        if 'model' in paths and (not Path(paths['model']).is_dir() if directory_model else
                                 not Path(paths['model']).is_file()):
            raise _Invalid()
        if operation == 'mlx.train' and not Path(paths['data']).is_dir():
            raise _Invalid()
        if operation == 'llama.imatrix' and not Path(paths['data']).is_file():
            raise _Invalid()
        if 'adapter' in paths and not Path(paths['adapter']).is_dir():
            raise _Invalid()
        outputs = []
        for key, kind in (('output_dir','directory'),('output_file','file')):
            if key not in params:
                continue
            path = Path(params[key])
            if not path.is_absolute():
                raise _Invalid()
            if path.exists() or path.is_symlink():
                return _error('output_exists', 'Output paths must not already exist.')
            path = path.resolve()
            if path == workspace or not path.is_relative_to(workspace):
                return _error('path_not_allowed', 'Outputs must be inside the workspace root.')
            if not path.parent.is_dir():
                return _error('output_parent_missing', 'The output parent directory must exist.')
            if any(path == Path(p) or path.is_relative_to(Path(p)) or Path(p).is_relative_to(path)
                   for p in paths.values()):
                return _error('source_output_overlap', 'Inputs and outputs must be separate paths.')
            paths[key] = str(path)
            outputs.append({'path':str(path),'kind':kind})
        if operation == 'hf.download':
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?', params['repo_id']):
                raise _Invalid()
            if 'revision' in params and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,199}',params['revision']):
                raise _Invalid()
        if 'allow_patterns' in params and any(p.startswith(('/','-')) or any(part in ('','..') for part in p.split('/')) for p in params['allow_patterns']):
            raise _Invalid()
        missing = _requirements(operation, settings)
        if missing:
            return _error(missing[0]['code'], 'Backend dependencies or recipe bindings are unavailable.')
        executable = _tool(tool, settings)
        if not executable:
            return _error('configuration_required' if tool in RECIPE_OPS or tool == 'llama.convert'
                          else 'dependency_missing', 'The required backend executable or script is unavailable.')
        python = _tool('python', settings) or sys.executable
        command = [python, executable] if executable.endswith('.py') else [executable]
        metadata = {'generation_verified':False,'kernel_verified':False,'downloads_model':operation == 'hf.download'}
        out = paths.get('output_dir',paths.get('output_file'))
        if operation == 'trl.train':
            from .recipes.trl_sft import load_config
            try:
                recipe_config = load_config(paths['config'], [str(root) for root in roots])
                for key in ('model','train_file','eval_file'):
                    if key in recipe_config:
                        source = Path(recipe_config[key])
                        if Path(out).is_relative_to(source) or source.is_relative_to(Path(out)):
                            return _error('source_output_overlap','Inputs and outputs must be separate paths.')
                if recipe_config.get('lora') and not _python_has(python,['peft']):
                    return _error('dependency_missing','LoRA training requires PEFT.')
            except ImportError:
                return _error('dependency_missing','YAML recipe validation requires PyYAML.')
            command += ['--config',paths['config'],'--output-dir',out,'--workspace-root',str(workspace)]
            for root in roots:
                command += ['--allowed-root',str(root)]
        elif operation in RECIPE_OPS:
            args = _recipe_args(operation,settings)
            if args is None:
                return _error('configuration_required','An exact recipe argument template is required.')
            command += [a.replace('{config}', paths['config']).replace('{output_dir}',out) for a in args]
            metadata['recipe_arguments_operator_configured'] = True
        elif operation == 'hf.download':
            candidate_python = Path(executable).parent / 'python'
            sdk_python = str(candidate_python) if candidate_python.is_file() else python
            command = [sdk_python,str(Path(__file__).parent / 'backend_task.py'),'hf-download',
                       '--repo-id',params['repo_id'],'--output-dir',out]
            if 'revision' in params:
                command += ['--revision',params['revision']]
            for pattern in params.get('allow_patterns',[]):
                command += ['--allow-pattern',pattern]
            metadata['authentication'] = 'anonymous_token_false'
        elif operation == 'hermes.export':
            command += ['sessions','export',out,'--session-id',params['session_id'],'--format','jsonl','--redact']
            metadata['redaction_requested'] = True
        elif operation == 'mlx.generate':
            command += ['--model',paths['model'],'--prompt=' + params['prompt'],
                        '--max-tokens',str(params.get('max_tokens',128)),'--temp','0']
            if 'adapter' in paths:
                command += ['--adapter-path',paths['adapter']]
        elif operation == 'mlx.convert':
            command += ['--hf-path',paths['model'],'--mlx-path',out]
            if {'bits', 'quant_predicate'} & params.keys():
                group_size = params.get('group_size', _MLX_AFFINE_GROUP_SIZE)
                quantization = {'mode':'affine', 'group_size':group_size}
                if 'quant_predicate' in params:
                    preset = params['quant_predicate']
                    # MLX-LM passes an omitted group size as None to the preset builder.
                    command += ['-q', '--quant-predicate', preset, '--q-group-size', str(group_size)]
                    quantization.update(quant_predicate=preset, bits=list(_MLX_MIXED_BITS[preset]))
                else:
                    command += ['-q', '--q-bits', str(params['bits'])]
                    if 'group_size' in params:
                        command += ['--q-group-size', str(group_size)]
                    quantization['bits'] = [params['bits']]
                metadata.update(quantization=quantization, quantizes_output_weights=True, quality_verified=False)
        elif operation == 'mlx.train':
            command += ['--model',paths['model'],'--train','--data',paths['data'],'--adapter-path',out,
                        '--iters',str(params.get('iters',100)), '--batch-size',str(params.get('batch_size',1))]
            for key in ('steps_per_report', 'steps_per_eval', 'save_every'):
                if key in params:
                    command += ['--' + key.replace('_', '-'), str(params[key])]
            outputs.extend({'path':str(Path(out) / name), 'kind':'file'}
                           for name in ('adapter_config.json', 'adapters.safetensors'))
        elif operation == 'mlx.fuse':
            command += ['--model',paths['model'],'--adapter-path',paths['adapter'],'--save-path',out]
        elif operation in {'mlx.serve','llama.serve'}:
            command += ['--model',paths['model'],'--host','127.0.0.1','--port',str(params.get('port',8080))]
            metadata['endpoint'] = 'http://127.0.0.1:' + str(params.get('port',8080))
        elif operation == 'llama.convert':
            command += [paths['model'],'--outfile',out,'--outtype',params.get('dtype','f16')]
        elif operation == 'llama.quantize':
            if 'imatrix' in paths:
                command += ['--imatrix',paths['imatrix']]
            command += [paths['model'],out,params['quant']]
        elif operation == 'llama.imatrix':
            command += ['-m',paths['model'],'-f',paths['data'],'-o',out,'--chunks',str(params.get('chunks',10))]
        elif operation == 'unsloth.train':
            command += ['train','--config',paths['config'],'--output-dir',out]
        elif operation.startswith('llamafactory.'):
            verb = operation.split('.')[1]
            command += [verb,paths['config'],('output_dir=' if verb == 'train' else 'export_dir=') + out]
        elif operation == 'nemo.automodel.train':
            if executable.endswith('.py'):
                command += ['--config',paths['config']]
            else:
                command += [paths['config']]
            command += ['--checkpoint.checkpoint_dir',out,'--wandb.enable','false']
        elif operation == 'lm_eval.evaluate':
            if any(char in paths['model'] for char in (',','=')):
                raise _Invalid()
            command += ['--model','hf','--model_args','pretrained=' + paths['model'] + ',trust_remote_code=False',
                        '--tasks',','.join(params['tasks']),'--output_path',out]
            if 'limit' in params:
                command += ['--limit',str(params['limit'])]
        return {'ok':True,'plan':{'operation':operation,'backend':operation.split('.')[0],
                'command':command,'cwd':str(workspace),'timeout_seconds':3600 if operation.endswith('.serve') else 86400,
                'outputs':outputs,'metadata':metadata,
                'env':{'HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','WANDB_MODE':'disabled',
                       **({'PYTHONUNBUFFERED':'1'} if operation == 'mlx.train' else {}),
                       **({'HF_HUB_OFFLINE':'0'} if operation == 'hf.download' else {'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})}}}
    except (ValueError, TypeError, OSError, RuntimeError, _Invalid):
        return _error('invalid_parameters','Operation parameters or paths are invalid.')
