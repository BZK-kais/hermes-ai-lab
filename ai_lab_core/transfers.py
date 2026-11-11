"""Explicit storage transfer plans. Discovery never moves artifact bytes."""
from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess
import sys
from .credentials import CredentialError
from .process_env import runtime_env
from .providers import _AdapterError, _sdk_call, validate_connection

HARD_MAX_BYTES = 50 * 1024**3
MODULES = {'hf_storage':'huggingface_hub','s3':'boto3'}
STORAGE_TYPES = {*MODULES, 'gcs'}
GCS_TRANSFER_UNSUPPORTED = 'Hermes Google sign-in supports bounded metadata requests in AI Lab.'
SCHEMA = {'type':'object','properties':{
    'connection_id':{'type':'string','pattern':r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}'},
    'key':{'type':'string','description':'Relative object path; traversal is forbidden.'},
    'local_path':{'type':'string','format':'absolute-path'}},
    'required':['connection_id','key','local_path'],'additionalProperties':False}


def failure(code,message):
    return {'ok':False,'error':{'code':code,'message':message}}


def valid_key(value):
    return (isinstance(value,str) and 0 < len(value) <= 1024 and not value.startswith(('/','-'))
            and '\\' not in value and all(32 <= ord(c) < 127 for c in value)
            and all(part not in ('','.','..') for part in value.split('/')))


def max_bytes(settings):
    limit = settings.get('transfer_max_bytes',HARD_MAX_BYTES)
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1 <= limit <= HARD_MAX_BYTES:
        raise ValueError()
    return limit


def local_file(value,workspace,download):
    if not isinstance(value,str) or not value or any(ord(c)<32 for c in value):
        raise ValueError()
    raw = Path(value)
    root = Path(workspace)
    if not raw.is_absolute() or not root.is_absolute():
        raise ValueError()
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError()
    if raw.is_symlink():
        raise ValueError()
    path = raw.resolve(strict=not download)
    if path == root or not path.is_relative_to(root) or not path.parent.is_dir():
        raise ValueError()
    if download and path.exists():
        raise FileExistsError()
    if not download and not path.is_file():
        raise ValueError()
    return path,root


def _python(c,settings):
    value = c.get('python',settings.get('backend_paths',{}).get('python',sys.executable))
    path = Path(value)
    return str(path.resolve()) if path.is_absolute() and path.is_file() and os.access(path,os.X_OK) else None


def _installed(python,module):
    if not python:
        return False
    try:
        result = subprocess.run([python,'-I','-c',
            'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)',module],
            capture_output=True,timeout=3,env=runtime_env())
        return result.returncode == 0
    except (OSError,subprocess.TimeoutExpired):
        return False


def catalog(settings:dict)->dict:
    refs = []
    for c in settings.get('integrations',[])[:32]:
        if not isinstance(c,dict) or c.get('type') not in STORAGE_TYPES:
            continue
        validation = validate_connection(c)
        python = _python(c,settings) if validation.get('ok') else None
        missing = []
        if not validation.get('ok'):
            missing.append(validation['error'])
        elif c['type'] == 'gcs':
            missing.append({'code':'unsupported_transport','message':GCS_TRANSFER_UNSUPPORTED})
        elif not _installed(python,MODULES[c['type']]):
            missing.append({'code':'dependency_missing','requirement':MODULES[c['type']]})
        refs.append({'connection_id':c.get('id'),'type':c.get('type'),'available':not missing,
                     'missing_requirements':missing,'credential_reference_configured':bool(c.get('credential_ref')),
                     'credential_validated':False})
    return {'ok':True,'operations':[{'operation':op,'backend':'storage','params_schema':SCHEMA,
             'connections':refs,'execution_verified':False} for op in ('storage.download','storage.upload')]}


def _prepare_transfer(operation:str,params:dict,settings:dict)->dict:
    if operation not in ('storage.download','storage.upload'):
        return failure('unknown_operation','Unknown storage transfer operation.')
    if not isinstance(params,dict) or set(params) != set(SCHEMA['required']):
        return failure('invalid_parameters','Transfer parameters are invalid.')
    if not isinstance(params['connection_id'],str) or not re.fullmatch(SCHEMA['properties']['connection_id']['pattern'],params['connection_id']) or not valid_key(params['key']):
        return failure('invalid_parameters','Transfer parameters are invalid.')
    matches = [c for c in settings.get('integrations',[]) if isinstance(c,dict) and c.get('id')==params['connection_id']]
    if len(matches)!=1 or matches[0].get('type') not in STORAGE_TYPES:
        return failure('configuration_required','Configure one matching storage connection.')
    c = matches[0]
    validated = validate_connection(c)
    if not validated.get('ok'):
        return validated
    if c['type'] == 'gcs':
        return failure('unsupported_transport',GCS_TRANSFER_UNSUPPORTED)
    download = operation == 'storage.download'
    try:
        cap = max_bytes(settings)
        path,root = local_file(params['local_path'],settings.get('workspace_root',''),download)
        if not download and path.stat().st_size > cap:
            return failure('size_limit','Upload exceeds the configured transfer size limit.')
    except FileExistsError:
        return failure('output_exists','Download output must not exist.')
    except (ValueError,TypeError,OSError,RuntimeError):
        return failure('path_not_allowed','Transfer path or size configuration is invalid.')
    python = _python(c,settings)
    if not _installed(python,MODULES[c['type']]):
        return failure('dependency_missing','The selected Python needs the optional storage SDK.')
    return {'ok':True,
        'connection':{**c, 'python':python},
        'request':{'key':params['key'],'local_path':str(path),'workspace_root':str(root),'max_bytes':cap},
        'plan':{'operation':operation,'backend':'storage','cwd':str(root),
        'timeout_seconds':7200,'outputs':[{'path':str(path),'kind':'file'}] if download else [],
        'metadata':{'connection_id':c['id'],'provider_type':c['type'],'key':params['key'],
                    'max_bytes':cap,'remote_size_checked_before_download':download,
                    'remote_write':not download,'execution_scope':'live_hermes_request'}}}


def plan_operation(operation:str,params:dict,settings:dict)->dict:
    result = _prepare_transfer(operation,params,settings)
    return {'ok':True,'plan':result['plan']} if result['ok'] else result


def run_transfer(operation:str,params:dict,settings:dict)->dict:
    from .jobs import run_live

    result = _prepare_transfer(operation,params,settings)
    if not result['ok']:
        return result
    plan = result['plan']

    def transfer():
        try:
            return _sdk_call(result['connection'],operation,result['request'],timeout=plan['timeout_seconds'])
        except CredentialError:
            return failure('credential_unavailable','The configured credential is unavailable.')
        except subprocess.TimeoutExpired:
            return failure('timeout','Storage transfer timed out. Inspect the destination before retrying.')
        except (OSError, ValueError, _AdapterError):
            return failure('transfer_failed','The storage transfer worker could not complete.')

    return run_live(plan,transfer,state_dir=settings['state_dir'])
