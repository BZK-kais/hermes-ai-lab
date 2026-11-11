#!/usr/bin/env python3
"""Build a portable AI Lab SSH runner. Does not connect or install on any host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile

_SETTINGS_KEYS = {"workspace_root", "model_roots", "data_roots", "runtime_bin_dirs", "backend_paths", "backend_recipe_args"}


def validate_admin_settings(workspace: str, settings: dict | None = None) -> dict:
    """Administrative build-time configuration, never taken from job parameters."""
    if not isinstance(workspace, str) or not workspace.startswith('/') or not re.fullmatch(r'/[A-Za-z0-9_./-]+', workspace) or any(p in ('', '.', '..') for p in workspace[1:].split('/')):
        raise ValueError('Use a safe absolute remote workspace.')
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - _SETTINGS_KEYS:
        raise ValueError('Unknown administrator settings.')
    if 'workspace_root' in settings and settings['workspace_root'] != workspace:
        raise ValueError('Administrator workspace must match the requested workspace.')
    result = {"workspace_root": workspace, "model_roots": [], "data_roots": [], "runtime_bin_dirs": [],
              "backend_paths": {}, "backend_recipe_args": {}} | settings
    for key in ('model_roots', 'data_roots', 'runtime_bin_dirs'):
        values = result[key]
        if not isinstance(values, list) or len(values) > 32 or any(not isinstance(p, str) or not p.startswith('/') or len(p) > 1024 or any(ord(c) < 32 for c in p) for p in values):
            raise ValueError('Invalid administrator directory list.')
    paths = result['backend_paths']
    if not isinstance(paths, dict) or len(paths) > 32 or any(not isinstance(k, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', k) or not isinstance(v, str) or not v.startswith('/') or len(v) > 1024 or any(ord(c) < 32 for c in v) for k, v in paths.items()):
        raise ValueError('Invalid administrator backend paths.')
    recipes = result['backend_recipe_args']
    if not isinstance(recipes, dict) or len(recipes) > 16:
        raise ValueError('Invalid administrator recipe bindings.')
    for operation, args in recipes.items():
        if operation not in {'nemo.bridge.train', 'nemo.rl.train'} or not isinstance(args, list) or not 1 <= len(args) <= 64:
            raise ValueError('Invalid administrator recipe binding.')
        if any(not isinstance(a, str) or len(a) > 1024 or any(ord(c) < 32 for c in a) or re.search(r'token|password|secret|api[_-]?key', a, re.I) for a in args):
            raise ValueError('Recipe bindings cannot carry credentials.')
        if not any('{config}' in a for a in args) or not any('{output_dir}' in a for a in args):
            raise ValueError('Recipe binding requires config and output placeholders.')
    return result


_RUNNER = r'''
"""Generated AI Lab runner: fixed administrator configuration, JSON plans only."""
import argparse,contextlib,hashlib,json,os,pathlib,signal,subprocess,sys,zipfile


def fail(code):
    return {'ok':False,'error':{'code':code,'message':'Remote worker refused or could not complete the plan.'}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    result_path=None
    try:
        archive_path=pathlib.Path(sys.argv[0]).resolve(strict=True)
        with zipfile.ZipFile(archive_path) as archive:
            trusted=json.loads(archive.read('worker-settings.json'))
            workspace=pathlib.Path(trusted['workspace_root']).resolve(strict=True)
            plan_path=pathlib.Path(args.plan).resolve(strict=True)
            if not plan_path.is_relative_to(workspace) or not plan_path.is_file() or plan_path.stat().st_size>65536:
                raise ValueError()
            request=json.loads(plan_path.read_text())
            if not isinstance(request,dict) or set(request)-{'operation','parameters','settings'}:
                raise ValueError()
            supplied=request.get('settings',{})
            if not isinstance(supplied,dict) or any(k not in trusted or v!=trusted[k] for k,v in supplied.items()):
                return fail('administrator_settings_required'),2
            if not isinstance(request.get('parameters',{}),dict) or not isinstance(request.get('operation'),str):
                raise ValueError()
            # Extract only code into this workspace: subprocess backend paths must be real files.
            digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()[:24]
            runtime=workspace/'.ai-lab-worker'/'runtime'/digest
            if not runtime.resolve().is_relative_to(workspace):
                raise ValueError()
            runtime.mkdir(parents=True,exist_ok=True)
            for info in archive.infolist():
                name=pathlib.PurePosixPath(info.filename)
                if not info.filename.startswith('ai_lab_core/'):
                    continue
                if '..' in name.parts or name.suffix!='.py' or info.file_size>2097152:
                    raise ValueError()
                destination=runtime.joinpath(*name.parts)
                if not destination.resolve().is_relative_to(runtime.resolve()):
                    raise ValueError()
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    with destination.open('xb') as target:
                        target.write(archive.read(info))
        sys.path.insert(0,str(runtime))
        if request['operation']=='probe':
            from ai_lab_core.hardware_profiles import detect_profile
            with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
                result=detect_profile()
            return result,0 if result.get('ok') else 1
        from ai_lab_core.backends import plan_operation
        from ai_lab_core.process_env import runtime_env
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
            planned=plan_operation(request['operation'],request.get('parameters',{}),trusted)
        if not planned.get('ok'):
            return {'ok':False,'error':{'code':planned.get('error',{}).get('code','planning_failed'),'message':'Remote backend planning failed.'}},2
        plan=planned['plan']
        result_path=plan_path.with_name('result.json')
        if result_path.exists() or result_path.is_symlink():
            return fail('reconciliation_required'),2
        # Planner constructs argv and env from allowlisted operations and trusted bindings.
        with open(os.devnull,'wb') as sink:
            proc=subprocess.Popen(plan['command'],cwd=plan['cwd'],env=runtime_env(plan.get('env',{})),
                stdin=subprocess.DEVNULL,stdout=sink,stderr=sink,start_new_session=True)
            def terminate_owned(signum,frame):
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
                raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,terminate_owned)
            signal.signal(signal.SIGINT,terminate_owned)
            try:
                code=proc.wait(timeout=plan['timeout_seconds'])
                timed_out=False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                code=proc.wait()
                timed_out=True
        outputs=all((pathlib.Path(o['path']).is_file() if o['kind']=='file' else pathlib.Path(o['path']).is_dir())
            and not pathlib.Path(o['path']).is_symlink() for o in plan.get('outputs',[]))
        result={'ok':code==0 and outputs and not timed_out,'exit_code':code,'outputs_verified':outputs,
                'reason':'timeout' if timed_out else 'exit_nonzero' if code else 'outputs_missing' if not outputs else 'completed'}
        with result_path.open('x') as handle:
            json.dump(result,handle)
        return result,0 if result['ok'] else 1
    except Exception:
        return fail('worker_error'),1

if __name__=='__main__':
    outcome,code=main()
    print(json.dumps(outcome,allow_nan=False))
    raise SystemExit(code)
'''


def build_worker(output: str | Path, workspace: str, admin_settings: dict | None = None) -> Path:
    settings = validate_admin_settings(workspace, admin_settings)
    target = Path(output)
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise ValueError('Output must be a new file in an existing directory.')
    source = Path(__file__).resolve().parents[1] / 'ai_lab_core'
    files = sorted(source.rglob('*.py'))
    if not files or any(path.is_symlink() or path.stat().st_size > 2097152 for path in files):
        raise ValueError('Invalid package source.')
    with target.open('xb') as handle, zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path('ai_lab_core') / path.relative_to(source)))
        archive.writestr('__main__.py', _RUNNER)
        archive.writestr('worker-settings.json', json.dumps(settings))
    return target.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='Existing absolute workspace on the remote machine')
    parser.add_argument('--output', required=True, help='New local zipapp file, conventionally remote_worker.py')
    parser.add_argument('--admin-config', help='Optional non-secret administrator settings JSON')
    args = parser.parse_args()
    try:
        settings = None
        if args.admin_config:
            path = Path(args.admin_config)
            if path.stat().st_size > 32768:
                raise ValueError('Administrator settings too large.')
            settings = json.loads(path.read_text())
        build_worker(args.output, args.workspace, settings)
    except Exception:
        print(json.dumps({'ok':False,'error':{'code':'build_failed','message':'Remote worker build failed; check non-secret settings and output path.'}}))
        return 1
    print(json.dumps({'ok':True,'built':True,'configured_remote':False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
