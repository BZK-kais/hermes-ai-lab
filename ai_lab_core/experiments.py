"""Profile-scoped plugin entry points for the frozen local learning experiment."""
import json
from pathlib import Path
import sys

from .endpoint import failure
from .integration_config import permitted_path
from .profile import device_lock_path


def protocol_version(dataset):
    from .skill_experiment import _read_manifest, _verified_path
    manifest = _read_manifest(dataset)
    protocol = json.loads(_verified_path(dataset, manifest, 'protocol.json').read_text())
    version = protocol['version']
    if version not in ('triage-v1', 'pair-paraphrase-v1'):
        raise ValueError('unsupported protocol')
    return version


def dispatch(action, params, settings):
    try:
        if action == 'train_lora':
            from .improvement_training import start
            return start(params, settings)
        if action == 'report':
            if set(params) != {'report_path'}:
                raise ValueError('arguments')
            path = permitted_path(params['report_path'], settings, output=True)
            if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError('report')
            report = json.loads(path.read_text())
            # Detailed examples stay in the operator-selected local report.
            public = {k: report[k] for k in ('ok', 'passed', 'status', 'decoder', 'model_sha256',
                      'adapter_sha256', 'adapter_path', 'prompt_sha256', 'selected_prompt_id',
                      'elapsed_seconds', 'limitations') if k in report}
            for section in ('producer', 'receiver', 'cpu_baseline'):
                if section in report:
                    public[section] = {k: v for k, v in report[section].items()
                                       if k not in ('cases', 'predictions', 'inputs', 'texts')}
            return {'ok': True, 'report_path': str(path), 'summary': public}
        required = {'model', 'dataset_dir', 'output_dir'}
        if action == 'evaluate':
            required |= {'adapter', 'prompt_file'}
        elif action != 'start':
            raise ValueError('action')
        if set(params) != required:
            raise ValueError('arguments')
        workspace = Path(settings['workspace_root']).resolve()
        if not workspace.is_dir():
            raise ValueError('workspace')
        model = permitted_path(params['model'], settings)
        dataset = permitted_path(params['dataset_dir'], settings)
        output = permitted_path(params['output_dir'], settings, output=True, must_exist=False)
        if not model.is_dir() or not dataset.is_dir() or output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise ValueError('paths')
        if any(output == p or output in p.parents or p in output.parents for p in (model, dataset)):
            raise ValueError('overlap')
        version = protocol_version(dataset)
        python = Path(settings.get('backend_paths', {}).get('mlx.python', ''))
        if not python.is_absolute() or not python.is_file():
            return failure('configuration_required', 'Configure the existing MLX Python interpreter as backend_paths.mlx.python.')
        command = [sys.executable, str(Path(__file__).with_name('experiment_task.py')),
                   '--mode', 'train' if action == 'start' else 'evaluate',
                   '--dataset', str(dataset), '--model', str(model), '--output', str(output),
                   '--python', str(python), '--lock', str(device_lock_path())]
        if version == 'pair-paraphrase-v1':
            analysis = Path(settings.get('backend_paths', {}).get('analysis.python', ''))
            if not analysis.is_absolute() or not analysis.is_file():
                return failure('configuration_required', 'Configure the CPU evaluation interpreter as backend_paths.analysis.python.')
            # Only an operator-prepared local baseline is deserialized, never
            # a model package's embedded pickle or an agent-selected path.
            baseline = workspace / 'baselines' / 'pair-paraphrase-v1.json'
            command += ['--analysis-python', str(analysis), '--baseline-freeze', str(baseline)]
        if action == 'evaluate':
            adapter = permitted_path(params['adapter'], settings)
            prompt = permitted_path(params['prompt_file'], settings)
            if not adapter.is_dir() or not prompt.is_file():
                raise ValueError('adapter')
            command += ['--adapter', str(adapter), '--prompt-file', str(prompt)]
        from .jobs import submit
        plan = {'operation': 'skill.experiment' if action == 'start' else 'skill.evaluate',
                'backend': 'mlx', 'command': command, 'cwd': str(workspace),
                'timeout_seconds': 4500 if version == 'pair-paraphrase-v1' else 2400,
                'outputs': [{'path': str(output / 'report.json'), 'kind': 'file'}],
                'metadata': {'dataset': dataset.name, 'protocol': version, 'output_dir': str(output),
                             'profile_state': settings['state_dir'], 'cloud': False,
                             'quality_claim': 'pending_evaluation'},
                'env': {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1'}}
        return submit(plan, state_dir=settings['state_dir'])
    except (ValueError, OSError, TypeError, KeyError):
        return failure('invalid_experiment', 'Supply compatible local inputs and a new output directory inside this profile workspace.')
