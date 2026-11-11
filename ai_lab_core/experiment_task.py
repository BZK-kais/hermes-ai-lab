"""Fixed durable-job entrypoint for native Hermes experiment tools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def receiver(args):
    from ai_lab_core.skill_eval import read_cases, compare
    from ai_lab_core.skill_runtime import run_batch
    from ai_lab_core.skill_experiment import _read_manifest, _verified_path
    start = time.monotonic()
    root = Path(args.dataset)
    manifest = _read_manifest(root)
    protocol = json.loads(_verified_path(root, manifest, 'protocol.json').read_text())
    cases = read_cases(_verified_path(root, manifest, 'receiver_test.jsonl'))
    if len(cases) != 24:
        raise ValueError('receiver cases')
    prompt_file = Path(args.prompt_file)
    if prompt_file.stat().st_size > 100000:
        raise ValueError('prompt size')
    chosen = json.loads(prompt_file.read_text())
    prompt = chosen['prompt']
    if hashlib.sha256(prompt.encode()).hexdigest() != chosen['prompt_sha256']:
        raise ValueError('prompt integrity')
    output = Path(args.output)
    output.mkdir(mode=0o700, exist_ok=False)
    options = {'python': args.python, 'model': args.model, 'lock_path': args.lock,
               'prompt': prompt, 'timeout': 600, 'expected_model': chosen['model_sha256']}
    base = run_batch([case['text'] for case in cases], **options)
    if not base['ok']:
        return base
    adapted = run_batch([case['text'] for case in cases], adapter=args.adapter,
                        expected_adapter=chosen['adapter_sha256'], **options)
    if not adapted['ok']:
        return adapted
    thresholds = protocol['thresholds']['receiver']
    result = compare(cases, base, adapted, **thresholds,
                     max_ambiguous_wrong_orientation_increase=protocol['thresholds']['max_ambiguous_wrong_orientation_increase'])
    if not result['ok']:
        return result
    return {'ok': True, 'passed': result['passed'], 'receiver': result,
            'model_sha256': adapted['model_sha256'], 'adapter_sha256': adapted['adapter_sha256'],
            'prompt_sha256': adapted['prompt_sha256'], 'adapter_path': args.adapter,
            'elapsed_seconds': round(time.monotonic()-start, 3),
            'dataset_sha256': manifest['receiver_test.jsonl'],
            'limitations': ['Synthetic labels assigned by an agent, without external human validation.',
                            'Two profiles on one Mac do not demonstrate cross-machine portability.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    for name in ('dataset', 'model', 'output', 'python', 'lock'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--prompt-file')
    parser.add_argument('--analysis-python')
    parser.add_argument('--baseline-freeze')
    args = parser.parse_args()
    try:
        from ai_lab_core.experiments import protocol_version
        version = protocol_version(Path(args.dataset))
        if version == 'pair-paraphrase-v1':
            from ai_lab_core.pair_experiment import run_experiment, run_receiver
            options = dict(python=args.python, model=args.model, lock_path=args.lock,
                           timeout=3600, analysis_python=args.analysis_python,
                           baseline_freeze=args.baseline_freeze,
                           progress=lambda event: print(json.dumps(event), flush=True))
            if args.mode == 'train':
                result = run_experiment(args.dataset, args.output, **options)
            else:
                result = run_receiver(args.dataset, args.output, adapter=args.adapter,
                                      prompt_file=args.prompt_file, **options)
        elif args.mode == 'train':
            from ai_lab_core.skill_experiment import run_experiment
            result = run_experiment(args.dataset, args.output, python=args.python, model=args.model,
                                    lock_path=args.lock, timeout=1800,
                                    progress=lambda event: print(json.dumps(event), flush=True))
        else:
            result = receiver(args)
    except Exception:
        result = {'ok': False, 'error': {'code': 'experiment_failed',
                  'message': 'The local experiment could not complete.'}}
    output = Path(args.output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = output / 'report.json'
    if args.mode != 'train' or not report.exists():
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('ok', 'passed', 'error', 'elapsed_seconds') if k in result}), flush=True)
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
