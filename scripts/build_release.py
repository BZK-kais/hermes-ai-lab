"""Build a deterministic source archive for the native Hermes plugin."""

import argparse
from hashlib import sha256
from pathlib import Path
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = 'MANIFEST.sha256'
ROOT_FILES = ('__init__.py', 'plugin.yaml', 'pyproject.toml', 'README.md', 'STATUS.md', 'LICENSE')
SCRIPTS = ('build_release.py', 'build_remote_worker.py', 'check_hermes_registration.py',
           'evaluate_pair_baselines.py', 'install_cli.py', 'install_local.py', 'prepare_demo.py')
EXAMPLES = {
    'lifecycle-v1': ('README.md', 'dev.jsonl', 'test.jsonl', 'manifest.sha256'),
    'smoke-data': ('train.jsonl', 'valid.jsonl'),
    'optimization': ('extraction-fr.json', 'retrieval-qa.json', 'mac-interactive.json'),
    'experiments/controller-v1': ('README.md', 'dev.jsonl', 'test.jsonl', 'manifest.sha256', 'provenance.json'),
    'experiments/lora-improve-v1': ('README.md', 'dev.jsonl', 'test.jsonl', 'train.jsonl', 'manifest.sha256', 'provenance.json'),
    'experiments/pair-paraphrase-v1': ('README.md', 'contract.json', 'protocol.json', 'prompt_candidates.json', 'manifest.sha256',
                                      'provenance.json', 'train.jsonl', 'dev.jsonl', 'producer_test.jsonl', 'receiver_test.jsonl'),
    'experiments/triage-v1': ('README.md', 'contract.json', 'protocol.json', 'prompt_candidates.json', 'manifest.sha256',
                            'train.jsonl', 'dev.jsonl', 'producer_test.jsonl', 'receiver_test.jsonl'),
}


def release_files(project):
    paths = [project / name for name in ROOT_FILES]
    paths += [project / 'scripts' / name for name in SCRIPTS]
    paths += [project / 'examples' / folder / name for folder, names in EXAMPLES.items() for name in names]
    paths += [project / 'skills/ai-lab/SKILL.md']
    paths += list((project / 'skills/ai-lab/references').rglob('*.md'))
    paths += list((project / 'docs').rglob('*.md'))
    paths += list((project / 'ai_lab_core').rglob('*.py'))
    paths += [path for path in (project / 'ai_lab_core/recipes').iterdir()
              if path.suffix in {'.json', '.yaml', '.yml'}]
    paths += [project / 'ai_lab_core/assets/smoke-data' / name for name in EXAMPLES['smoke-data']]
    for path in sorted(paths):
        if any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(project).parts):
            continue
        if not path.is_file() or path.resolve() != path:
            raise ValueError('release file is missing or a symbolic link')
        yield path.relative_to(project).as_posix(), path.read_bytes()


def release_payload(project):
    files = dict(release_files(project))
    if files['scripts/evaluate_pair_baselines.py'] != files['ai_lab_core/assets/evaluate_pair_baselines.py']:
        raise ValueError('CPU baseline worker sources differ')
    for name in EXAMPLES['smoke-data']:
        if files[f'examples/smoke-data/{name}'] != files[f'ai_lab_core/assets/smoke-data/{name}']:
            raise ValueError('packaged smoke fixtures differ')
    files[MANIFEST] = ''.join(f'{sha256(data).hexdigest()}  {name}\n' for name, data in sorted(files.items())).encode('utf-8')
    return files


def copy_release(destination, files):
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def build_archive(project, output):
    files = release_payload(project)
    with output.open('xb') as stream:
        try:
            with zipfile.ZipFile(stream, 'w') as archive:
                for name, data in sorted(files.items()):
                    entry = zipfile.ZipInfo('ai-lab/' + name)
                    entry.create_system = 3
                    entry.external_attr = 0o100644 << 16
                    archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        except BaseException:
            output.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New ZIP file with an existing parent directory.')
    args = parser.parse_args()
    try:
        build_archive(PROJECT, args.output.expanduser())
    except (OSError, ValueError, KeyError):
        parser.exit(1, 'Could not build the archive. Check the release sources and choose a new output path.\n')
    print(args.output.expanduser().absolute())


if __name__ == '__main__':
    main()
