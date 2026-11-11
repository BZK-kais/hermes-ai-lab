"""Bounded, read-only discovery of local hardware, tools and model artifacts."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], include_stderr: bool = False) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=3,
                                env={**os.environ, 'HF_HUB_OFFLINE': '1'})
        return (result.stdout + (result.stderr if include_stderr else ''))[:16384] if result.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return ''


def hardware() -> dict:
    info = {'ok': True, 'system': platform.system(), 'architecture': platform.machine(),
            'cpu_count': os.cpu_count(), 'gpu_execution_verified': False}
    if platform.system() == 'Darwin':
        raw = _run(['/usr/sbin/system_profiler', 'SPHardwareDataType', '-json'])
        try:
            data = json.loads(raw)['SPHardwareDataType'][0]
            info.update({key: data.get(source) for key, source in
                         [('model', 'machine_model'), ('chip', 'chip_type'),
                          ('memory', 'physical_memory')]})
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    smi = shutil.which('nvidia-smi')
    if smi:
        rows = _run([smi, '--query-gpu=name,memory.total,driver_version',
                     '--format=csv,noheader,nounits'])
        devices = []
        for row in rows.splitlines()[:16]:
            fields = [item.strip() for item in row.split(',')]
            if len(fields) == 3 and fields[1].isdigit():
                devices.append({'name': fields[0][:120], 'memory_mib': int(fields[1]),
                                'driver_version': fields[2][:40]})
        info['nvidia_devices'] = devices
    return info


def runtime_inventory(extra_bin_dirs: list[str] | None = None, *, inspect_mlx_package: bool = True) -> dict:
    tools = []
    if extra_bin_dirs is not None and (not isinstance(extra_bin_dirs, list) or
            any(not isinstance(item, str) for item in extra_bin_dirs)):
        return {'ok': False, 'error': {'code': 'invalid_runtime_dirs',
                                      'message': 'Runtime directories must be paths.'}}
    extra_dirs = [Path(item).expanduser() for item in (extra_bin_dirs or [])[:32]]
    for name in ('llama-server', 'llama-cli', 'llama-quantize', 'hf', 'mlx_lm.server'):
        candidates = [str(directory / name) for directory in extra_dirs]
        candidates.extend([shutil.which(name), str(Path('/opt/homebrew/bin') / name),
                           str(Path(sys.executable).parent / name)])
        path = next((p for p in candidates if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
        entry = {'name': name, 'installed': bool(path), 'path': path,
                 'service_reachable': None, 'generation_verified': False}
        if path:
            raw = _run([path, '--version'], include_stderr=True) if not name.startswith('mlx') else ''
            match = re.search(r'(?im)^(?:version:\s*|hf(?:\s+version)?[: ]+)?(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)\b', raw)
            entry['version'] = match.group(1) if match else None
            build = re.search(r'build (\d+), commit ([a-fA-F0-9]{7,40})', raw)
            if build:
                entry.update({'build': int(build.group(1)), 'commit': build.group(2)})
            if name == 'mlx_lm.server' and inspect_mlx_package:
                python = Path(path).parent / 'python'
                version = _run([str(python), '-I', '-c',
                                "from importlib.metadata import version; print(version('mlx-lm'))"]).strip()
                entry['version'] = version if re.fullmatch(r'\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?', version) else None
        tools.append(entry)
    return {'ok': True, 'runtimes': tools, 'python': {'path': sys.executable,
            'version': platform.python_version()}, 'generation_verified': False}


def inventory_models(roots: list[str], limit: int = 50) -> dict:
    """Inspect filenames, sizes and small shard indexes; never open tensor payloads.

    Symlink files must resolve inside their configured root (HF snapshot links into
    that cache's blobs work). Directory symlinks are not traversed.
    """
    if not isinstance(roots, list) or any(not isinstance(r, str) for r in roots):
        return {'ok': False, 'error': {'code': 'invalid_roots', 'message': 'Model roots must be paths.'}}
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        return {'ok': False, 'error': {'code': 'invalid_limit', 'message': 'Limit must be between 1 and 500.'}}
    models, seen_models, physical = [], set(), set()
    total_bytes = 0
    truncated = False
    ignored = {'.locks', '.git', '.venv', 'venv', 'node_modules', 'tmp', 'temp', '__pycache__'}
    visited = 0
    for value in roots[:32]:
        try:
            root = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not root.is_dir():
            continue

        def safe_file(path: Path) -> bool:
            try:
                return path.resolve(strict=True).is_relative_to(root) and path.is_file()
            except (OSError, RuntimeError):
                return False

        pending = [(root, 0)]
        while pending:
            directory, depth = pending.pop()
            try:
                with os.scandir(directory) as stream:
                    entries = []
                    for entry in stream:
                        visited += 1
                        if visited > 10000:
                            truncated = True
                            break
                        entries.append(entry)
            except OSError:
                continue
            if visited > 10000:
                break
            names = {e.name: Path(e.path) for e in entries if e.is_file(follow_symlinks=True)}
            tensors = [p for n, p in names.items()
                       if (n.endswith('.safetensors') or
                           (n.endswith('.bin') and n.startswith(('pytorch_model', 'model'))))
                       and safe_file(p)]
            indexes = [p for n, p in names.items() if n.endswith(('.safetensors.index.json', '.bin.index.json'))]
            configs = 'config.json' in names
            groups = []
            if tensors or indexes or configs:
                groups.append(('huggingface', directory, tensors, indexes))
            ggufs = [p for n, p in names.items() if n.lower().endswith('.gguf') and safe_file(p)]
            shard_groups = {}
            for p in ggufs:
                match = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', p.name)
                if match:
                    shard_groups.setdefault((match.group(1), int(match.group(3))), []).append(p)
                else:
                    groups.append(('gguf', p, [p], []))
            for (prefix, count), files in shard_groups.items():
                groups.append(('gguf', directory / (prefix + '-sharded.gguf'), files,
                               [directory / f'{prefix}-{i:05d}-of-{count:05d}.gguf'
                                for i in range(1, count + 1)] if 1 <= count <= 10000 else []))
            for kind, path, files, required in groups:
                identity = str(path)
                if identity in seen_models:
                    continue
                if len(models) >= limit:
                    truncated = True
                    break
                seen_models.add(identity)
                missing, issue = [], None
                if kind == 'huggingface':
                    for index in required:
                        try:
                            if not safe_file(index) or index.stat().st_size > 2_000_000:
                                issue = 'index_unreadable'
                                continue
                            index_data = json.loads(index.read_text())
                            if not isinstance(index_data, dict):
                                issue = 'invalid_index'
                                continue
                            weights = index_data.get('weight_map', {})
                            if not isinstance(weights, dict) or not weights:
                                issue = 'invalid_index'
                                continue
                            for filename in set(weights.values()):
                                if not isinstance(filename, str):
                                    issue = 'invalid_index'
                                elif not safe_file(directory / filename):
                                    missing.append(filename[:200])
                        except (OSError, ValueError, TypeError, RuntimeError):
                            issue = 'index_unreadable'
                else:
                    shard_counts = [int(m.group(1)) for p in files
                                    if (m := re.search(r'-of-(\d{5})\.gguf$', p.name))]
                    if any(count == 0 or count > 10000 for count in shard_counts):
                        issue = 'invalid_shard_count'
                    missing = [p.name for p in required if not safe_file(p)]
                logical_bytes, new_bytes = 0, 0
                for p in files:
                    try:
                        stat = p.stat()
                    except OSError:
                        continue
                    if stat.st_size == 0:
                        issue = issue or 'empty_weights'
                    logical_bytes += stat.st_size
                    key = (stat.st_dev, stat.st_ino)
                    if key not in physical:
                        physical.add(key)
                        new_bytes += stat.st_size
                total_bytes += new_bytes
                models.append({'path': identity, 'format': kind, 'files': len(files),
                               'size_bytes': logical_bytes, 'unique_size_bytes': new_bytes,
                               'complete': bool(files) and not missing and not issue,
                               'verification': 'file_inventory_only', 'missing_files': sorted(missing)[:50],
                               'issue': issue or ('weights_missing' if not files else None),
                               'error': {'code': issue, 'message': 'Model files are incomplete or invalid.'}
                               if issue else None})
            if len(models) >= limit:
                if pending or any(e.is_dir(follow_symlinks=False) for e in entries):
                    truncated = True
                break
            for entry in sorted(entries, key=lambda e: e.name, reverse=True):
                if entry.name in ignored or entry.name.startswith('.'):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if depth < 6:
                        pending.append((Path(entry.path), depth + 1))
                    else:
                        truncated = True
        if visited > 10000 or len(models) >= limit:
            break
    return {'ok': True, 'models': models, 'unique_size_bytes': total_bytes,
            'truncated': truncated or len(roots) > 32, 'generation_verified': False}
