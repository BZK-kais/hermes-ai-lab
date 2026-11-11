"""Portable registry bundles with a bounded manifest and two optional LoRA files."""
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import tempfile
from zipfile import ZIP_STORED, ZipFile

from .controller import _json
from .controller_contract import fields, number, sha256
from .controller_engine import _json as canonical_json
from .skill_worker import ADAPTER_FILES

KIND = 'ai-lab-configuration-v1'
MAX_MANIFEST = 128 * 1024
MAX_ADAPTER = 1024**3
MANIFEST = 'manifest.json'
CHECKSUM = 'manifest.sha256'
FILES = tuple(f'adapter/{name}' for name in ADAPTER_FILES)
TENSOR_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}')
NO_METADATA = object()


def _adapter_config(raw):
    config = _json(raw)
    if type(config) is not dict or config.get('fine_tune_type') != 'lora':
        raise ValueError('adapter configuration')
    layers = config['num_layers']
    parameters = config['lora_parameters']
    if (not number(layers, -1, 8192, integer=True)
            or not fields(parameters, ('rank', 'scale', 'dropout'), ('keys',))
            or not number(parameters['rank'], 1, 8192, integer=True)
            or not number(parameters['scale'], 0, 2**31)
            or not number(parameters['dropout'], 0, 1)):
        raise ValueError('adapter parameters')
    if 'keys' in parameters and parameters['keys'] is not None:
        keys = parameters['keys']
        if (type(keys) is not list or len(keys) > 8192
                or not all(type(key) is str and TENSOR_NAME.fullmatch(key) for key in keys)):
            raise ValueError('adapter parameter keys')
    return canonical_json({'fine_tune_type': 'lora', 'num_layers': layers, 'lora_parameters': parameters}).encode()


def _weights_header(stream, size):
    prefix = stream.read(8)
    width = int.from_bytes(prefix, 'little')
    if len(prefix) != 8 or not 0 < width <= min(4 * 1024**2, size - 8):
        raise ValueError('adapter tensor header')
    header = stream.read(width)
    if len(header) != width:
        raise ValueError('adapter tensor header')
    tensors = _json(header)
    if type(tensors) is not dict:
        raise ValueError('adapter tensors')
    metadata = tensors.pop('__metadata__', NO_METADATA)
    if not tensors:
        raise ValueError('adapter tensors')
    for name, item in tensors.items():
        if (not TENSOR_NAME.fullmatch(name) or not fields(item, ('dtype', 'shape', 'data_offsets'))
                or type(item['dtype']) is not str or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,15}', item['dtype'])
                or type(item['shape']) is not list or len(item['shape']) > 16
                or not all(number(dim, 0, MAX_ADAPTER, integer=True) for dim in item['shape'])
                or type(item['data_offsets']) is not list or len(item['data_offsets']) != 2
                or not all(number(offset, 0, size - 8 - width, integer=True) for offset in item['data_offsets'])):
            raise ValueError('adapter tensor descriptor')
    return prefix + header, tensors, metadata


def _project_weights(source, target, size):
    original_header, tensors, metadata = _weights_header(source, size)
    header = original_header
    if metadata is not NO_METADATA:
        descriptors = canonical_json(tensors).encode()
        width = len(original_header) - 8
        if len(descriptors) > width:
            raise ValueError('adapter tensor header')
        # Space padding keeps the header length and physical tensor offsets.
        header = original_header[:8] + descriptors.ljust(width, b' ')
    original = hashlib.sha256(original_header)
    projected = hashlib.sha256(header)
    target.write(header)
    copied = len(header)
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        copied += len(chunk)
        if copied > size:
            raise ValueError('adapter changed')
        target.write(chunk)
        original.update(chunk)
        projected.update(chunk)
    if copied != size:
        raise ValueError('adapter changed')
    return {'sha256': original.hexdigest()}, {'bytes': copied, 'sha256': projected.hexdigest()}


def _aggregate(assets):
    fingerprints = [[Path(name).name, assets[name]['sha256']] for name in sorted(assets)]
    return hashlib.sha256(canonical_json(fingerprints).encode()).hexdigest()


def _copy(source, target, limit):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        size += len(chunk)
        if size > limit:
            raise ValueError('package size')
        target.write(chunk)
        digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def export_bundle(path, descriptor, adapter):
    fd, temporary = tempfile.mkstemp(prefix='.ai-lab-package-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w+b') as target:
            with ZipFile(target, 'w', compression=ZIP_STORED) as bundle:
                assets = {}
                identity = descriptor['identity'].copy()
                if adapter is not None:
                    originals = {}
                    for name, source_name in zip(FILES, ADAPTER_FILES):
                        source = Path(adapter) / source_name
                        limit = MAX_ADAPTER if source.suffix == '.safetensors' else 100_000
                        with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                            info = os.fstat(stream.fileno())
                            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                                raise ValueError('invalid adapter file')
                            with bundle.open(name, 'w', force_zip64=True) as entry:
                                if source.suffix == '.json':
                                    raw = stream.read(limit + 1)
                                    if len(raw) != info.st_size:
                                        raise ValueError('adapter changed')
                                    originals[name] = {'sha256': hashlib.sha256(raw).hexdigest()}
                                    assets[name] = _copy(io.BytesIO(_adapter_config(raw)), entry, limit)
                                else:
                                    originals[name], assets[name] = _project_weights(stream, entry, info.st_size)
                    if _aggregate(originals) != identity['adapter_sha256']:
                        raise ValueError('adapter changed')
                    identity['adapter_sha256'] = _aggregate(assets)
                raw = canonical_json({'kind': KIND, **descriptor, 'identity': identity, 'files': assets}).encode()
                if len(raw) > MAX_MANIFEST:
                    raise ValueError('manifest size')
                bundle.writestr(MANIFEST, raw)
                bundle.writestr(CHECKSUM, hashlib.sha256(raw).hexdigest().encode())
            target.flush()
            os.fsync(target.fileno())
        # Link publishes the complete package exclusively, even for concurrent exports.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def read_bundle(path, stage, validate):
    limit = MAX_ADAPTER + MAX_MANIFEST + 256 * 1024
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError('invalid package file')
        # ZipFile materializes its entire central directory before listing entries.
        # Our format has four small records and no archive comment or ZIP64 directory.
        stream.seek(-22, os.SEEK_END)
        end = stream.read(22)
        if (end[:4] != b'PK\x05\x06' or end[4:8] != b'\0' * 4 or end[20:] != b'\0' * 2
                or end[8:10] != end[10:12] or int.from_bytes(end[10:12], 'little') not in (2, 4)
                or int.from_bytes(end[12:16], 'little') > 4096
                or int.from_bytes(end[16:20], 'little') + int.from_bytes(end[12:16], 'little') != info.st_size - 22):
            raise ValueError('invalid package directory')
        with ZipFile(stream) as bundle:
            entries = bundle.infolist()
            names = {entry.filename for entry in entries}
            if (len(entries) != len(names)
                    or names not in ({MANIFEST, CHECKSUM}, {MANIFEST, CHECKSUM, *FILES})):
                raise ValueError('invalid package entries')
            for entry in entries:
                cap = {MANIFEST: MAX_MANIFEST, CHECKSUM: 64, FILES[0]: 100_000, FILES[1]: MAX_ADAPTER}[entry.filename]
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (entry.orig_filename != entry.filename or entry.flag_bits & 1 or mode not in (0, stat.S_IFREG)
                        or entry.compress_type != ZIP_STORED or entry.compress_size != entry.file_size
                        or not 0 < entry.file_size <= cap):
                    raise ValueError('invalid package entry')
            raw = bundle.read(MANIFEST)
            if bundle.read(CHECKSUM) != hashlib.sha256(raw).hexdigest().encode():
                raise ValueError('manifest checksum')
            manifest = _json(raw)
            validate(manifest)
            files = manifest['files']
            if type(files) is not dict or set(files) != names - {MANIFEST, CHECKSUM}:
                raise ValueError('invalid package inventory')
            if bool(files) != (manifest['identity']['adapter_sha256'] is not None):
                raise ValueError('missing adapter files')
            for name in sorted(files):
                recorded = files[name]
                entry = bundle.getinfo(name)
                if (type(recorded) is not dict or set(recorded) != {'bytes', 'sha256'}
                        or type(recorded['bytes']) is not int or recorded['bytes'] != entry.file_size
                        or not sha256(recorded['sha256'])):
                    raise ValueError('invalid file digest')
                destination = stage / Path(name).name
                with bundle.open(entry) as source, destination.open('xb') as target:
                    if _copy(source, target, entry.file_size) != recorded:
                        raise ValueError('adapter checksum')
                    target.flush()
                    os.fsync(target.fileno())
                with destination.open('rb') as stream:
                    if name == FILES[0]:
                        raw = stream.read()
                        if _adapter_config(raw) != raw:
                            raise ValueError('adapter configuration is not portable')
                    else:
                        _, _, metadata = _weights_header(stream, entry.file_size)
                        if (metadata is not NO_METADATA
                                and metadata not in ({}, {'format': 'pt'}, {'format': 'mlx'}, {'format': 'np'})):
                            raise ValueError('adapter contains extra metadata')
    return manifest
