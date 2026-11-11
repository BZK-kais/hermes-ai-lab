"""Install and enable the native AI Lab plugin in an existing Hermes profile."""

import argparse
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

from build_release import MANIFEST, PROJECT, copy_release, release_payload


NAME = 'ai-lab'


class InstallError(Exception):
    pass


def _read_config(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise InstallError('Hermes config must be a regular file.')
    return path.read_bytes() if path.exists() else None


def _prepare_config(original, *, activate=True):
    import yaml
    source = (original or b'').decode('utf-8')
    raw = yaml.safe_load(source)
    before = {} if raw is None else raw
    plugins = before.get('plugins', {}) if isinstance(before, dict) else None
    if not isinstance(plugins, dict):
        raise InstallError('Hermes config must contain a plugins mapping.')
    for key in ('enabled', 'disabled'):
        values = plugins.get(key, [])
        if not isinstance(values, list) or any(type(value) is not str for value in values):
            raise InstallError('Hermes plugin lists must contain names.')

    enabled = plugins.get('enabled', [])
    disabled = plugins.get('disabled', [])
    already_enabled = NAME in enabled and NAME not in disabled
    if already_enabled or not activate:
        return original, already_enabled
    from ruamel.yaml import YAML

    yaml_rt = YAML(typ='rt')
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    expected = {**before, 'plugins': {**plugins}}
    document = yaml_rt.load(source)
    if document is None:
        document = {}
    entries = document.setdefault('plugins', {})
    if NAME not in enabled:
        entries.setdefault('enabled', []).append(NAME)
        expected['plugins']['enabled'] = [*enabled, NAME]
    if NAME in disabled:
        entries['disabled'][:] = [name for name in disabled if name != NAME]
        expected['plugins']['disabled'] = list(entries['disabled'])
    output = io.StringIO()
    yaml_rt.dump(document, output)
    rendered = output.getvalue()
    if yaml.safe_load(rendered) != expected:
        raise InstallError('Hermes config cannot be updated without changing other settings.')
    return rendered.encode('utf-8'), True


def _write_config(path, original, data):
    if _read_config(path) != original:
        raise InstallError('Hermes config changed during installation. Retry the install.')
    if data == original:
        return
    mode = stat.S_IMODE(path.stat().st_mode) if original is not None else 0o600
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.ai-lab-config-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _same_copy(destination, files):
    for name, data in files.items():
        path = destination / name
        if path.resolve() != path or not path.is_file() or path.read_bytes() != data:
            return False
    return True


def _verify_owned_copy(destination):
    """Verify the installer payload and tolerate only ordinary Python cache files."""
    error = 'The existing AI Lab copy does not match a clean release manifest and was preserved.'

    def owned(path, kind):
        info = path.lstat()
        if not kind(info.st_mode) or info.st_uid != os.getuid():
            raise InstallError(error)
        if kind is stat.S_ISREG and info.st_nlink != 1:
            raise InstallError(error)

    try:
        owned(destination, stat.S_ISDIR)
        manifest = destination / MANIFEST
        owned(manifest, stat.S_ISREG)
        with manifest.open('rb') as stream:
            raw = stream.read(262145)
        if len(raw) > 262144:
            raise InstallError(error)
        files = {}
        for line in raw.decode('utf-8').splitlines():
            match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
            if not match:
                raise InstallError(error)
            checksum, name = match.groups()
            path = PurePosixPath(name)
            if (str(path) != name or path.is_absolute() or any(part.startswith('.') for part in path.parts)
                    or '\\' in name or path in files or name == MANIFEST):
                raise InstallError(error)
            files[path] = checksum
        required = ('__init__.py', 'plugin.yaml', 'scripts/build_release.py', 'scripts/install_local.py',
                    'ai_lab_core/__init__.py')
        if any(PurePosixPath(name) not in files for name in required):
            raise InstallError(error)
        directories = {parent for path in files for parent in path.parents if parent.parts}
        cache_stems = {}
        for path in files:
            if path.suffix == '.py':
                cache_stems.setdefault(path.parent / '__pycache__', set()).add(path.stem)
        seen = set()
        for path in destination.rglob('*'):
            relative = PurePosixPath(path.relative_to(destination).as_posix())
            if relative in directories or relative in cache_stems:
                owned(path, stat.S_ISDIR)
            elif relative in files:
                owned(path, stat.S_ISREG)
                digest = sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1048576), b''):
                        digest.update(chunk)
                if digest.hexdigest() != files[relative]:
                    raise InstallError(error)
                seen.add(relative)
            elif relative != PurePosixPath(MANIFEST):
                owned(path, stat.S_ISREG)
                cache = re.fullmatch(r'(.+)\.cpython-\d+t?(?:\.opt-[12])?\.pyc', path.name)
                if not cache or cache[1] not in cache_stems.get(relative.parent, ()):
                    raise InstallError(error)
        if seen != files.keys():
            raise InstallError(error)
        import yaml

        if yaml.safe_load((destination / 'plugin.yaml').read_text(encoding='utf-8')).get('name') != NAME:
            raise InstallError(error)
    except (OSError, UnicodeError, AttributeError) as exc:
        raise InstallError(error) from exc


def _backup_path(home):
    directory = home / 'ai-lab-install-backups'
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise InstallError('The AI Lab backup directory must be a regular directory owned by this user.')
    return Path(tempfile.mkdtemp(dir=directory, prefix='backup-')) / NAME


def _publish_copy(destination, files, home, upgrade, update_config):
    backup = None
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.ai-lab-stage-') as temporary:
        staged = Path(temporary) / NAME
        copy_release(staged, files)
        if upgrade:
            _verify_owned_copy(destination)
            backup = _backup_path(home)
            try:
                destination.rename(backup)
            except OSError:
                backup.parent.rmdir()
                raise
        published = False
        try:
            staged.rename(destination)
            published = True
            update_config()
        except BaseException:
            try:
                if published:
                    destination.rename(staged)
                if backup is not None:
                    if os.path.lexists(destination):
                        raise OSError('destination occupied')
                    backup.rename(destination)
                    backup.parent.rmdir()
            except OSError as exc:
                if backup is not None and backup.exists():
                    raise InstallError(f'Automatic restoration failed. The previous AI Lab copy is retained at {backup}.') from exc
                raise
            raise
    return backup


def _install(project, home, link, upgrade=False):
    if not home.is_dir():
        raise InstallError('Create the Hermes profile before installing AI Lab.')
    plugins_dir = home / 'plugins'
    if plugins_dir.is_symlink() or (plugins_dir.exists() and not plugins_dir.is_dir()):
        raise InstallError('The profile plugins directory must be a regular directory.')
    plugins_dir.mkdir(exist_ok=True)
    descriptor = os.open(plugins_dir / '.ai-lab-install.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'rb') as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise InstallError('The plugin installation lock is invalid.')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _install_locked(project, home, link, upgrade)


def _install_locked(project, home, link, upgrade):
    destination = home / 'plugins' / NAME
    config = home / 'config.yaml'
    original = _read_config(config)
    prepared, enabled = _prepare_config(original, activate=not upgrade)
    files = None if link else release_payload(project)
    if files is not None and (project / MANIFEST).exists() and (project / MANIFEST).read_bytes() != files[MANIFEST]:
        raise InstallError('The extracted archive does not match its checksum manifest.')
    exists = os.path.lexists(destination)
    if upgrade and (link or not exists):
        raise InstallError('Upgrade requires an existing AI Lab copy installed from a release.')
    backup = None

    def update_config():
        _write_config(config, original, prepared)

    if exists:
        matches = destination.is_symlink() and destination.resolve() == project if link else (
            not destination.is_symlink() and destination.is_dir() and _same_copy(destination, files)
        )
        if not matches and not upgrade:
            raise InstallError('A different AI Lab installation already exists and was preserved.')
        if matches:
            if upgrade:
                _verify_owned_copy(destination)
            update_config()
        else:
            backup = _publish_copy(destination, files, home, upgrade=True, update_config=update_config)
    elif link:
        destination.symlink_to(project, target_is_directory=True)
        try:
            update_config()
        except BaseException:
            destination.unlink()
            raise
    else:
        _publish_copy(destination, files, home, upgrade=False, update_config=update_config)
    return {'ok': True, 'installed': True, 'enabled': enabled, 'path': str(destination),
            'mode': 'link' if link else 'copy', 'effective_on': 'next Hermes session',
            'upgraded': backup is not None, 'backup_path': str(backup) if backup else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, help='Existing profile home, or HERMES_HOME/the standard Hermes home.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--link', action='store_true', help='Link a stable development checkout instead of copying.')
    mode.add_argument('--upgrade', action='store_true', help='Replace a verified installed copy and retain its backup.')
    args = parser.parse_args()
    try:
        from hermes_cli import __version__
        from hermes_cli.config import is_managed
        from hermes_cli.managed_scope import managed_config_keys
        from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
        from packaging.specifiers import SpecifierSet
        import yaml

        manifest = yaml.safe_load((PROJECT / 'plugin.yaml').read_text(encoding='utf-8'))
        if manifest['name'] != NAME or __version__ not in SpecifierSet(manifest['requires_hermes']):
            raise InstallError('This plugin requires a compatible Hermes version.')
        home = (args.home or get_hermes_home()).expanduser().resolve()
        token = set_hermes_home_override(home)
        try:
            if is_managed() or managed_config_keys() & {'plugins', 'plugins.enabled', 'plugins.disabled'}:
                raise InstallError('Hermes plugin activation is managed for this profile.')
            report = _install(PROJECT, home, args.link, args.upgrade)
        finally:
            reset_hermes_home_override(token)
    except InstallError as exc:
        parser.exit(1, str(exc) + '\n')
    except ImportError:
        parser.exit(1, 'Run this installer with the Python interpreter used by Hermes.\n')
    except Exception:
        parser.exit(1, 'AI Lab installation could not complete. Check the profile and archive.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
