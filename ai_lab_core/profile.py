"""Bind AI Lab data to the Hermes home captured at plugin registration.

Context-local bindings prevent multiplexed profiles from borrowing ambient
process configuration. This is state separation, not an OS sandbox.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

_HOME = ContextVar('ai_lab_profile_home', default=None)
PROJECT = Path(__file__).resolve().parents[1]


@contextmanager
def profile_scope(home):
    token = _HOME.set(Path(home).resolve() if home is not None else None)
    try:
        yield
    finally:
        _HOME.reset(token)


def bound_home():
    return _HOME.get()


def profile_home():
    home = bound_home()
    if home is not None:
        return home
    value = os.environ.get('HERMES_HOME')
    return Path(value).expanduser().resolve() if value else None


def data_root():
    home = profile_home()
    legacy = PROJECT / '.local'
    if home is not None:
        # An existing development profile can migrate explicitly to home/ai-lab.
        if (home == (Path.home() / '.hermes').resolve()
                and not (home / 'ai-lab').exists()
                and (legacy / 'integrations.json').is_file()):
            return legacy
        return home / 'ai-lab'
    if any((legacy / name).is_file() for name in ('integrations.json', 'connections.json')):
        return legacy
    xdg = Path(os.environ.get('XDG_DATA_HOME', '')).expanduser()
    return (xdg if xdg.is_absolute() else Path.home() / '.local/share') / 'hermes-ai-lab'


def device_lock_path():
    """Use one MLX device lock per user across all profiles and installations."""
    return Path.home() / '.cache/hermes-ai-lab/device-mlx.lock'


def config_path(filename, environment_name):
    # Native handlers never read another profile's process-wide AI_LAB_* path.
    if bound_home() is not None:
        return data_root() / filename
    return Path(os.environ.get(environment_name, str(data_root() / filename)))
