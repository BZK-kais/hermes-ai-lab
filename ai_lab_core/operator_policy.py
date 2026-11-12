"""Limit remote changes to named operator CLI commands."""
from asyncio import current_task
from contextlib import contextmanager
from contextvars import ContextVar
from threading import current_thread

from .endpoint import failure

_operator = ContextVar('ai_lab_operator_command', default=None)
_PROVIDER_READS = {
    'runpod': frozenset({'probe', 'list', 'get', 'logs'}),
    'hf_jobs': frozenset({'probe', 'list', 'get', 'logs'}),
    'hf_endpoints': frozenset({'probe', 'list', 'get'}),
    'hf_storage': frozenset({'probe', 'list', 'get'}),
    'gcs': frozenset({'probe', 'list', 'get'}),
    's3': frozenset({'probe', 'list', 'get'}),
    'ssh': frozenset({'probe', 'status'}),
}


def _owner():
    try:
        task = current_task()
    except RuntimeError:  # The CLI normally runs outside an event loop.
        task = None
    return current_thread(), task


@contextmanager
def operator_scope(command):
    """Enter the named CLI dispatch for this thread and task."""
    owner = _owner()
    active = True

    def allows(requested):
        return active and requested == command and _owner() == owner

    token = _operator.set(allows)
    try:
        yield
    finally:
        active = False
        _operator.reset(token)


def _authorized(command):
    allows = _operator.get()
    return allows is not None and allows(command)


def provider_denial(kind, action):
    # MCP tools/call is opaque, even when a tool's name sounds read-only.
    if not isinstance(action, str):
        return failure('invalid_input', 'A provider action name is required.')
    if action in _PROVIDER_READS.get(kind, ()) or _authorized('provider-action'):
        return None
    return failure('operator_required',
                   'Use the operator CLI command ailab provider-action or hermes ai-lab provider-action.')


def upload_denial():
    if _authorized('run'):
        return None
    return failure('operator_required',
                   'Use the operator CLI command ailab run or hermes ai-lab run for storage.upload.')
