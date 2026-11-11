"""Verify the installed Hermes SDK and real handlers from outside the project.

Run with Hermes's Python interpreter. Doctor supplies an isolated Hermes home
and blocks sockets. This intentionally uses the installed private Doctor
helper for a local compatibility check; it is not a plugin dependency.
"""

import json
from pathlib import Path
import tempfile
import os
import shutil

from hermes_cli.plugin_dev import _doctor_runtime
from tools.registry import registry


def main():
    project = Path(__file__).resolve().parents[1]
    cases = (
        ("ai_lab_status", {}, True),
        ("ai_lab_list_local_models", {"limit": 1}, True),
        ("ai_lab_operations", {}, True),
        ("ai_lab_connections", {}, True),
        ("ai_lab_analyze", {"request": {}}, True),
        ("ai_lab_benchmark", {"request": {"action": "start"}}, False),
        ("ai_lab_optimize", {"request": {"action": "start"}}, False),
        ("ai_lab_onboarding", {"request": {"action": "check"}}, True),
        ("ai_lab_mission", {"request": {"action": "prepare"}}, True),
        ("ai_lab_mission", {"request": {"action": "autoplan"}}, True),
        ("ai_lab_mission", {"request": {"action": "lessons"}}, True),
        ("ai_lab_registry", {"request": {"action": "status"}}, True),
        # Invalid discovery inputs exercise real services without network I/O.
        ("ai_lab_hf_model_info", {"repo_id": "../invalid"}, False),
        ("ai_lab_probe_endpoint", {"base_url": "https://example.com", "kind": "openai"}, False),
    )
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="ai-lab-outside-project-") as cwd:
            os.chdir(cwd)
            # Doctor copies its input. Stage source only: the development tree
            # now contains profile plugin symlinks back to this repository.
            staged = Path(cwd) / 'plugin-source'
            staged.mkdir()
            for name in ('__init__.py', 'plugin.yaml'):
                shutil.copy2(project / name, staged / name)
            shutil.copytree(project / 'ai_lab_core', staged / 'ai_lab_core',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copytree(project / 'skills', staged / 'skills')
            with _doctor_runtime(staged) as runtime:
                expected = {
                    'ai_lab_' + name for name in ('status', 'list_local_models', 'hf_model_info',
                    'probe_endpoint', 'connections', 'connection_probe', 'operations', 'plan', 'run',
                    'job', 'provider_action', 'dataset_validate', 'artifact_inspect', 'generate',
                    'profile', 'experiment', 'analyze', 'benchmark', 'optimize', 'onboarding', 'mission', 'registry')
                }
                assert set(runtime.registered_tools) == expected, "Missing registered tool"
                for name, args, expected_ok in cases:
                    raw = registry.dispatch(name, args, scope=runtime.manager.scope_key)
                    assert isinstance(raw, str), "Handler did not return JSON text"
                    result = json.loads(raw)
                    assert result.get("ok") is expected_ok, name + " returned an unexpected result"
                    assert result.get("error", {}).get("code") != "service_error", "Service import or execution failed"
                    print(name + ": real registry dispatch passed")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
