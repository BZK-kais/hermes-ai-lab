"""Plugin contract tests; connector behavior is tested separately."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
import argparse
import contextlib
import io
import tempfile
from unittest.mock import patch


_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ai_lab_test_plugin", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
_SERVICE_MODULE = plugin.__name__ + ".ai_lab_core.service"


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.tools = {}
        self.commands = {}
        self.skills = {}
        def command(name, help, setup_fn, handler_fn):
            self.commands[name] = (setup_fn, handler_fn)
        ctx = types.SimpleNamespace(register_tool=lambda **kw: self.tools.__setitem__(kw["name"], kw),
                                    register_cli_command=command,
                                    register_skill=lambda name,path,description: self.skills.__setitem__(name, path),
                                    dispatch_tool=lambda name,args: self.tools[name]['handler'](args))
        plugin.register(ctx)

    def test_native_cli_uses_registered_tool_and_rejects_bad_parameter_file(self):
        setup, handler = self.commands['ai-lab']
        parser = argparse.ArgumentParser()
        setup(parser)
        service = types.ModuleType(_SERVICE_MODULE)
        service.profile = lambda: {'ok': True, 'profile_id': 'unit-profile'}
        with patch.dict(sys.modules, {_SERVICE_MODULE: service}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(handler(parser.parse_args(['profile'])), 0)
        self.assertEqual(json.loads(output.getvalue())['profile_id'], 'unit-profile')
        service.run = lambda operation, parameters: {'ok': True, 'operation': operation, 'parameters': parameters}
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp)/'run.json'; params.write_text('{"operation":"mlx.train","parameters":{"iters":4}}')
            with patch.dict(sys.modules, {_SERVICE_MODULE: service}), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(handler(parser.parse_args(['run', '--args-file', str(params)])), 0)
            self.assertEqual(json.loads(output.getvalue())['parameters'], {'iters': 4})
        self.assertTrue(self.skills['workflow'].name == 'SKILL.md')
        for verb in ('analyze', 'benchmark', 'optimize', 'mission', 'registry', 'onboarding'):
            service.__dict__[verb] = lambda request: {'ok': True, 'received': request}
            with tempfile.TemporaryDirectory() as tmp:
                params = Path(tmp)/'request.json'; params.write_text('{"task": {}}')
                with patch.dict(sys.modules, {_SERVICE_MODULE: service}), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(handler(parser.parse_args([verb, '--args-file', str(params)])), 0)
                self.assertEqual(json.loads(output.getvalue())['received'], {'task': {}})
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp)/'invalid.json'; params.write_text('[]')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(handler(parser.parse_args(['experiment','--args-file',str(params)])), 1)
            self.assertEqual(json.loads(output.getvalue())['error']['code'], 'invalid_arguments')

    def test_dispatch_serializes_service_results(self):
        calls = []
        service = types.ModuleType(_SERVICE_MODULE)
        def operation(name):
            def invoke(**kwargs):
                calls.append((name, kwargs))
                return {"ok": True, "operation": name}
            return invoke
        cases = {
            "status": {},
            "list_local_models": {"limit": 5},
            "hf_model_info": {"repo_id": "org/model", "revision": "abc", "file_limit": 2},
            "probe_endpoint": {"base_url": "http://127.0.0.1:8080", "kind": "openai"},
        }
        for name in cases:
            setattr(service, name, operation(name))
        with patch.dict(sys.modules, {_SERVICE_MODULE: service}):
            for name, args in cases.items():
                result = self.tools["ai_lab_" + name]["handler"](args, session_id="test")
                self.assertIsInstance(result, str)
                self.assertEqual(json.loads(result), {"ok": True, "operation": name})
        self.assertEqual(calls, list(cases.items()))

    def test_rejects_invalid_arguments_before_service_dispatch(self):
        cases = [
            ("status", {"path": "/tmp"}),
            ("list_local_models", {"limit": True}),
            ("list_local_models", {"limit": 201}),
            ("hf_model_info", {}),
            ("hf_model_info", {"repo_id": "org/model", "file_limit": 0}),
            ("probe_endpoint", {"base_url": "http://localhost", "kind": "unknown"}),
            ("probe_endpoint", {"base_url": "http://localhost", "headers": {}}),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args):
                result = json.loads(self.tools["ai_lab_" + name]["handler"](args))
                self.assertEqual(result["error"]["code"], "invalid_arguments")

    def test_service_failure_does_not_disclose_exception(self):
        service = types.ModuleType(_SERVICE_MODULE)
        def fail():
            raise RuntimeError("private-provider-details")
        service.status = fail
        with patch.dict(sys.modules, {_SERVICE_MODULE: service}):
            result = self.tools["ai_lab_status"]["handler"]({})
        self.assertNotIn("private-provider-details", result)
        self.assertEqual(json.loads(result)["error"]["code"], "service_error")


if __name__ == "__main__":
    unittest.main()
