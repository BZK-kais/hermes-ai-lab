import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_lab_core.profile import profile_scope
from ai_lab_core.integration_config import load_integrations
from ai_lab_core import service


class ProfileScopeTests(unittest.TestCase):
    def test_only_original_default_home_can_keep_legacy_project_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            (project / '.local').mkdir()
            (project / '.local/integrations.json').write_text('{}')
            with patch('ai_lab_core.profile.PROJECT', project), patch('ai_lab_core.profile.Path.home', return_value=project / 'user'):
                with profile_scope(project / 'user/.hermes'):
                    self.assertEqual(load_integrations()['state_dir'], str(project / '.local/state'))
                with profile_scope(project / 'user/.hermes/profiles/second'):
                    self.assertEqual(load_integrations()['state_dir'], str(project / 'user/.hermes/profiles/second/ai-lab/state'))

    def test_bound_profiles_ignore_ambient_path_overrides_and_restore_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            env = {'AI_LAB_CONFIG': str(Path(tmp) / 'poison-config'),
                   'AI_LAB_INTEGRATIONS': str(Path(tmp) / 'poison-integrations'),
                   'AI_LAB_STATE_DIR': str(Path(tmp) / 'poison-state')}
            for key in ('AI_LAB_CONFIG', 'AI_LAB_INTEGRATIONS'):
                Path(env[key]).write_text('not-json')
            with patch.dict(os.environ, env), profile_scope(a):
                first = service.profile()
                with profile_scope(b):
                    second = service.profile()
                self.assertEqual(service.profile(), first)
            self.assertEqual(first['state_dir'], str(a.resolve() / 'ai-lab/state'))
            self.assertEqual(second['state_dir'], str(b.resolve() / 'ai-lab/state'))
            self.assertNotEqual(first['profile_id'], second['profile_id'])

    def test_virtualenv_interpreter_binding_keeps_its_symlink_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve()
            target = home / 'python-real'; target.write_text('placeholder')
            executable = home / 'venv/bin/python'; executable.parent.mkdir(parents=True)
            executable.symlink_to(target)
            data = home / 'ai-lab'; data.mkdir()
            (data / 'integrations.json').write_text(json.dumps({'backend_paths': {'mlx.python': str(executable)}}))
            with profile_scope(home):
                self.assertEqual(load_integrations()['backend_paths']['mlx.python'], str(executable))

    def test_native_config_cannot_redirect_writable_state_to_other_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            (a / 'ai-lab').mkdir(parents=True)
            (a / 'ai-lab/integrations.json').write_text(json.dumps({'state_dir': str(b / 'state')}))
            with profile_scope(a), self.assertRaises(ValueError):
                load_integrations()

    def test_jobs_are_separate_in_real_databases(self):
        from ai_lab_core import jobs
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            with profile_scope(a):
                state_a = load_integrations()['state_dir']
                self.assertEqual(jobs.list_jobs(state_dir=state_a)['jobs'], [])
            with profile_scope(b):
                state_b = load_integrations()['state_dir']
                self.assertEqual(jobs.list_jobs(state_dir=state_b)['jobs'], [])
            self.assertNotEqual(Path(state_a, 'jobs.sqlite3').stat().st_ino,
                                Path(state_b, 'jobs.sqlite3').stat().st_ino)


if __name__ == '__main__':
    unittest.main()
