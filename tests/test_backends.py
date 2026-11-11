import json
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab_core.backends import OPS, catalog, plan_operation
from ai_lab_core.recipes.trl_sft import load_config, ConfigurationError


class BackendPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        self.data = self.root / 'data'
        self.data.mkdir()
        self.weight = self.root / 'model.gguf'
        self.weight.write_bytes(b'fixture')
        self.train = self.data / 'train.jsonl'
        self.train.write_text('{"text":"fixture"}\n')
        self.config = self.root / 'recipe.json'
        self.config.write_text(json.dumps({'model':str(self.model),'train_file':str(self.train)}))
        self.binary = self.root / 'engine'
        self.binary.write_text('#!/bin/sh\nexit 97\n')
        self.binary.chmod(0o755)
        self.settings = {'workspace_root':str(self.root),'backend_paths':{
            tool:str(self.binary) for tool,_ in OPS.values()},'backend_recipe_args':{
                name:['--config','{config}','--output-dir','{output_dir}']
                for name in ('nemo.bridge.train','nemo.rl.train')}}
        self.settings['backend_paths'].pop('trl.train')
        self.package_check = patch('ai_lab_core.backends._python_has',return_value=True)
        self.package_check.start()
        self.addCleanup(self.package_check.stop)

    def plan(self,op,**params):
        return plan_operation(op,params,self.settings)

    def test_every_operation_builds_a_plan_without_executing_engine(self):
        common = {'config':str(self.config),'output_dir':str(self.root/'result')}
        cases = {
            'hf.download':{'repo_id':'org/model','output_dir':str(self.root/'download')},
            'mlx.convert':{'model':str(self.model),'output_dir':str(self.root/'converted'),'bits':4},
            'mlx.train':{'model':str(self.model),'data':str(self.data),'output_dir':str(self.root/'adapter')},
            'mlx.fuse':{'model':str(self.model),'adapter':str(self.data),'output_dir':str(self.root/'fused')},
            'mlx.serve':{'model':str(self.model)},
            'mlx.generate':{'model':str(self.model),'prompt':'Say hello','max_tokens':8},
            'hermes.export':{'session_id':'session_123456','output_file':str(self.root/'session.jsonl')},
            'llama.convert':{'model':str(self.model),'output_file':str(self.root/'converted.gguf')},
            'llama.quantize':{'model':str(self.weight),'output_file':str(self.root/'q.gguf'),'quant':'Q4_K_M'},
            'llama.imatrix':{'model':str(self.weight),'data':str(self.train),'output_file':str(self.root/'matrix.gguf')},
            'llama.serve':{'model':str(self.weight)},
            'lm_eval.evaluate':{'model':str(self.model),'tasks':['hellaswag'],'output_dir':str(self.root/'eval')},
        }
        for op in OPS:
            with self.subTest(operation=op):
                result = self.plan(op,**cases.get(op,common))
                self.assertTrue(result['ok'],result)
                plan = result['plan']
                self.assertEqual(plan['operation'],op)
                self.assertTrue(all(isinstance(arg,str) for arg in plan['command']))
                self.assertFalse((self.root/'result').exists())
                if op.endswith('.serve'):
                    self.assertIn('127.0.0.1',plan['command'])
                elif op != 'mlx.generate':
                    self.assertTrue(plan['outputs'])

    def test_unknown_arguments_injection_and_invalid_types_rejected(self):
        for extra in ({'command':['sh','-c','touch /tmp/oops']},{'port':True},{'host':'0.0.0.0'}, {'port':80}):
            with self.subTest(extra=extra):
                result = self.plan('llama.serve',model=str(self.weight),**extra)
                self.assertEqual(result['error']['code'],'invalid_parameters')
        self.assertFalse(self.plan('hf.download',repo_id='--token',output_dir=str(self.root/'out'))['ok'])

    def test_symlink_escape_output_overwrite_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as outer:
            (self.root/'escape').symlink_to(outer,target_is_directory=True)
            result = self.plan('llama.quantize',model=str(self.weight),quant='Q4_K_M',
                               output_file=str(self.root/'escape'/'x.gguf'))
            self.assertEqual(result['error']['code'],'path_not_allowed')
        result = self.plan('llama.quantize',model=str(self.weight),quant='Q4_K_M',output_file=str(self.weight))
        self.assertEqual(result['error']['code'],'output_exists')
        result = self.plan('mlx.convert',model=str(self.model),output_dir=str(self.model/'nested'))
        self.assertEqual(result['error']['code'],'source_output_overlap')

    def test_missing_bindings_are_visible_and_never_fake_success(self):
        self.settings['backend_paths']['llama.convert'] = str(self.root/'missing.py')
        self.settings['backend_recipe_args'] = {}
        self.assertEqual(self.plan('nemo.bridge.train',config=str(self.config),
                                  output_dir=str(self.root/'result'))['error']['code'],'configuration_required')
        row = next(r for r in catalog(self.settings)['operations'] if r['operation']=='llama.convert')
        self.assertFalse(row['available'])
        self.assertTrue(row['missing_requirements'])
        self.assertFalse(row['execution_verified'])

    def test_hf_anonymous_sdk_and_offline_transform(self):
        result = self.plan('hf.download',repo_id='org/model',output_dir=str(self.root/'out'))['plan']
        self.assertIn('hf-download',result['command'])
        self.assertEqual(result['metadata']['authentication'],'anonymous_token_false')
        result = self.plan('mlx.convert',model=str(self.model),output_dir=str(self.root/'out'))['plan']
        self.assertEqual(result['env']['HF_HUB_OFFLINE'],'1')
        self.assertIn('--hf-path',result['command'])

    def test_mlx_convert_plans_native_mixed_presets_with_explicit_group_size(self):
        output = self.root / 'mixed'
        cases = (
            ('mixed_2_6', [2, 6], {}),
            ('mixed_3_4', [3, 4], {'group_size': 32}),
            ('mixed_3_6', [3, 6], {'group_size': 64}),
            ('mixed_4_6', [4, 6], {'group_size': 128}),
        )
        for preset, bits, options in cases:
            with self.subTest(preset=preset):
                result = self.plan('mlx.convert', model=str(self.model), output_dir=str(output),
                                   quant_predicate=preset, **options)
                self.assertTrue(result['ok'], result)
                plan = result['plan']
                group_size = options.get('group_size', 64)
                self.assertEqual(plan['command'], [str(self.binary), '--hf-path', str(self.model),
                                                   '--mlx-path', str(output), '-q', '--quant-predicate',
                                                   preset, '--q-group-size', str(group_size)])
                self.assertEqual(plan['metadata']['quantization'], {
                    'mode': 'affine', 'bits': bits, 'group_size': group_size, 'quant_predicate': preset,
                })
                self.assertTrue(plan['metadata']['quantizes_output_weights'])
                self.assertFalse(plan['metadata']['quality_verified'])
        self.assertFalse(output.exists())

    def test_mlx_convert_refuses_unsupported_or_conflicting_quantization(self):
        for options in ({'quant_predicate': 'mixed_2_4'}, {'quant_predicate': 'module.custom'},
                        {'quant_predicate': 'mixed_3_4', 'bits': 4}, {'group_size': 64},
                        {'bits': 4, 'group_size': 16}, {'bits': 4, 'group_size': 64.0},
                        {'quant_predicate': 'mixed_3_4', 'group_size': True}):
            with self.subTest(options=options):
                result = self.plan('mlx.convert', model=str(self.model), output_dir=str(self.root / 'mixed'), **options)
                self.assertEqual(result['error']['code'], 'invalid_parameters')

    def test_mlx_convert_preserves_existing_command_shapes(self):
        output = self.root / 'converted'
        prefix = [str(self.binary), '--hf-path', str(self.model), '--mlx-path', str(output)]
        for options, expected in (({}, []), ({'bits': 4}, ['-q', '--q-bits', '4']),
                                  ({'bits': 4, 'group_size': 128}, ['-q', '--q-bits', '4', '--q-group-size', '128'])):
            with self.subTest(options=options):
                result = self.plan('mlx.convert', model=str(self.model), output_dir=str(output), **options)
                self.assertTrue(result['ok'], result)
                self.assertEqual(result['plan']['command'], prefix + expected)

    def test_trl_config_rejects_remote_models_and_unknown_code(self):
        valid = load_config(self.config,[str(self.root)])
        self.assertNotIn('lora',valid)
        for value in ({'model':'org/model','train_file':str(self.train)},
                      {'model':str(self.model),'train_file':str(self.train),'trust_remote_code':True}):
            self.config.write_text(json.dumps(value))
            with self.assertRaises(ConfigurationError):
                load_config(self.config,[str(self.root)])

    def test_public_download_runner_passes_token_false(self):
        from ai_lab_core.backend_task import main
        download = Mock()
        with patch.dict(sys.modules, {'huggingface_hub':SimpleNamespace(snapshot_download=download)}), \
             patch.object(sys,'argv',['task','hf-download','--repo-id','org/model',
                                     '--output-dir',str(self.root/'out')]), \
             patch('builtins.print'):
            self.assertEqual(main(),0)
        self.assertIs(download.call_args.kwargs['token'],False)

    def test_sft_runner_calls_real_training_interface_without_implicit_peft(self):
        from ai_lab_core.recipes.trl_sft import main
        model_loader, tokenizer_loader, trainer_class = Mock(), Mock(), Mock()
        model_loader.from_pretrained.return_value = object()
        tokenizer = tokenizer_loader.from_pretrained.return_value
        modules = {
            'datasets':SimpleNamespace(load_dataset=Mock(return_value={'train':['fixture']})),
            'transformers':SimpleNamespace(AutoModelForCausalLM=model_loader,AutoTokenizer=tokenizer_loader),
            'trl':SimpleNamespace(SFTConfig=Mock(),SFTTrainer=trainer_class),
        }
        with patch.dict(sys.modules,modules), patch.object(sys,'argv',[
                'recipe','--config',str(self.config),'--output-dir',str(self.root/'out'),
                '--workspace-root',str(self.root),'--allowed-root',str(self.root)]),patch('builtins.print'):
            self.assertEqual(main(),0)
        self.assertIs(model_loader.from_pretrained.call_args.kwargs['token'],False)
        self.assertIs(model_loader.from_pretrained.call_args.kwargs['trust_remote_code'],False)
        self.assertIsNone(trainer_class.call_args.kwargs['peft_config'])
        trainer_class.return_value.train.assert_called_once()
        trainer_class.return_value.save_model.assert_called_once_with(str(self.root/'out'))
        tokenizer.save_pretrained.assert_called_once()

    def test_mlx_generation_prompt_cannot_be_parsed_as_an_option(self):
        plan = self.plan('mlx.generate',model=str(self.model),prompt='--trust-remote-code\nhello',
                         max_tokens=8,adapter=str(self.data))['plan']
        self.assertIn('--prompt=--trust-remote-code\nhello',plan['command'])
        self.assertNotIn('--trust-remote-code',plan['command'])
        self.assertIn('--adapter-path',plan['command'])
        self.assertEqual(plan['outputs'],[])
        for count in (0,2049,True):
            self.assertFalse(self.plan('mlx.generate',model=str(self.model),prompt='hi',max_tokens=count)['ok'])
        self.assertFalse(self.plan('mlx.generate',model=str(self.weight),prompt='hi')['ok'])

    def test_hermes_export_requires_bounded_id_workspace_output_and_redaction(self):
        plan = self.plan('hermes.export',session_id='session_123456',
                         output_file=str(self.root/'export.jsonl'))['plan']
        self.assertEqual(plan['command'][1:],['sessions','export',str(self.root/'export.jsonl'),
                        '--session-id','session_123456','--format','jsonl','--redact'])
        self.assertNotIn('--upload',plan['command'])
        for session in ('*','--all','a'*129,'../other','abc'):
            self.assertFalse(self.plan('hermes.export',session_id=session,
                                      output_file=str(self.root/'export.jsonl'))['ok'])
        self.assertEqual(self.plan('hermes.export',session_id='session_123456',
                                  output_file=str(self.config))['error']['code'],'output_exists')
        self.assertFalse(self.plan('hermes.export',session_id='session_123456',
                                  output_file='/outside/export.jsonl')['ok'])
        self.assertFalse(self.plan('hermes.export',session_id='session_123456',
                                  output_file=str(self.root/'export.jsonl'),redact=False)['ok'])

    def test_hf_revision_sha_and_restricted_files_forward_to_anonymous_sdk(self):
        sha = 'a'*40
        plan = self.plan('hf.download',repo_id='org/model',revision=sha,
                         allow_patterns=['*.safetensors','tokenizer.json'],output_dir=str(self.root/'out'))['plan']
        self.assertIn(sha,plan['command'])
        self.assertEqual(plan['command'].count('--allow-pattern'),2)
        for patterns in (['../secret'],['/absolute'],[],['--token']):
            self.assertFalse(self.plan('hf.download',repo_id='org/model',allow_patterns=patterns,
                                      output_dir=str(self.root/'out'))['ok'])


if __name__ == '__main__':
    unittest.main()
