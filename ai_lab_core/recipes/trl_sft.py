"""Real optional TRL SFT recipe for local models and local JSON/JSONL datasets.

JSON example: {"model":"/models/base", "train_file":"/data/train.jsonl",
              "max_steps":10, "batch_size":1, "max_length":512, "lora":false}
No engine dependency is imported while cataloging or planning jobs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


class ConfigurationError(ValueError):
    pass


def load_config(path, allowed_roots):
    config_path = Path(path).resolve(strict=True)
    if config_path.stat().st_size > 1_000_000:
        raise ConfigurationError()
    if config_path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        config = yaml.safe_load(config_path.read_text())
    else:
        config = json.loads(config_path.read_text())
    allowed = {'model','train_file','eval_file','max_steps','batch_size','max_length',
               'learning_rate','lora','lora_rank','lora_alpha','gradient_accumulation_steps'}
    if not isinstance(config,dict) or set(config) - allowed or not {'model','train_file'} <= set(config):
        raise ConfigurationError()
    roots = [Path(p).resolve(strict=True) for p in allowed_roots]
    for key in ('model','train_file','eval_file'):
        if key not in config:
            continue
        if not isinstance(config[key],str) or not Path(config[key]).is_absolute():
            raise ConfigurationError()
        candidate = Path(config[key]).resolve(strict=True)
        if not any(candidate.is_relative_to(root) for root in roots):
            raise ConfigurationError()
        if (key == 'model' and not candidate.is_dir()) or (key != 'model' and (
                not candidate.is_file() or candidate.suffix.lower() not in ('.json','.jsonl'))):
            raise ConfigurationError()
        config[key] = str(candidate)
    for key, default, upper in [('max_steps',100,1000000),('batch_size',1,1024),
            ('max_length',512,1048576),('gradient_accumulation_steps',1,100000),
            ('lora_rank',16,1024),('lora_alpha',32,4096)]:
        value = config.get(key,default)
        if isinstance(value,bool) or not isinstance(value,int) or not 1 <= value <= upper:
            raise ConfigurationError()
        config[key] = value
    rate = config.get('learning_rate',0.00002)
    if isinstance(rate,bool) or not isinstance(rate,(float,int)) or not 0 < rate <= 0.1:
        raise ConfigurationError()
    config['learning_rate'] = rate
    if not isinstance(config.get('lora',False),bool):
        raise ConfigurationError()
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workspace-root',required=True)
    parser.add_argument('--allowed-root',action='append',required=True)
    args = parser.parse_args()
    try:
        cfg = load_config(args.config,args.allowed_root)
        out = Path(args.output_dir)
        workspace = Path(args.workspace_root).resolve(strict=True)
        if not out.is_absolute() or out.exists() or out.is_symlink():
            raise ConfigurationError()
        out = out.resolve()
        if out == workspace or not out.is_relative_to(workspace):
            raise ConfigurationError()
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
        tokenizer = AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,
                                                  trust_remote_code=False,token=False)
        model = AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,
                                                    trust_remote_code=False,token=False)
        data_files = {'train':cfg['train_file']}
        if 'eval_file' in cfg:
            data_files['validation'] = cfg['eval_file']
        dataset = load_dataset('json',data_files=data_files)
        peft_config = None
        if cfg.get('lora',False):
            from peft import LoraConfig
            peft_config = LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],
                                     task_type='CAUSAL_LM')
        training_args = SFTConfig(output_dir=str(out),max_steps=cfg['max_steps'],
            per_device_train_batch_size=cfg['batch_size'],max_length=cfg['max_length'],
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            learning_rate=cfg['learning_rate'],report_to=[],save_strategy='no')
        trainer = SFTTrainer(model=model,args=training_args,processing_class=tokenizer,
            train_dataset=dataset['train'],eval_dataset=dataset.get('validation'),peft_config=peft_config)
        trainer.train()
        if 'validation' in dataset:
            trainer.evaluate()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
    except ImportError:
        print(json.dumps({'ok':False,'error':{'code':'dependency_missing',
                                           'message':'The configured Python needs TRL, Transformers, datasets and optional PEFT/YAML.'}}))
        return 2
    except (ValueError,OSError,TypeError,RuntimeError):
        print(json.dumps({'ok':False,'error':{'code':'training_failed',
                                           'message':'SFT configuration, runtime or training failed.'}}))
        return 1
    print(json.dumps({'ok':True,'output_dir':str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
