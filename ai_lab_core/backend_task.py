"""Small subprocess entrypoints for optional SDK operations."""
from __future__ import annotations
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['hf-download'])
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--allow-pattern',action='append',default=None)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, revision=args.revision,
                          local_dir=args.output_dir, token=False, allow_patterns=args.allow_pattern)
    except ImportError:
        print(json.dumps({'ok':False,'error':{'code':'dependency_missing','message':'Hugging Face SDK is required.'}}))
        return 2
    except Exception:
        print(json.dumps({'ok':False,'error':{'code':'download_failed','message':'Public model download failed.'}}))
        return 1
    print(json.dumps({'ok':True,'path':args.output_dir}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
