"""Optional SDK transfer worker. Artifact bytes and credentials never enter results."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai_lab_core.credentials import CredentialError,resolve_secret
from ai_lab_core.providers import validate_connection
from ai_lab_core.transfers import HARD_MAX_BYTES,valid_key,local_file,failure


class _Discard:
    def write(self,value):
        return len(value)
    def flush(self):
        pass


class TransferError(Exception):
    def __init__(self,code):
        self.code=code


def _size(value,cap):
    if isinstance(value,bool) or not isinstance(value,int) or value < 0:
        raise TransferError('size_unknown')
    if value > cap:
        raise TransferError('size_limit')
    return value


def _hf_token(c):
    if c.get('credential_ref') == 'native':
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise CredentialError('Credential unavailable.')
        return token
    return resolve_secret(c['credential_ref'])


def _transport(c,key,local,download,cap,scratch):
    kind = c['type']
    if kind == 'hf_storage':
        from huggingface_hub import HfApi,hf_hub_download
        token = _hf_token(c)
        api = HfApi(token=token)
        if download:
            revision = api.model_info(c['repo_id']).sha
            info = api.get_paths_info(c['repo_id'],[key],revision=revision,repo_type='model')
            if len(info)!=1:
                raise TransferError('not_found')
            expected = _size(getattr(info[0],'size',None),cap)
            downloaded = Path(hf_hub_download(repo_id=c['repo_id'],filename=key,revision=revision,
                                             repo_type='model',token=token,local_dir=str(scratch)))
            if not downloaded.resolve().is_relative_to(scratch.resolve()):
                raise TransferError('invalid_download_path')
            os.replace(downloaded,local)
            return expected
        api.upload_file(path_or_fileobj=str(local),path_in_repo=key,repo_id=c['repo_id'],repo_type='model',token=token)
        return local.stat().st_size
    if kind == 'gcs':
        from google.cloud import storage
        blob = storage.Client(project=c['project']).bucket(c['bucket']).blob(key)
        if download:
            blob.reload(timeout=60)
            expected = _size(blob.size,cap)
            blob.download_to_filename(str(local),timeout=120,if_generation_match=blob.generation)
            return expected
        blob.upload_from_filename(str(local),timeout=120)
        return local.stat().st_size
    if kind == 's3':
        import boto3
        client = boto3.client('s3',region_name=c['region'])
        if download:
            head = client.head_object(Bucket=c['bucket'],Key=key)
            expected = _size(head.get('ContentLength'),cap)
            extra = {'VersionId':head['VersionId']} if head.get('VersionId') else {}
            client.download_file(c['bucket'],key,str(local),**({'ExtraArgs':extra} if extra else {}))
            return expected
        client.upload_file(str(local),c['bucket'],key)
        return local.stat().st_size
    raise TransferError('invalid_config')


def execute_transfer(operation,connection,key,local_path,workspace_root,max_bytes=HARD_MAX_BYTES):
    if operation not in ('storage.download','storage.upload') or not valid_key(key):
        return failure('invalid_parameters','Transfer parameters are invalid.')
    validation = validate_connection(connection)
    if not validation.get('ok'):
        return validation
    if connection.get('type') not in ('hf_storage','gcs','s3'):
        return failure('invalid_config','A storage connection is required.')
    if isinstance(max_bytes,bool) or not isinstance(max_bytes,int) or not 1 <= max_bytes <= HARD_MAX_BYTES:
        return failure('invalid_parameters','Transfer size limit is invalid.')
    download = operation == 'storage.download'
    temporary = None
    scratch = None
    try:
        target,root = local_file(local_path,workspace_root,download)
        if download:
            fd,name = tempfile.mkstemp(prefix='.'+target.name+'-',suffix='.part',dir=str(target.parent))
            os.close(fd)
            temporary = Path(name)
            scratch = Path(tempfile.mkdtemp(prefix='.ai-lab-transfer-',dir=str(target.parent)))
        else:
            _size(target.stat().st_size,max_bytes)
        # SDKs may print progress or provider exceptions: withhold both streams.
        with contextlib.redirect_stdout(_Discard()),contextlib.redirect_stderr(_Discard()):
            expected = _transport(connection,key,temporary if download else target,download,max_bytes,scratch)
        if download:
            actual = _size(temporary.stat().st_size,max_bytes)
            if actual != expected:
                raise TransferError('size_mismatch')
            # Revalidate the parent after transfer, and publish without clobber races.
            current,_ = local_file(str(target),str(root),True)
            if current != target:
                raise TransferError('path_not_allowed')
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            os.link(temporary,target)
            temporary.unlink()
            temporary = None
        else:
            actual = _size(target.stat().st_size,max_bytes)
        return {'ok':True,'operation':operation,'connection_id':connection['id'],'key':key,
                'local_path':str(target),'bytes':actual,'atomic_publication':download}
    except CredentialError:
        return failure('credential_unavailable','The configured credential is unavailable.')
    except ImportError:
        return failure('dependency_missing','The optional storage SDK is unavailable.')
    except FileExistsError:
        return failure('output_exists','Download output already exists.')
    except TransferError as exc:
        return failure(exc.code,'Storage transfer did not complete.')
    except (ValueError,TypeError,OSError,RuntimeError):
        return failure('transfer_failed','Storage transfer failed or the local path is invalid.')
    except Exception:
        return failure('provider_error','Storage provider transfer failed.')
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
        if scratch is not None:
            shutil.rmtree(scratch,ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=['storage.download','storage.upload'])
    parser.add_argument('--connection-json',required=True)
    parser.add_argument('--key',required=True)
    parser.add_argument('--local-path',required=True)
    parser.add_argument('--workspace-root',required=True)
    parser.add_argument('--max-bytes',required=True,type=int)
    args=parser.parse_args()
    try:
        connection=json.loads(args.connection_json)
    except (ValueError,TypeError):
        result=failure('invalid_config','Storage configuration is invalid.')
    else:
        result=execute_transfer(args.operation,connection,args.key,args.local_path,args.workspace_root,args.max_bytes)
    print(json.dumps(result))
    return 0 if result.get('ok') else 1


if __name__=='__main__':
    raise SystemExit(main())
