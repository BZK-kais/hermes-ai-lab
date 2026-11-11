import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_lab_core.transfers import plan_operation,catalog,HARD_MAX_BYTES
from ai_lab_core.transfer_task import execute_transfer
from ai_lab_core.credentials import secret_scope


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.c={'id':'archive','type':'s3','bucket':'test-bucket','region':'eu-west-3','credential_ref':'hermes:AI_LAB_S3_ACCESS_KEY_ID'}
        self.settings={'workspace_root':str(self.root),'integrations':[self.c],'transfer_max_bytes':100}
        self.target=self.root/'model.bin'

    def run_transfer(self,op='storage.download',c=None):
        return execute_transfer(op,c or self.c,'artifacts/model.bin',str(self.target),str(self.root),100)

    def fake_s3(self,payload=b'artifact',advertised=None,fail=False,race=False):
        owner=self
        class Client:
            calls=[]
            def head_object(self,**kwargs):
                self.calls.append(('head',kwargs))
                return {'ContentLength':len(payload) if advertised is None else advertised,'VersionId':'version-1'}
            def download_file(self,bucket,key,filename,**kwargs):
                self.calls.append(('download',kwargs))
                Path(filename).write_bytes(payload)
                if race:
                    owner.target.write_bytes(b'other writer')
                if fail:
                    raise RuntimeError('SECRET https://signed.invalid?token=secret')
            def upload_file(self,filename,bucket,key):
                self.calls.append(('upload',Path(filename).read_bytes(),bucket,key))
        client=Client()
        return client

    def test_metadata_precedes_atomic_download_and_version_is_pinned(self):
        client=self.fake_s3()
        with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
            result=self.run_transfer()
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.target.read_bytes(),b'artifact')
        self.assertEqual([c[0] for c in client.calls],['head','download'])
        self.assertEqual(client.calls[1][1]['ExtraArgs'],{'VersionId':'version-1'})
        self.assertEqual(list(self.root.iterdir()),[self.target])

    def test_size_limit_stops_before_transfer_and_mismatch_removes_partial(self):
        for advertised in (101,-1,None):
            client=self.fake_s3(advertised=advertised if advertised is not None else 'unknown')
            with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
                result=self.run_transfer()
            self.assertFalse(result['ok'])
            self.assertEqual(len(client.calls),1)
            self.assertEqual(list(self.root.iterdir()),[])
        client=self.fake_s3(payload=b'123',advertised=2)
        with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
            result=self.run_transfer()
        self.assertEqual(result['error']['code'],'size_mismatch')
        self.assertEqual(list(self.root.iterdir()),[])

    def test_failure_cleanup_and_concurrent_destination_is_never_overwritten(self):
        client=self.fake_s3(fail=True)
        with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
            result=self.run_transfer()
        self.assertNotIn('SECRET',json.dumps(result))
        self.assertNotIn('signed.invalid',json.dumps(result))
        self.assertEqual(list(self.root.iterdir()),[])
        client=self.fake_s3(race=True)
        with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
            result=self.run_transfer()
        self.assertEqual(result['error']['code'],'output_exists')
        self.assertEqual(self.target.read_bytes(),b'other writer')
        self.assertEqual(list(self.root.iterdir()),[self.target])

    def test_upload_streams_existing_file_only_when_explicit(self):
        self.target.write_bytes(b'local bytes')
        client=self.fake_s3()
        with patch('ai_lab_core.transfer_task._s3_client',return_value=client):
            self.assertTrue(self.run_transfer('storage.upload')['ok'])
        self.assertEqual(client.calls,[('upload',b'local bytes','test-bucket','artifacts/model.bin')])
        self.assertEqual(self.run_transfer()['error']['code'],'output_exists')

    def test_hf_pins_revision_and_never_returns_resolved_credential(self):
        calls=[]
        class Api:
            def __init__(self,token):calls.append(('token',token))
            def model_info(self,repo):return SimpleNamespace(sha='a'*40)
            def get_paths_info(self,repo,paths,**kwargs):
                calls.append(('metadata',kwargs))
                return [SimpleNamespace(size=3)]
        def download(**kwargs):
            calls.append(('download',kwargs))
            path=Path(kwargs['local_dir'])/kwargs['filename']
            path.parent.mkdir(parents=True)
            path.write_bytes(b'hf!')
            return str(path)
        c={'id':'hf','type':'hf_storage','repo_id':'org/model','credential_ref':'hermes:AI_LAB_HF_TOKEN'}
        module=SimpleNamespace(HfApi=Api,hf_hub_download=download)
        with patch.dict(sys.modules,{'huggingface_hub':module}),secret_scope({'AI_LAB_HF_TOKEN':'sensitive-fixture'}),patch('ai_lab_core.transfer_task._guard_hf_http'):
            result=self.run_transfer(c=c)
        self.assertTrue(result['ok'],result)
        self.assertEqual(self.target.read_bytes(),b'hf!')
        self.assertEqual(calls[2][1]['revision'],'a'*40)
        self.assertNotIn('sensitive-fixture',json.dumps(result))
        self.target.unlink()
        with patch.dict(sys.modules,{'huggingface_hub':module}),secret_scope({}):
            result=self.run_transfer(c=c)
        self.assertEqual(result['error']['code'],'credential_unavailable')
        self.assertEqual(list(self.root.iterdir()),[])

    def test_plans_are_confined_and_mark_live_hermes_execution(self):
        params={'connection_id':'archive','key':'a/model.bin','local_path':str(self.target)}
        with patch('ai_lab_core.transfers._installed',return_value=True):
            plan=plan_operation('storage.download',params,self.settings)['plan']
            self.assertEqual(plan['timeout_seconds'],7200)
            self.assertNotIn('command',plan)
            self.assertNotIn('hermes:',json.dumps(plan))
            self.assertEqual(plan['metadata']['execution_scope'],'live_hermes_request')
            self.assertEqual(len(catalog(self.settings)['operations']),2)
            for bad in ('../escape','/absolute','-flag','a//b'):
                self.assertFalse(plan_operation('storage.download',{**params,'key':bad},self.settings)['ok'])
            self.assertFalse(plan_operation('storage.download',{**params,'extra':'no'},self.settings)['ok'])
            with tempfile.TemporaryDirectory() as outside:
                (self.root/'outside').symlink_to(outside,target_is_directory=True)
                self.assertFalse(plan_operation('storage.download',{**params,'local_path':str(self.root/'outside'/'file')},self.settings)['ok'])
            self.settings['transfer_max_bytes']=HARD_MAX_BYTES+1
            self.assertFalse(plan_operation('storage.download',params,self.settings)['ok'])
        with patch('ai_lab_core.transfers._installed',return_value=False):
            self.settings.pop('transfer_max_bytes')
            result=plan_operation('storage.download',params,self.settings)
            self.assertEqual(result['error']['code'],'dependency_missing')


if __name__=='__main__':unittest.main()
