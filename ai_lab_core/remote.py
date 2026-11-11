"""SSH access to one explicitly configured host and one trusted local runner script.

API callers select a JSON plan file, never a shell command or Python program.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from .providers import failure, _artifact, _relative, _slug

ACTION_SCHEMAS = {
    "probe": {},
    "exec": {"plan_path": "workspace-relative JSON file", "run_id": "unique identifier"},
    "status": {"run_id": "identifier"},
    "cancel": {"run_id": "identifier"},
}
_FIELDS = {"id", "type", "host", "user", "port", "python", "workspace", "remote_workspace", "script"}
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")
_REMOTE_PATH = re.compile(r"/[A-Za-z0-9_./-]{1,512}\Z")
MAX_BYTES = 65536


def validate_connection(c: dict) -> dict:
    if not isinstance(c, dict) or set(c) - _FIELDS or c.get("type") != "ssh" or not _slug(c.get("id")):
        return failure("invalid_config", "Invalid SSH connection configuration.")
    if not isinstance(c.get("host"), str) or not _HOST.fullmatch(c["host"]) or ".." in c["host"]:
        return failure("invalid_config", "Configure a safe SSH hostname or IPv4 address.")
    if "user" in c and (not isinstance(c["user"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", c["user"])):
        return failure("invalid_config", "Invalid SSH user.")
    port = c.get("port", 22)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return failure("invalid_config", "Invalid SSH port.")
    python = c.get("python", "python3")
    if not isinstance(python, str) or not re.fullmatch(r"(?:/[A-Za-z0-9_./-]+|python3(?:\.[0-9]+)?)", python) or ".." in python:
        return failure("invalid_config", "Invalid remote Python executable.")
    work = c.get("remote_workspace", "")
    if work and (not isinstance(work, str) or not _REMOTE_PATH.fullmatch(work) or any(x in {"", ".", ".."} for x in work[1:].split("/"))):
        return failure("invalid_config", "Invalid remote workspace.")
    if "workspace" in c and (not isinstance(c["workspace"], str) or not Path(c["workspace"]).is_absolute()):
        return failure("invalid_config", "Configure an absolute local workspace.")
    if "script" in c and (not _relative(c["script"]) or not c["script"].endswith(".py")):
        return failure("invalid_config", "Configure a workspace-relative trusted Python script.")
    return {"ok": True, "id": c["id"], "type": "ssh", "state": "configured", "actions": ACTION_SCHEMAS}


def _destination(c):
    return (c["user"] + "@" if c.get("user") else "") + c["host"]


def _options(c, scp=False):
    return ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
            "-o", "ClearAllForwardings=yes", "-o", "PermitLocalCommand=no",
            "-P" if scp else "-p", str(c.get("port", 22))]


def _run_process(argv, *, payload=None, timeout=20):
    # No shell, no private key arguments, no raw stdout/stderr returned.
    with tempfile.TemporaryFile() as output:
        completed = subprocess.run(argv, input=payload, stdout=output,
                                   stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        if completed.returncode:
            raise RuntimeError("SSH operation failed.")
        if output.tell() > MAX_BYTES:
            raise RuntimeError("Remote output exceeded the limit.")
        output.seek(0)
        return output.read(MAX_BYTES)



# Trusted fixed control program. The uploaded runner is a configuration decision,
# while the API plan is data. Never evaluate code from the plan here.
_REMOTE_CONTROL = r'''
import json,os,pathlib,signal,subprocess,sys
p=json.load(sys.stdin)
a=p['action']
if a=='probe':
 print(json.dumps({'ok':True,'python_available':True,'platform':sys.platform}))
 raise SystemExit
root=pathlib.Path(p['root'])
job=root/'.ai-lab-remote'/p['run_id']
if a=='prepare':
 job.mkdir(parents=True,exist_ok=False)
 print(json.dumps({'ok':True}))
elif a=='exec':
 state=job/'state.json'
 if state.exists():
  print(json.dumps({'ok':False,'error':{'code':'reconciliation_required','message':'Run already submitted.'}}))
  raise SystemExit
 worker="""import json,pathlib,subprocess,sys,time
j=pathlib.Path(sys.argv[1])
f=j/'state.json'
while not (j/'start').exists(): time.sleep(0.01)
with (j/'output.log').open('ab') as out:
 r=subprocess.run([sys.executable,str(j/'runner.py'),'--plan',str(j/'plan.json')],stdout=out,stderr=out)
f.write_text(json.dumps({'state':'completed' if r.returncode==0 else 'failed','exit_code':r.returncode}))
"""
 with (job/'launcher.log').open('ab') as log:
  child=subprocess.Popen([sys.executable,'-c',worker,str(job)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 state.write_text(json.dumps({'state':'running','pid':child.pid}))
 (job/'start').touch()
 print(json.dumps({'ok':True,'state':'submitted'}))
else:
 state=json.loads((job/'state.json').read_text())
 if a=='cancel' and state.get('state')=='running':
  cmd=pathlib.Path('/proc')/str(state['pid'])/'cmdline'
  if not cmd.exists() or str(job).encode() not in cmd.read_bytes().split(b'\x00'):
   raise RuntimeError('Cannot establish ownership of the remote process.')
  try: os.killpg(state['pid'],signal.SIGTERM)
  except ProcessLookupError: pass
  state={'state':'cancel_requested'}
  (job/'state.json').write_text(json.dumps(state))
 print(json.dumps({'ok':True,'state':state.get('state'),'exit_code':state.get('exit_code')}))
'''


def _control(c, action, run_id=None):
    command = " ".join(shlex.quote(x) for x in [c.get("python", "python3"), "-c", _REMOTE_CONTROL])
    payload = {"action": action, "root": c.get("remote_workspace"), "run_id": run_id}
    raw = _run_process(["ssh", *_options(c), "-T", _destination(c), command],
                       payload=json.dumps(payload).encode())
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
        raise ValueError("Invalid remote response.")
    # Only fixed state enums, booleans and bounded exit code cross the boundary.
    if not data["ok"]:
        return failure("reconciliation_required", "Run exists; inspect its state before another submission.")
    result = {"ok": True}
    if action == "probe":
        result["python_available"] = data.get("python_available") is True
        result["platform"] = data.get("platform") if data.get("platform") in {"linux", "darwin", "win32"} else "unknown"
    elif data.get("state") in {"submitted", "running", "completed", "failed", "cancel_requested"}:
        result["state"] = data["state"]
        if isinstance(data.get("exit_code"), int):
            result["exit_code"] = data["exit_code"]
    elif action != "prepare":
        raise ValueError("Invalid remote state.")
    return result


def probe_connection(c: dict) -> dict:
    return remote_action(c, "probe", {})


def remote_action(c: dict, action: str, params: dict) -> dict:
    checked = validate_connection(c)
    if not checked["ok"]:
        return checked
    if not isinstance(action, str) or action not in ACTION_SCHEMAS or not isinstance(params, dict) or set(params) != set(ACTION_SCHEMAS[action]):
        return failure("invalid_input", "Invalid remote action or parameters.")
    try:
        if not shutil.which("ssh"):
            return failure("dependency_missing", "SSH is not installed.")
        run_id = params.get("run_id")
        if action != "probe" and (not _slug(run_id) or not c.get("remote_workspace")):
            return failure("configuration_required", "Configure a remote workspace and supply a safe run identifier.")
        if action == "exec":
            if not shutil.which("scp") or not c.get("script"):
                return failure("configuration_required", "Configure a trusted runner script and install SCP.")
            script = _artifact(c, c["script"])
            plan = _artifact(c, params.get("plan_path"))
            if Path(plan).suffix != ".json" or Path(plan).stat().st_size > MAX_BYTES:
                return failure("invalid_input", "Use a bounded JSON plan file.")
            # Validate data before any network action; no arbitrary shell or script body API.
            parsed = json.loads(Path(plan).read_text())
            if not isinstance(parsed, dict):
                return failure("invalid_input", "The plan must be a JSON object.")
            root = Path(c["workspace"]).resolve(strict=True)
            intents = root / ".ai-lab" / "remote-intents"
            if not intents.resolve().is_relative_to(root):
                return failure("invalid_config", "Invalid local intent directory.")
            intents.mkdir(parents=True, exist_ok=True)
            if not intents.resolve().is_relative_to(root):
                return failure("invalid_config", "Invalid local intent directory.")
            digest = hashlib.sha256((c["id"] + ":" + run_id).encode()).hexdigest()
            try:
                with (intents / (digest + ".json")).open("x") as handle:
                    json.dump({"connection_id": c["id"], "run_id": run_id, "state": "intent_recorded"}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                return failure("reconciliation_required", "Run intent already exists; inspect remote state before resubmission.")
            prepared = _control(c, "prepare", run_id)
            if not prepared["ok"]:
                return prepared
            directory = c["remote_workspace"] + "/.ai-lab-remote/" + run_id
            for source, name in ((script, "runner.py"), (plan, "plan.json")):
                _run_process(["scp", *_options(c, scp=True), "--", source,
                              _destination(c) + ":" + directory + "/" + name], timeout=30)
        result = _control(c, action, run_id)
        return result | {"connection_id": c["id"], "type": "ssh", "actions": ACTION_SCHEMAS,
                         "logs_exposed": False, "host_key_policy": "strict"}
    except subprocess.TimeoutExpired:
        return failure("timeout", "SSH timed out; remote work may still be running. Inspect its state before resubmission.")
    except Exception:
        return failure("remote_error", "SSH operation failed; no remote output or credential details are exposed.")


provider_action = remote_action
probe_remote = probe_connection
validate_remote = validate_connection
