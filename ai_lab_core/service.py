"""Shared implementation behind the Hermes tools and the local CLI."""

from . import __version__
from .config import load_settings
from .endpoint import failure, probe
from .hub import get_model_info
from .local import hardware, inventory_models, runtime_inventory


def status() -> dict:
    try:
        from .integration_config import load_integrations
        settings = load_integrations()
    except (OSError, ValueError):
        return failure("invalid_config", "AI Lab connection settings could not be read.")
    endpoints = [{"name": item["name"], **probe(item["base_url"], item["kind"])} for item in settings["endpoints"]]
    return {"ok": True, "version": __version__, "hardware": hardware(), "runtimes": runtime_inventory(settings["runtime_bin_dirs"]),
            "endpoints": endpoints, "models": inventory_models(settings["model_roots"], limit=20),
            "capabilities": {"local_inventory": "implemented", "public_hub_metadata": "implemented",
                             "local_endpoint_probe": "implemented", "named_operation_planning": "implemented",
                             "persistent_jobs": "implemented", "provider_adapters": "implemented",
                             "remote_hardware_verification": "not_performed"},
            "note": "Installed binaries and API listings do not verify model inference."}


def list_local_models(limit: int = 50) -> dict:
    try:
        from .integration_config import load_integrations
        return inventory_models(load_integrations()["model_roots"], limit=limit)
    except (OSError, ValueError):
        return failure("invalid_config", "AI Lab connection settings could not be read.")


def hf_model_info(repo_id: str, revision: str = "main", file_limit: int = 30) -> dict:
    return get_model_info(repo_id, revision, file_limit)


def probe_endpoint(base_url: str, kind: str = "llama_cpp") -> dict:
    return probe(base_url, kind)


def connections():
    from .integration_config import load_integrations
    from .hardware_profiles import detect_profile
    settings = load_integrations()
    selected = [{"id": c["id"], "type": c["type"], "state": "configured_not_probed"} for c in settings["integrations"]]
    selected.extend({"id": e["name"], "type": "inference", "kind": e["kind"], "state": "configured_not_probed"} for e in settings["endpoints"])
    types = {"runpod", "hf_jobs", "hf_endpoints", "hf_storage", "gcs", "s3", "ssh", "mcp", "inference"}
    return {"ok": True, "hardware": detect_profile(), "connections": selected,
            "unconfigured_adapter_types": sorted(types - {c["type"] for c in selected}),
            "note": "Configured references do not prove authorization, device access or successful execution."}


def connection_probe(connection_id):
    from .integration_config import find_connection
    c = find_connection(connection_id)
    if c["type"] == "inference":
        from .inference import probe_inference
        return probe_inference(c)
    if c["type"] == "mcp":
        from .mcp_client import request
        return request(c)
    if c["type"] == "ssh":
        from .remote import probe_connection
    else:
        from .providers import probe_connection
    return probe_connection(c)


def operations():
    from .backends import catalog
    from .transfers import catalog as transfer_catalog
    from .integration_config import load_integrations
    settings = load_integrations()
    result = catalog(settings)
    result["operations"].extend(transfer_catalog(settings)["operations"])
    from .providers import ACTION_SCHEMAS
    result["provider_actions"] = ACTION_SCHEMAS
    return result


def plan(operation, parameters):
    from .backends import plan_operation
    from .integration_config import load_integrations
    if operation in {"storage.download", "storage.upload"}:
        from .transfers import plan_operation
    return plan_operation(operation, parameters, load_integrations())


def run(operation, parameters):
    from pathlib import Path
    import sys

    from .backends import plan_operation
    from .integration_config import load_integrations
    from .jobs import submit
    settings = load_integrations()
    if operation in {"storage.download", "storage.upload"}:
        from .transfers import run_transfer
        return run_transfer(operation, parameters, settings)
    result = plan_operation(operation, parameters, settings)
    if not result.get("ok"):
        return result
    job_plan = result["plan"]
    if job_plan["backend"] == "mlx":
        runner = Path(__file__).with_name("device_runner.py")
        job_plan = {**job_plan, "command": [sys.executable, "-I", "-B", str(runner), *job_plan["command"]]}
    return submit(job_plan, state_dir=settings["state_dir"])


def job(action="list", job_id=None, limit=20):
    from . import jobs
    from .integration_config import load_integrations
    state = load_integrations()["state_dir"]
    if action == "list":
        return jobs.list_jobs(limit=limit, state_dir=state)
    if action == "reconcile":
        return jobs.reconcile(state_dir=state)
    if not job_id:
        return failure("invalid_input", "A job identifier is required.")
    if action == "logs":
        return jobs.job_logs(job_id, limit=4000 if limit == 20 else limit, state_dir=state)
    method = {"get": jobs.get_job, "status": jobs.get_job, "cancel": jobs.cancel_job}.get(action)
    if not method:
        return failure("invalid_input", "Unknown job action.")
    return method(job_id, state_dir=state)


def provider_action(connection_id, action, parameters):
    from .integration_config import find_connection
    from .operator_policy import provider_denial
    c = find_connection(connection_id)
    denial = provider_denial(c['type'], action)
    if denial is not None:
        return denial
    if c["type"] == "ssh":
        from .remote import provider_action as invoke
    elif c["type"] == "mcp":
        from .mcp_client import request
        return request(c, tool=action, arguments=parameters)
    else:
        from .providers import provider_action as invoke
    return invoke(c, action, parameters)


def dataset_validate(path):
    from .datasets import validate_dataset
    from .integration_config import load_integrations
    return validate_dataset(path, load_integrations())


def artifact_inspect(path, hash_files=False):
    from .artifacts import inspect_artifact
    from .integration_config import load_integrations
    return inspect_artifact(path, load_integrations(), hash_files)


def generate(connection_id, model, prompt, max_tokens=128):
    from .inference import infer
    from .integration_config import find_connection
    c = find_connection(connection_id)
    if c["type"] != "inference":
        return failure("invalid_input", "Select an inference connection.")
    return infer(c, model, prompt, max_tokens)


def profile():
    from .profile import profile_home, data_root
    from .integration_config import load_integrations
    import hashlib
    settings = load_integrations()
    home = profile_home()
    return {'ok': True, 'profile_home': str(home) if home else None,
            'profile_id': hashlib.sha256(str(home or data_root()).encode()).hexdigest()[:16],
            'workspace_root': settings['workspace_root'], 'state_dir': settings['state_dir'],
            'scope': 'hermes_profile' if home else 'standalone',
            'note': 'Separate application state; not an operating-system sandbox.'}


def experiment(action, **parameters):
    from .experiments import dispatch
    from .integration_config import load_integrations
    return dispatch(action, parameters, load_integrations())


def analyze(request):
    from .optimization import analyze as build_plan
    from .environment import collect
    from .integration_config import load_integrations
    validation = build_plan(request, {})
    if not validation['ok']:
        return validation
    environment = collect(load_integrations())
    return {**build_plan(request, environment), 'environment': environment}


def benchmark(request):
    from .inference_benchmark import dispatch
    from .integration_config import load_integrations
    return dispatch(request, load_integrations())


def optimize(request):
    from .controller import dispatch
    from .integration_config import load_integrations
    return dispatch(request, load_integrations())


def onboarding(request):
    from .onboarding import dispatch
    if type(request) is not dict or set(request) != {'action'}:
        return failure('invalid_arguments', 'Use the operator setup CLI to configure roots or executables.')
    return dispatch(request)


def mission(request):
    from .mission import dispatch
    from .integration_config import load_integrations
    return dispatch(request, load_integrations())


def registry(request):
    from .registry import dispatch
    from .integration_config import load_integrations
    return dispatch(request, load_integrations())
