# Provider adapters

Implemented entry points:

- `providers.validate_connection(c)`, `probe_connection(c)`, `provider_action(c, action, params)`.
- `remote.validate_connection(c)`, `probe_connection(c)`, `remote_action(c, action, params)`; aliases `validate_remote`, `probe_remote`, `provider_action`.
- `credentials.validate_reference(ref)` and runtime-only `resolve_secret(ref)` / `resolve_credential(ref)`.
- Public `ACTION_SCHEMAS` constants describe action parameter fields. Results use `{ok, error:{code,message}}` on failure. No credentials or provider exceptions are returned.

## Configuration

Providers accept only `id,type,base_url,project,bucket,credential_ref,python,workspace,region,repo_id`, all strings. Unknown fields and arbitrary provider base URLs are rejected. `python` is an optional absolute local interpreter path with the SDK installed; SDK operations run in an isolated child process. No SDK is installed automatically.

Types and required fields:

| Type | Required configuration | Actions |
| --- | --- | --- |
| `runpod` | credential_ref | probe, list, get, create, stop, delete, logs |
| `hf_jobs` | project (HF namespace), credential_ref | probe, list, get, create, cancel, logs |
| `hf_endpoints` | project (HF namespace), credential_ref | probe, list, get, create, stop (pause), delete |
| `hf_storage` | repo_id (existing model repo), credential_ref | probe, list, get (metadata), upload |
| `gcs` | project, bucket; native ADC | probe, list, get (metadata), upload |
| `s3` | region, bucket; native boto3 credentials | probe, list, get (metadata), upload |

HF requires an explicit credential reference, including `native` to use the existing HF SDK token. Runpod accepts `env:NAME`, `file:/absolute/token`, or `gcp:projects/PROJECT/secrets/SECRET/versions/VERSION`. Token files are read up to 16 KiB; GCP access uses the existing `gcloud` command. The secret resolver is an internal runtime function and must never be registered as an agent tool. These boundaries do not isolate secrets from code running under the same OS identity.

Native GCS/S3 authentication is handled by the SDK; this adapter does not accept raw service-account JSON or AWS keys. HF requires huggingface_hub v1+ with `set_client_factory` and httpx; GCS requires google-cloud-storage; S3 requires boto3. Missing SDKs produce `dependency_missing`. No account has been verified by adding a configuration.

## Creation and recovery

Creates require explicit `budget_usd`, `timeout_seconds` (1–86400), `idempotency_key` and `config`, plus an existing local `workspace`. An exclusive intent file under `.ai-lab/intents` is flushed before the request. Reuse of a key is refused, including after an ambiguous network failure. Inspect provider resources using list/get before any new intent. Intent files do not include configuration, environment or secrets. No automatic duplicate POST is performed; SDK create request hooks reject a second POST.

Runpod config supports name, image or templateId, gpu `{id,count}`, disk and dataCenterIds. No arbitrary environment/registry authentication/command forwarding is accepted. HF Jobs config supports image, command argv and hardware flavor; it injects a provider timeout and attempts=1. This is an explicit cloud workload, not a local shell. HF Endpoint config supports named repository/hardware/region/vendor/revision/task and replica/scale-to-zero settings; endpoints are authenticated.

Budget amounts are declarations, **not provider billing caps**. Results report `budget.enforced:false`. Only HF Jobs has an enforced runtime timeout here. Runpod and HF Endpoints report `timeout.enforced:false`; no watchdog is installed or claimed. Stop/pause and delete are separate from persistent storage deletion. Volume provisioning/deletion, billing queries, distributed cluster launch and automatic orphan deletion are not implemented in this slice.

## Network and output

Runpod uses REST v2 `https://api.runpod.io/v2`: GET/POST `/pods`, GET/DELETE `/pods/{id}`, POST `/pods/{id}/action` with stop, and SSE GET `/pods/{id}/logs`. Requests use a 10-second socket timeout, no environment proxy, no redirect, 2 MiB JSON cap and no retries. Lists return at most 100 resources. Logs consume a bounded SSE sample and expose event metadata only; message contents are withheld. HF log contents are also withheld. Successful control-plane calls do not prove successful training or artifact persistence.

HF SDK HTTP clients are process-local with proxy use and redirects refused, 10-second network timeouts and host allowlisting. HF uploads additionally permit provider-controlled `*.hf.co` / `*.huggingface.co` transfer destinations; Xet is disabled in the worker so Rust transfers cannot bypass that HTTP policy. Unexpected redirect-based upload paths therefore fail closed. The isolated SDK operation has a 45-second process deadline, or 300 seconds for an upload. GCS API requests are restricted to storage.googleapis.com; credential refresh remains native SDK behavior. S3 uses the configured regional AWS S3 endpoint with one attempt, no proxy, and 10-second connect/read timeouts. SDK-native credential acquisition is not a claim of independently sandboxed credentials or uniformly enforced auth-service egress.

Uploads require an existing local file confined to workspace and an explicit relative destination key. GCS uses generation=0 and S3 IfNoneMatch=* to avoid overwriting. S3 uses single PUT, limited to 5 GiB; multipart/resume is not implemented. HF upload commits to the explicitly configured existing model repository and may replace that repository path. get is metadata-only: no weight download or presigned URL is returned.

## SSH

SSH configuration accepts only `id,type=ssh,host,user,port,python,workspace,remote_workspace,script`. `port` is an integer; `python` is the remote executable. No SSH key directory is enumerated. The existing SSH client authentication/host configuration is used with BatchMode, strict known-host checks, forwarding cleared, PermitLocalCommand=no, bounded connection/process deadlines, and no local shell invocation.

`probe` executes a fixed Python availability/platform probe. `exec` accepts only `plan_path` and `run_id`: a bounded workspace JSON object and unique identifier. The trusted script is fixed in local configuration, confined to workspace, copied with SCP along with the plan, then launched as `python runner.py --plan plan.json` via fixed control code. API callers cannot supply shell commands or a script body. Local and remote intents prevent automatic resubmission. `status` reads selected state; `cancel` signals only a Linux process group whose `/proc` command line identifies the recorded worker directory. The configured runner must treat its plan safely. Remote state/log files persist; logs are not exposed or automatically removed. A crash between submission and state update can require manual reconciliation. No remote compute watchdog or billing limit is claimed.

## Primary references checked 2026-11-09

- [Runpod REST v2](https://docs.runpod.io/api-reference-v2/overview), [create](https://docs.runpod.io/api-reference-v2/pods/create-a-pod), [SSE logs](https://docs.runpod.io/api-reference-v2/pods/stream-pod-logs), [lifecycle](https://docs.runpod.io/pods/manage-pods)
- [HF HfApi methods](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api), [Inference Endpoints](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_endpoints), [upload](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [GCS upload](https://docs.cloud.google.com/storage/docs/uploading-objects)
- [S3 PutObject](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html)

Validation uses synthetic transports and mocked SDKs. No actual account, cloud mutation or SSH session was exercised.

## Portable worker protocol

`scripts/build_remote_worker.py --workspace /srv/ai-lab --output /local/workspace/remote_worker.py` builds a new zipapp locally; it does not connect, install, or modify a PC configuration. Python accepts the zipapp despite its `.py` extension. Point the SSH connection's trusted `script` at this workspace-relative file. The archive contains only `ai_lab_core` Python source, a bootstrap and filtered build-time administrator settings; it excludes local configuration, state, credentials, bytecode and other projects.

The remote workspace must already exist. An optional `--admin-config` provides non-secret `model_roots`, `data_roots`, `runtime_bin_dirs`, `backend_paths` and `backend_recipe_args` for that machine. These are build-time administrator bindings. The worker plan uses `{operation, parameters, settings}`; settings may only repeat a subset of the embedded values exactly. Job callers cannot broaden roots, replace backend programs or inject recipe command templates. Missing installed tools/bindings return planning errors. The `probe` operation returns the hardware profile without claiming kernel/training validation.

Before backend execution the runner extracts packaged source beneath `<workspace>/.ai-lab-worker/runtime/<bundle-hash>` so recipes invoked as Python files have real filesystem paths. It then calls `plan_operation`, executes only the constructed argv with its environment and timeout, and checks expected output files/directories. It writes a compact `result.json` beside the remote plan and returns a nonzero status for failed execution/missing outputs. Backend stdout/stderr is withheld. SSH status tracks the runner exit; `result.json` is the more precise artifact check. Kernel crashes and interrupted uploads still require reconciliation; no remote account or hardware has been tested by building this artifact.

Storage provider `get` remains metadata-only. Durable byte transfers are exposed separately as named operations `storage.download` and `storage.upload` in `transfers.py`, through the common job manager (7200-second deadline). They use configured HF/GCS/S3 SDKs, enforce a maximum size of 50 GiB (configurable downward), inspect remote size before download, publish a local result atomically without overwrite and clean up failed partial downloads. HF pins the model repository revision; GCS pins the generation; S3 pins a VersionId when available. SDK-native transfer implementation differs by provider; generic cross-process resume is not promised. The old synchronous provider `upload` retains its 300-second deadline and documented size/overwrite rules. No real cloud transfer has been performed.
