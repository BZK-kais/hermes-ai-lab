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
| `runpod` | `credential_ref=hermes:AI_LAB_RUNPOD_API_KEY` | probe, list, get, create, stop, delete, logs |
| `hf_jobs` | project, `credential_ref=hermes:AI_LAB_HF_TOKEN` | probe, list, get, create, cancel, logs |
| `hf_endpoints` | project, same HF reference | probe, list, get, create, stop, delete |
| `hf_storage` | repo_id, same HF reference | probe, list, get metadata, upload |
| `gcs` | project, bucket; Hermes Google OAuth | probe, list, get metadata |
| `s3` | region, bucket, `credential_ref=hermes:AI_LAB_S3_ACCESS_KEY_ID` | probe, list, get metadata, upload |

All keys come from the active Hermes secret scope. S3 additionally uses the declared `AI_LAB_S3_SECRET_ACCESS_KEY` and optional `AI_LAB_S3_SESSION_TOKEN`. The SDK always receives explicit credentials. Legacy env/file/gcp/native references and all native SDK credential lookup are refused. Google uses Hermes’s OAuth request helper; the plugin never holds or refreshes its token. See [migration and declarations](credential-migration.md).

HF requires huggingface_hub v1+ with `set_client_factory` and httpx; S3 requires boto3. GCS metadata uses the Hermes runtime. Missing SDKs produce `dependency_missing`. No account has been verified merely by adding a configuration.

## Creation and recovery

Creates require explicit `budget_usd`, `timeout_seconds` (1–86400), `idempotency_key` and `config`, plus an existing local `workspace`. An exclusive intent file under `.ai-lab/intents` is flushed before the request. Reuse of a key is refused, including after an ambiguous network failure. Inspect provider resources using list/get before any new intent. Intent files do not include configuration, environment or secrets. No automatic duplicate POST is performed; SDK create request hooks reject a second POST.

Runpod config supports name, image or templateId, gpu `{id,count}`, disk and dataCenterIds. No arbitrary environment/registry authentication/command forwarding is accepted. HF Jobs config supports image, command argv and hardware flavor; it injects a provider timeout and attempts=1. This is an explicit cloud workload, not a local shell. HF Endpoint config supports named repository/hardware/region/vendor/revision/task and replica/scale-to-zero settings; endpoints are authenticated.

Budget amounts are declarations, **not provider billing caps**. Results report `budget.enforced:false`. Only HF Jobs has an enforced runtime timeout here. Runpod and HF Endpoints report `timeout.enforced:false`; no watchdog is installed or claimed. Stop/pause and delete are separate from persistent storage deletion. Volume provisioning/deletion, billing queries, distributed cluster launch and automatic orphan deletion are not implemented in this slice.

## Network and output

Runpod uses REST v2 `https://api.runpod.io/v2`: GET/POST `/pods`, GET/DELETE `/pods/{id}`, POST `/pods/{id}/action` with stop, and SSE GET `/pods/{id}/logs`. Requests use a 10-second socket timeout, no environment proxy, no redirect, 2 MiB JSON cap and no retries. Lists return at most 100 resources. Logs consume a bounded SSE sample and expose event metadata only; message contents are withheld. HF log contents are also withheld. Successful control-plane calls do not prove successful training or artifact persistence.

HF SDK HTTP clients are process-local with proxy use and redirects refused, 10-second network timeouts and host allowlisting. HF uploads additionally permit provider-controlled `*.hf.co` / `*.huggingface.co` transfer destinations; Xet is disabled in the worker so Rust transfers cannot bypass that HTTP policy. Provider action redirects fail closed. Explicit HF byte transfers permit SDK-managed HTTPS redirects only within that host allowlist, and strip bearer headers outside huggingface.co. The isolated SDK operation has a 45-second process deadline, or 300 seconds for an upload. GCS requests project selected metadata fields and use the Hermes OAuth helper and declared storage.googleapis.com host. The response size is checked after Hermes buffers it; no streaming memory bound is claimed. S3 uses the configured regional AWS S3 endpoint with one attempt, no proxy, and 10-second connect/read timeouts. Only operation-specific credentials cross the anonymous stdin channel to SDK children; their environments contain no inherited provider keys.

Uploads require an existing local file confined to workspace and an explicit relative destination key. S3 provider uploads use IfNoneMatch=* to avoid overwriting. GCS uploads are unavailable. S3 uses single PUT, limited to 5 GiB; multipart/resume is not implemented. HF upload commits to the explicitly configured existing model repository and may replace that repository path. get is metadata-only: no weight download or presigned URL is returned.

## SSH

SSH connection metadata is validated, but every remote action returns `auth_route_unavailable`. No SSH or SCP process is launched and no native identity/config/agent is consulted. A supported Hermes credential route is required before this connector can execute.

## Primary references checked 2026-11-09

- [Runpod REST v2](https://docs.runpod.io/api-reference-v2/overview), [create](https://docs.runpod.io/api-reference-v2/pods/create-a-pod), [SSE logs](https://docs.runpod.io/api-reference-v2/pods/stream-pod-logs), [lifecycle](https://docs.runpod.io/pods/manage-pods)
- [HF HfApi methods](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api), [Inference Endpoints](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_endpoints), [upload](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [GCS upload](https://docs.cloud.google.com/storage/docs/uploading-objects)
- [S3 PutObject](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/put_object.html)

Validation uses synthetic transports and mocked SDKs. No actual account, cloud mutation or SSH session was exercised.

## Portable worker protocol

`scripts/build_remote_worker.py --workspace /srv/ai-lab --output /local/workspace/remote_worker.py` builds a new zipapp locally; it does not connect, install, or modify a PC configuration. Python accepts the zipapp despite its `.py` extension. The bundle may be deployed by the operator separately; the catalogue SSH connector cannot deploy it. The archive contains only `ai_lab_core` Python source, a bootstrap and filtered build-time administrator settings; it excludes local configuration, state, credentials, bytecode and other projects.

The remote workspace must already exist. An optional `--admin-config` provides non-secret `model_roots`, `data_roots`, `runtime_bin_dirs`, `backend_paths` and `backend_recipe_args` for that machine. These are build-time administrator bindings. The worker plan uses `{operation, parameters, settings}`; settings may only repeat a subset of the embedded values exactly. Job callers cannot broaden roots, replace backend programs or inject recipe command templates. Missing installed tools/bindings return planning errors. The `probe` operation returns the hardware profile without claiming kernel/training validation.

Before backend execution the runner extracts packaged source beneath `<workspace>/.ai-lab-worker/runtime/<bundle-hash>` so recipes invoked as Python files have real filesystem paths. It then calls `plan_operation`, executes only the constructed argv with its environment and timeout, and checks expected output files/directories. It writes a compact `result.json` beside the remote plan and returns a nonzero status for failed execution/missing outputs. Backend stdout/stderr is withheld. `result.json` is the artifact check; this release does not launch it via SSH. Kernel crashes and interrupted uploads still require reconciliation; no remote account or hardware has been tested by building this artifact.

Storage provider `get` remains metadata-only. Explicit `storage.download` and `storage.upload` use HF/S3 SDKs in a live-session child with a 7200-second deadline and a durable job receipt. The execution lock prevents overlap with local model jobs. Secrets are not persisted or inherited by detached workers. Remote size is checked before download, successful local downloads publish atomically without overwrite, and failed partial downloads are removed. HF pins the model revision; S3 pins VersionId where available. GCS byte transfers are unavailable until Hermes supplies a bounded binary transport. No real cloud transfer has been performed.
