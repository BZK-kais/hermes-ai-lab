# Hermes AI Lab 1.1.1

[Lire le README en français](README.fr.md)

AI Lab is a native Hermes plugin for planning experiments, comparing model configurations, and using a configuration that meets your criteria. It exposes **22 Hermes tools** and the **`ai-lab:workflow`** skill. Jobs, reports, and active configurations belong to a profile. The same tool contract is available through the `ailab` CLI, MCP over stdio, and a local HTTP API.

## Install in Hermes

Extract the plugin archive, then run the installer with **the Python interpreter from your Hermes installation** (Hermes 0.23.1 or later) and an existing profile.

```sh
/path/to/hermes-python /path/to/extracted/ai-lab/scripts/install_local.py \
  --home /path/to/hermes-profile
```

The installer copies the plugin to `plugins/ai-lab` and enables it for the next session. It preserves any different installation already present. The archive includes the code, documentation, and examples. The wheel includes the code and verification fixtures. Both distributions exclude local model weights, secrets, and user state.

To update an installation from a release, run `install_local.py --home /path/to/hermes-profile --upgrade` from the new archive. The installer verifies the existing copy and returns `backup_path` when it creates a backup.

In the selected profile, register your existing local directories and the Python interpreter for your MLX environment. Each path requires an explicit operator flag. `setup --check` checks the profile state and whether the executable is present.

```sh
hermes ai-lab setup \
  --model-root /path/to/models \
  --data-root /path/to/data \
  --mlx-python /path/to/mlx-venv/bin/python
hermes ai-lab setup --check
hermes ai-lab profile
```

Open a new Hermes session. For example, ask “Use `ai-lab:workflow` to make my local model faster on this JSON extraction dataset. Choose compatible settings, compare them, and save the results for future trials.” The skill handles technical choices and asks for essential task context or access when it is missing.

## Access and limits in version 1.1

Configure optional access in the Hermes plugin settings. The plugin reads declared keys with `agent.secret_scope.get_secret` in the active profile. Connection records contain only a reference such as `hermes:AI_LAB_HF_TOKEN`. The plugin rejects legacy `env:`, `file:`, `gcp:`, and `native` references. It does not read credential stores, run `gcloud`, or use implicit SDK credentials. See the [credential migration guide](docs/credential-migration.md) and [1.1 verification notes](docs/releases/2026-11-11-credentials.md).

Google access uses your OAuth client configured in Hermes and `hermes auth add ai-lab:google`. Hermes stores and refreshes the tokens. GCS supports read-only metadata access. GCS transfers and SSH execution return explicit errors pending a suitable Hermes integration. Hugging Face and S3 transfers run in the active session and issue a job receipt. They do not resume automatically if the session ends. Local jobs run detached.

Explicit operations can contact Hugging Face, Runpod, AWS S3, Google Cloud Storage, and configured endpoints. They can read models and data under approved directories, write profile artifacts and reports, and launch local executables and long-running jobs. Discovery does not provision cloud accounts, download model weights, start training, or incur remote charges. Declared budgets do not cap provider billing. SDK calls pass secrets through a private stdin channel in memory. Code running under the same system account may still be able to access those secrets.

## Cloud action boundary

Model-facing tools may read supported providers through the connector-specific metadata actions (`probe`, `list`, `get`, and `logs` where available). Creating, stopping, deleting or cancelling cloud resources, uploading to HF/S3, and calling any upstream MCP tool require an explicit **named operator CLI command**. `ai_lab_run` also refuses `storage.upload` through model-facing interfaces. This check applies to native Hermes tools, generic `ailab call`, stdio MCP and the local HTTP API, including authenticated requests.

```sh
ailab provider-action CONNECTION ACTION --params-file /path/parameters.json
ailab run storage.upload --params-file /path/transfer.json
hermes ai-lab provider-action --args-file /path/provider-request.json
hermes ai-lab run --args-file /path/run-request.json
```

The native provider request is `{"connection_id":"CONNECTION","action":"ACTION","parameters":{}}`; the run request contains `operation` and `parameters`. Existing credential and path requirements still apply. A JSON flag, a bearer token, a declared budget or an upstream MCP tool annotation does not grant operator authority. MCP connection discovery remains available through `ai_lab_connection_probe`; an upstream tool named `probe` or `list` is still an opaque tool call and requires the operator command.

Local model operations and explicitly requested inference keep their existing contracts. Remote inference can be billable, and downloads write local files. This boundary controls built-in provider routes; it is not an OS sandbox against trusted Python code or shell commands running as the same user. [Review fix and verification](docs/releases/2026-11-12-catalog-review.md)

## Automated experiment loop in the development checkout

`ai_lab_mission` supports `autoplan`, `autostart`, `autoreport`, and `lessons`. Given a task, objective, model, and local dataset, the planner chooses compatible trials, states its defaults, and reuses results from the same context. `autostart` launches the bounded campaign. `autoreport` verifies the local job before recording lessons. Exploration stays on the development split when qualification criteria are missing. The planner preserves any supplied thresholds.

The agent uses conditional recipes drawn from reviewed projects. The available MLX trials are documented alongside CUDA, MoE, and distributed options that cannot yet be executed. The loop learns by recording evaluation results in the profile. Candidate activation is a separate registry operation. See the [learning loop and its evidence](docs/recipes/learning-loop.md). The existing 1.0 archives predate this local development work.

## From mission to inference

1. **Prepare.** `ai_lab_analyze` identifies missing context. `ai_lab_mission` builds a protocol from the dataset, baseline, declared variants, thresholds, and budget. The held-out test set stays hidden during preparation. See the [mission guide](docs/mission.md).
2. **Measure.** Pass `start_request` to `ai_lab_optimize`, track the job with `ai_lab_job`, and read the report. The controller selects a finalist on the development split and evaluates the baseline and finalist on the held-out test set. A development-only campaign serves as a screening run. See the [decision rules](docs/controller.md).
3. **Capture and activate.** With `ai_lab_registry`, use `capture` to save the accepted configuration from a completed profile job. `activate` verifies the assets and selects the entry to use. Check both the job outcome and the qualification verdict.
4. **Generate.** `generate` submits a job with a snapshot of the active configuration. `result` retrieves its status or the verified response.
5. **Transfer.** `export` creates a ZIP with the inference settings, fingerprints, evaluation summary, and adapter if present. `import` associates the package with the matching local base weights and creates an inactive entry. To use it, qualify the configuration in the target profile, capture the result, and activate the new entry.
6. **Roll back.** `rollback` restores the previous activation, including the initial state with no active configuration. See the [registry commands](docs/registry.md).

The native `hermes ai-lab mission`, `optimize`, `job`, and `registry` commands accept `--args-file`. The [reproducible demo](examples/lifecycle-v1/README.md) prepares 16 synthetic cases where the value to copy appears in a JSON field. It verifies the qualification lifecycle on that task and one configuration.

The 1.0 controller runs **MLX on Apple Silicon**, one request at a time, using exact-match scoring for JSON or text. Results apply to the measured data, model, settings, and hardware. In the historical PAWS experiment, the previous LoRA scored **159/192** and the new one **158/192**, with **7 and 33 false equivalences**, respectively. The campaign did not qualify a candidate. [PAWS report](docs/experiments/2026-11-09-lora-improve.md)

## Operations and training progress

`hermes ai-lab operations` or `ailab operations` lists the **21 named operations**, their parameters, and prerequisites. They cover Hugging Face, MLX, llama.cpp, TRL/PEFT, Unsloth, LLaMA-Factory, NeMo, lm-eval, Hermes session export, and Hugging Face and S3 transfers. Operation availability depends on the engine, its dependencies, and configured connections. Execution on NVIDIA hardware and remote compute providers still needs verification in the target environments. [Connector contracts](docs/provider-adapters.md)

For `mlx.train`, supply the model, dataset directory, and a new output directory under the workspace. `steps_per_report`, `steps_per_eval`, and `save_every` control how often the job reports training loss and validation loss and saves checkpoints. The MLX-LM CLI uses its defaults when you omit these parameters.

```sh
ailab plan mlx.train --params-file /path/to/training.json
ailab run mlx.train --params-file /path/to/training.json
ailab jobs logs JOB_ID
```

Checkpoints are numbered. The job checks for the final `adapter_config.json` and `adapters.safetensors` files. To qualify the exported adapter, explicitly add it to an evaluation campaign. [Monitor MLX training](docs/mission.md#observer-un-entraînement-mlxtrain)

## Standalone CLI, MCP, and HTTP

Install the wheel with Python 3.9 or later in a dedicated environment. State is stored outside the package. When a Hermes profile is configured, state lives under `ai-lab/` in that profile. A fresh standalone installation uses the `hermes-ai-lab` user data directory.

```sh
python3 -m pip install /path/to/hermes_ai_lab-1.1.1-py3-none-any.whl
ailab setup --check
ailab operations
```

The `mcp` extra supplies the SDK for `ailab mcp`. The `providers` extra supplies the SDKs for the corresponding connectors. `ailab api --port 8765` serves the tool API on `127.0.0.1`. It requires `AI_LAB_HTTP_API_KEY`, configured in Hermes with at least 32 characters. The API does not create or read a token file. Authenticated operations fail explicitly without the Hermes runtime and declared access.

`GET /health` returns service status. `GET /tools` and `POST /tools/call` require the bearer token. A tool call contains `name` and `arguments`, for example `{"name":"ai_lab_operations","arguments":{}}`. This interface dispatches AI Lab tools. OpenAI Chat Completions clients use a separate API contract.

For dated results, release artifacts, and verification limits, see the [1.0 release report](docs/releases/2026-11-11-v1.md), [STATUS.md](STATUS.md), and the [release contract](docs/RELEASE-CONTRACT.md).
