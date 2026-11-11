# Hermes credential migration (1.1)

This release requires Hermes **0.23.1 or newer**. Credential configuration is optional: all local-only tools load without cloud accounts. The public manifest declares secrets as `config_schema` fields; Hermes owns their storage and the active profile scope. No secret is accepted in tool arguments or integration JSON.

| Connection | `credential_ref` / Hermes configuration |
| --- | --- |
| Runpod | `hermes:AI_LAB_RUNPOD_API_KEY` |
| HF Jobs / Endpoints / Storage | `hermes:AI_LAB_HF_TOKEN` |
| Inference endpoint | `hermes:AI_LAB_INFERENCE_API_KEY` |
| Upstream MCP | `hermes:AI_LAB_MCP_API_KEY` |
| S3 | `hermes:AI_LAB_S3_ACCESS_KEY_ID`, with `AI_LAB_S3_SECRET_ACCESS_KEY` and optional `AI_LAB_S3_SESSION_TOKEN` configured in Hermes |
| GCS metadata | No token reference; sign in with `hermes auth add ai-lab:google` |
| Loopback HTTP API | `AI_LAB_HTTP_API_KEY` in Hermes, at least 32 ASCII characters |

Set the relevant optional secrets using Hermes’s plugin settings. For Google, configure your own `AI_LAB_GOOGLE_CLIENT_ID` and `AI_LAB_GOOGLE_CLIENT_SECRET` through Hermes, then use its auth command. The manifest requests Storage read-only scope and permits only `storage.googleapis.com`. AI Lab calls `credentialed_provider_request`; it never obtains or refreshes the Google token.

Replace legacy `env:`, `file:`, `gcp:` or `native` references in non-secret integration settings with the matching reference above. Their old values are never read or automatically copied by this release. Existing files in other tools remain untouched. The HTTP API no longer accepts `--token-file` or creates `api.token`.

API-key providers call `agent.secret_scope.get_secret` in the live Hermes request. The plugin does not build or load a profile secret store. It follows Hermes’s single-profile policy and fails closed on an unscoped multiplexed read. Without Hermes, authenticated standalone CLI/MCP/HTTP operations are unavailable; credentialless operations remain usable.

Only the secrets required by a requested SDK operation cross to its isolated child, through anonymous stdin. They never enter argv, environment, SQLite plans or log files. Child replies are filtered and scrubbed. This is a routing and data-handling boundary, not an OS sandbox.

HF and S3 transfers run within the live request, with a bounded child process, the existing execution lock and a durable terminal job receipt. The calling tool waits for completion, up to the declared 7200-second child deadline. Detached queue cancellation returns `cancel_unsupported` for a live transfer; no remote cancellation or rollback is claimed. They cannot resume with credentials after the session is lost. Historical queued storage plans fail with a session-required error instead of using native SDK credentials. Local model jobs still use the detached queue, with a restricted nonsecret environment.

GCS metadata calls are supported. Binary GCS transfers are unavailable because Hermes’s OAuth helper has no bounded streaming/download or binary-upload interface. Native SSH execution is unavailable because no supported Hermes SSH credential route was found in v0.23.1. Both return explicit capability errors before network activity. These limits preserve the catalogue credential boundary; they are not successful integrations.

Reference: [official catalogue rule 11](https://github.com/NousResearch/hermes-agent/blob/v0.23.1/website/docs/developer-guide/plugins/catalog-submission.md), [public secret settings](https://github.com/NousResearch/hermes-agent/blob/v0.23.1/website/docs/developer-guide/plugins/index.md), [OAuth request helper](https://github.com/NousResearch/hermes-agent/blob/v0.23.1/hermes_cli/plugin_provider_requests.py).
