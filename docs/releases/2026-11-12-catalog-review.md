# AI Lab 1.1.1: provider boundary and English documentation

This targeted patch starts at the reviewed 1.1.0 commit `4a11ae951c2c2e1a23283a14a604313bb3b132a8`. It addresses the [maintainer review](https://github.com/NousResearch/hermes-agent/pull/154091#pullrequestreview-5869955061). The separate local 1.2 experiments are not bundled into this catalogue patch.

## Execution boundary

The model-facing provider tool now permits only connector-specific metadata reads. Cloud creation, stopping, deletion and cancellation, HF/S3 provider uploads, and every upstream MCP tools/call are refused with `operator_required`. An upstream MCP tool named `list` or `probe` is not exempt. `ai_lab_connection_probe` keeps protocol discovery on tools/list.

The second upload route, `ai_lab_run` with `storage.upload`, enforces the same operator requirement before credentials, job creation or transport. Planning and downloads remain distinct. Local model execution, profile setup and explicit inference retain their existing contracts; remote inference may be billable.

The named commands `ailab provider-action`, `ailab run`, `hermes ai-lab provider-action` and `hermes ai-lab run` enter an internal operator scope. Generic `ailab call`, native model dispatch, MCP and HTTP never enter it. The scope is bound to its named command, current thread and task and revoked on exit, including exceptions and copied contexts. No model-provided JSON field or API bearer token can grant this scope. Existing Hermes-managed credential requirements still apply.

This implements the maintainer's operator-only CLI alternative. It does not claim an interactive Hermes approval prompt or an OS security boundary against arbitrary trusted Python or shell code under the same account.

## English and French

README.md and the canonical workflow SKILL.md are now English. README.fr.md and references/workflow.fr.md preserve the French versions, with links in both directions. The source archive includes both languages. Operational policy is described in each version.

## Verification

The MCP 1 environment ran 99 discovered tests: 98 passed and the optional native-Hermes test was skipped there. The native Hermes test passes separately in the official Hermes 0.23.1 runtime. Focused cases cover each declared mutation family, both upload routes, names that appear read-only on MCP, invalid flags, unknown actions, context revocation, task/thread isolation, and named CLI execution exactly once with simulated providers.

Real stdio MCP 1 and authenticated loopback HTTP requests exercise the denials. The HTTP server is also created inside a live operator scope to verify that its worker cannot inherit operator authority. The independent `tests/check_provider_boundary.py` loads the actual plugin through Hermes, verifies all 22 registrations, checks raw registry and model_tools.handle_function_call denials before any credential, intent, job or transport side effects, and exercises the native CLI parser and handler.

All external provider operations use synthetic fixtures and simulated SDK/transports; no live cloud account, upload, paid workload or real credential is used. The independent native test simulates the MCP v1 provider client only after checking refusals; it does not establish live upstream MCP compatibility with the host's MCP v2 SDK. This 1.1 patch retains its declared optional MCP constraint `<2`.

Reproduce the unit suite with Python and the declared optional dependencies:

```sh
python -m unittest discover -s tests
```

Use a compatible Hermes Python and checkout for native verification:

```sh
PYTHONPATH=/path/to/hermes-checkout /path/to/hermes-python tests/check_provider_boundary.py
```

Official Hermes 0.23.1 catalogue validation passed all 15 checks on a clean source archive, with actual dependency resolution and installation in an isolated profile. Native registration exposed 22 tools and passed 14 dispatch checks. One non-blocking warning remains for private Hermes imports in the auxiliary installation and verification scripts. No main-profile configuration was used for catalogue validation.

Publication and maintainer admission are separate from these local checks. Passing them does not imply that the PR has been accepted.
