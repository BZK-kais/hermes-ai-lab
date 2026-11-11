# Credential routing verification — AI Lab 1.1.0

The catalogue candidate requires Hermes >=0.23.1. The runtime change replaces direct credential stores with declared Hermes secret names and the supported OAuth request helper. See [migration and limits](../credential-migration.md).

## Observed checks

- The exported source tree passes **90 regression tests** using Python 3.11 on macOS, including loopback HTTP and the optional MCP transport. Run `python -m unittest discover -s tests` with the optional test dependencies installed. No model training or paid provider operation was part of this check.
- Official Hermes **v0.23.1** reports **15/15 checks passed** for `hermes plugins validate /path/to/plugin --install-deps --json`. Dependencies were installed and resolved successfully in a new isolated home. This is a local validation result; GitHub catalogue CI and maintainer acceptance are separate.
- `tests/check_hermes_credentials.py /path/to/plugin`, run with the Hermes v0.23.1 runtime, loads two installed copies through the real PluginManager and registry. Runpod and HF calls use only the selected profile's synthetic key. A missing scoped key cannot fall back to the ambient fixture under multiplexing. The actual OAuth helper resolves the correct installed-plugin manifest and Google route for each profile. Network transports and credential stores are replaced with synthetic fixtures; live cloud access is not claimed.
- The regression for a disappearing SQLite rollback journal reproduced the earlier log-read failure before the fix. The corrected job tests pass; training observability additionally passed 20 repetitions in the development checkout. Symlink sidecars remain refused.

## Reviewer-visible warnings

The official validator reports three existing private-core imports, all in auxiliary installation/diagnostic scripts: `hermes_cli.plugin_dev._doctor_runtime` in `scripts/check_hermes_registration.py:14`, `hermes_cli.__version__` in `scripts/install_local.py:276`, and `hermes_cli.managed_scope.managed_config_keys` in `scripts/install_local.py:278`. These scripts are not imported by the registered plugin runtime. The security scan is safe, no-core-override check passes and all 22 registered tools match the manifest.

HF/S3 authenticated transfers now require a live request and do not support detached cancellation or credential recovery after session loss. GCS byte transfers and SSH execution return explicit unsupported-route errors. Local model jobs remain detached. Credentials never enter command arguments, inherited provider environment variables, persisted plans or job logs. The IPC payload stays in memory; same-user malicious code is not isolated by this design.
