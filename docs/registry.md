# Configuration registry

The registry keeps each Hermes profile's local controller qualifications and active configuration in its AI Lab state directory. A standalone installation uses its configured AI Lab state directory. A successful evaluation and capture leave the current activation unchanged until an explicit `activate`.

| Action | Arguments | Result |
| --- | --- | --- |
| `capture` | `job_id` | Records the configuration accepted by a full local `optimization.search` job. |
| `list` | | Lists entries, qualification status, and the active and previous identifiers. |
| `inspect` | `entry_id` | Shows artifact identities, measured holdout summary, and local asset paths. |
| `activate` | `entry_id` | Rechecks both artifact fingerprints and switches to the qualified entry. |
| `status` | | Shows the current activation and whether it can be rolled back. |
| `rollback` | | Restores the prior activation after checking its assets. A first activation can be rolled back to an unset state. |
| `generate` | `input`, `output_dir` | Queues a local MLX job with the active configuration frozen at submission. |
| `result` | `job_id` | Returns generation status or the verified output of a completed job in this profile. |
| `export` | `entry_id`, `output_path` | Writes a portable ZIP file to a new path in the profile workspace. |
| `import` | `package_path`, `model` | Verifies a package and binds it to a matching local base model. |

Capture requires a succeeded job from the same profile with a completion receipt, intact frozen request and selection, and an accepted configuration that passed development and holdout gates. A qualified candidate and a retained baseline are both eligible. A retained baseline carries no verified optimization gain. Historical reports created before completion receipts are available through the controller's report interface. To activate one of those configurations, submit a fresh local evaluation and capture its result.

Generation uses `backend_paths.mlx.python`. The isolated worker checks the evaluated Python, MLX, and MLX-LM versions and the measured inference source files. It reuses the tested chat template handling, input encoding, system prompt, token limit, prefill step, and output constraint. The worker verifies the model and adapter around loading and generation and serializes access to the local MLX device. Use a new `output_dir` inside the profile workspace. The request and report contain the input and generated text and live in that private directory. A later activation or rollback takes effect for subsequent submissions. Jobs already submitted retain their frozen configuration.

Packages contain the selected inference settings, including its system prompt, a compact evaluation summary, and asset hashes. The base model is referenced by digest and supplied separately when importing. For LoRA configurations the bundle carries the adapter tensors and a projection of the adapter configuration to the fields used by the loader. The proof records the original evaluated adapter digest, and the package records the portable adapter digest. This removes training paths while preserving the evaluated source files. Dataset rows, provider configuration, and local runtime bindings are omitted. Inspect the prompt before sharing a package if it contains private information.

Imported entries start inactive and have the status `imported_unverified`. Use `inspect` to find the rebound model and any imported adapter for a fresh local controller protocol. Capture the newly accepted configuration and activate that entry. Package checksums verify consistency of the transported bytes. Local controller proof establishes qualification for the importing profile. To move the ZIP through a configured storage provider, use the explicit `storage.upload` or `storage.download` operations. Remote execution remains unverified.
