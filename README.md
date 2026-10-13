# Lightweight-EDL-Medical-v2

Code for the M.Tech thesis **"Calibration Robustness of Lightweight Evidential Deep Learning under Data Augmentation and Progressive Image Degradation in Medical Image Classification"**. The primary study uses HAM10000; PAD-UFES-20 and CIFAR-100 are used for OOD evaluation.

The primary matrix is 3 lightweight backbones × 4 uncertainty frameworks × 4 augmentation conditions × 3 seeds = **144 runs**. The research decisions are frozen in `protocol/protocol_v1.md` and `configs/primary.yaml`; create and verify the `protocol-v1` tag after committing this synchronized release snapshot and before primary execution.

## Workflow and roles

| Tool | Used for |
|---|---|
| VS Code (`D:\Thesis\Lightweight-EDL-Medical-v2`) | Editing/reviewing source, YAML and Markdown. Local Python execution is not required for the user's workflow. |
| GitHub Desktop | Review diffs, commit and push code/protocol changes; never commit credentials, raw datasets or checkpoints. |
| GitHub | Authoritative source history and frozen release tag `protocol-v1`. |
| Google Colab | Environment checks, pytest, training, evaluation and analysis. Each new worker session checks out the same frozen tag. |
| Google Drive | Persistent manifests/artifact backups, run checkpoints, run summaries and isolated worker logs. |

Keep the Git clone in Colab outside Drive-synchronized folders. Read raw images from Colab local storage (`/content/data`), never directly from mounted Drive.

## Repository layout

```text
configs/primary.yaml       frozen machine-readable protocol
protocol/                  written protocol and committed manifests
edl/                       data, model, loss, training and evaluation packages
scripts/                   reproducible command-line entry points
tests/                     automated tests
notebooks/colab_runner.ipynb Colab setup and worker workflow
environment/               recorded Colab environment
```

## Frozen release: routine new-session workflow

The repository contains a frozen protocol and committed manifests. Do **not** rerun manifest-generation commands during routine primary worker sessions.

1. In Colab, mount the shared Drive and clone the repository outside Drive. Check out the exact tag with `git fetch origin --tags` and `git checkout protocol-v1`.
2. Install the pinned dependencies, verify the recorded Python/Torch/torchvision/scikit-learn versions and T4 GPU, and run `python -m pytest -q`.
3. Download or reuse datasets on the Colab local disk. After a runtime reset, HAM10000 is downloaded to `/content/data/ham10000`; PAD-UFES-20 and CIFAR-100 only need to be downloaded/extracted for OOD evaluation sessions. Verify the datasets against the committed manifests.
4. Run `python scripts/make_run_manifest.py --check` to check the global run manifest. Run `scripts/verify_data.py` to validate HAM10000 against the committed split manifest. These are checks only; they must not regenerate or change the manifests.
5. Run the assigned primary worker command below. All workers use the same `protocol-v1` tag, `configs/primary.yaml`, split manifest, global 144-row run manifest and shared Drive root. Only the local backbone filter and worker-log name differ. Do not supply different primary `--max-epochs` overrides.
6. After all 144 runs have reached valid terminal states, run post-training evaluation on each assigned backbone, then the statistical, plotting and efficiency scripts, and finally `scripts/aggregate_results.py --require-evaluation` once all required artifacts are present.

## One-time pre-freeze commands (historical workflow only)

The following commands document how the frozen artifacts were created. They are **not** part of normal execution after the `protocol-v1` tag exists:

- `scripts/make_split.py`: generate a new grouped HAM10000 split only when intentionally creating a new protocol version.
- `scripts/make_ood_manifests.py`: select and hash PAD-UFES-20/CIFAR-100 files for a new protocol version.
- `scripts/make_corruption_manifest.py`: generate and hash the corruption conditions for a new protocol version.
- Smoke tests and the five-run convergence pilot establish implementation acceptance and the initial cap for a new protocol version. Do not rerun them as routine steps for the frozen `protocol-v1` primary matrix.

A real scientific defect that requires changing training/evaluation behavior must be handled as a versioned protocol revision with consistent reruns of affected experiments. Do not silently regenerate manifests, replace failed runs with new seeds, or change hypotheses after observing test results.

## Three-worker Colab execution

Each worker loads the same global 144-row manifest and filters it locally; do not create separate per-worker manifests:

```bash
python scripts/run_experiments.py --phase primary --backbone efficientnet_b0   --worker-tag efficientnet --require-gpu T4
python scripts/run_experiments.py --phase primary --backbone mobilenet_v3_small --worker-tag mobilenet    --require-gpu T4
python scripts/run_experiments.py --phase primary --backbone shufflenet_v2      --worker-tag shufflenet   --require-gpu T4
```

Each run writes to its own run directory; each worker writes to its own `worker_logs/logs_<tag>_<phase>.csv`. The aggregate script cross-checks run artifacts and worker logs. Use `--require-gpu T4` on every worker so benchmark/training hardware is consistent. Users are responsible for ensuring that their multi-account Colab usage follows applicable Colab terms.

The initial cap is 57 epochs, with patience 8 and an absolute resource ceiling of 100. If the preregistered primary extension rule triggers, commit only the approved cap increase without changing the absolute ceiling or any other protocol settings; all workers must use the same approved commit/reference. Keep `protocol-v1` immutable, and record any approved extension as a distinct commit or tag. The code permits cap-only resume only when the training-source fingerprint and all non-cap settings still match.

## Safeguards enforced by the code

- Frozen manifest hashes are checked before use; split assignments are not regenerated by training/evaluation.
- Missing/unreadable images and metadata mismatches are fatal rather than silently filtered.
- Smoke, pilot and training phases do not evaluate the held-out test set. Test predictions are exported only by the post-training evaluation workflow after the full primary matrix is eligible.
- Checkpoint selection uses validation Macro-F1, with the earliest epoch winning ties. Only `best_checkpoint.pth` and `latest_resumable.pth` are persistent checkpoint roles per run.
- Checkpoints are saved locally, validated, safely copied to Drive and checked against the persistence index before being treated as recoverable.
- Resume checks run/config/split/phase identity and the training-source fingerprint.
- A failed run is resumed with the same run identity and seed; seeds are never swapped.
- Worker-log schemas must match; duplicate or inconsistent records are rejected.
- Final test/OOD/corruption/efficiency results must be regenerated from saved predictions and frozen scripts; no composite winner is invented.

## Audit and repair record

`CODE_AUDIT_AND_REPAIR_REPORT.md` is retained as a historical report for an earlier archive. Its old statements about a draft configuration or outstanding pre-freeze gates describe the state at that time and are **superseded by the current frozen `protocol-v1` release**. The current authority is the tagged source snapshot, `configs/primary.yaml`, and `protocol/protocol_v1.md`.

## Data and licences

HAM10000 (CC BY-NC 4.0): cite the dataset paper and do not redistribute images. PAD-UFES-20 and CIFAR-100: cite and follow their dataset terms. Code licence: MIT (`LICENSE`); confirm institute requirements.
