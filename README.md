# Lightweight-EDL-Medical-v2

Code for the M.Tech thesis **"Calibration Robustness of Lightweight Evidential Deep Learning under Data Augmentation
and Progressive Image Degradation in Medical Image Classification"** (HAM10000, with PAD-UFES-20 and CIFAR-100 as OOD sets).

An empirical comparison of 3 lightweight backbones x 4 uncertainty frameworks x 4 augmentations x 3 seeds = **144 runs**.
The experimental decisions are written in `protocol/protocol_v1.md` and, machine-readably, in `configs/primary.yaml`.

## Workflow
| Tool | Used for |
|---|---|
| VS Code (`D:\Thesis\Lightweight-EDL-Medical-v2`) | all code edits and local CPU tests |
| GitHub Desktop | review, commit, push (never commit data, checkpoints or tokens) |
| GitHub | authoritative code; the primary study runs from the tag `protocol-v1` |
| Google Colab | training and evaluation only; `notebooks/colab_runner.ipynb` clones the tag every session |
| Google Drive | per-run results (`<DRIVE_ROOT>/<phase>/runs/<run_id>/`), worker logs, protocol artifacts |

Keep the Git clone **outside** any Drive-synced folder. Images are read from Colab local disk (`/content/data`), never from Drive.

## Layout
```
configs/primary.yaml      all settings (validated at load; unresolved fields are TO_FILL_...)
protocol/                 written protocol; frozen manifests are committed here
edl/                      package: data/, models/, losses/, training/, evaluation/, augment.py, frameworks.py
scripts/                  command-line entry points (see below)
tests/                    pytest suite
notebooks/colab_runner.ipynb
environment/              recorded Colab versions
```

## One-time setup (in this order)
1. **Local environment** (outside Drive): `python -m venv .venv`, `pip install -r requirements-dev.txt`, then CPU torch:
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`. Run `python -m pytest`.
   Tests that need PyTorch are skipped when it is not installed, so install it before trusting the result.
2. **Global run manifest**: the clean repository includes a newly generated `protocol/run_manifest_v1.csv` with exactly
   144 unique factorial rows and its verified SHA-256 in `configs/primary.yaml`. Check it with
   `python scripts/make_run_manifest.py --check`; do not regenerate or hand-edit it after the protocol snapshot is frozen.
3. **Colab**: run cells 1-5 of the notebook, then `scripts/make_split.py` (cell 6a). It prints the split SHA-256 and checks it
   against the earlier verified reference. Put the hash in `data.split.sha256`, commit `protocol/split_manifest_v1.csv` (download it
   from `<DRIVE_ROOT>/protocol_artifacts/`), push, re-clone, run `scripts/verify_data.py`.
4. **OOD and corruption manifests** (cells 6c-6d): record their hashes in the config, inspect `corruption_example_grid.png`,
   and change the proposed severities in the config if they look wrong *before* freezing.
5. **Smoke test, then the five-run pilot** (cells 7a-7b). `scripts/pilot_report.py` prints the extension decision and
   `E_max = max(best epochs) + 8 + 15`. Put `training.max_epochs` and `training.absolute_epoch_ceiling` in the config.
6. **Freeze**: finish the checklist in `protocol/protocol_v1.md`, commit, tag `protocol-v1`, push the tag.
7. **Primary runs** (cell 8 on each worker). After all workers finish, run `scripts/evaluate_runs.py` on each assigned backbone/worker
   (the default evaluates ID, PAD-UFES-20, CIFAR-100, and the selected corruption subset), then run
   `scripts/analyze_stats.py`, `scripts/plot_results.py`, `scripts/run_efficiency.py`, and finally
   `scripts/aggregate_results.py --require-evaluation` from one worker.

## Running on several Colab workers
Every worker clones the same tag and loads the **same** 144-row run manifest, then filters it locally:
```
python scripts/run_experiments.py --phase primary --backbone efficientnet_b0   --worker-tag efficientnet --require-gpu T4
python scripts/run_experiments.py --phase primary --backbone mobilenet_v3_small --worker-tag mobilenet    --require-gpu T4
python scripts/run_experiments.py --phase primary --backbone shufflenet_v2      --worker-tag shufflenet   --require-gpu T4
```
Each run writes only to its own folder (`runs/<run_id>/`); each worker writes only to its own log
(`worker_logs/logs_<tag>_<phase>.csv`). Nothing shared is ever written by two processes. `scripts/aggregate_results.py` builds
`final_factorial_v1.csv` from the per-run files and cross-checks the worker logs. `--shard i/n` is available if you prefer to
balance the load by runs instead of by backbone. Run `--require-gpu T4` on every worker so all runs use the same GPU model.
Whether Google permits using several accounts this way is your responsibility to check against the Colab terms.
If the primary extension rule triggers, commit only the approved `training.max_epochs` increase (never change `absolute_epoch_ceiling`) and ensure all workers check out that exact same commit/ref before resuming. Keep the original `protocol-v1` tag immutable; record the cap-extension commit or a descriptive tag such as `protocol-v1-cap-extension-1` in the run records. The code permits this cap-only resume because the cap alone is excluded from the config identity hash and the training-source fingerprint must match.

## Rules enforced by the code
- the split and run manifests are loaded only after their SHA-256 matches the config (line endings normalized); no script regenerates the split
- missing or unreadable images abort; nothing is silently dropped
- training, smoke tests and pilots never read the held-out test partition; test predictions are only exported by `scripts/evaluate_runs.py` after the primary study is frozen and training is complete
- checkpoint = best validation macro-F1 (earliest epoch wins ties); only `best_checkpoint.pth` and `latest_resumable.pth` per run
- checkpoints are written locally, validated, copied to Drive under a temporary name, hash-checked and renamed; resume uses only copies listed in `persist_index.json`
- resume is refused if the run, config, split, phase or training-source fingerprint differs; a config-only commit made by the preregistered epoch-cap extension rule is allowed only when the training-source fingerprint and all non-cap settings match
- a failed run is recorded and the same run is resumed later; seeds are never swapped
- CSV logs with a different header are an error, never a silent new file

## Audit and repair record
`CODE_AUDIT_AND_REPAIR_REPORT.md` maps the 20-item consolidated review to the actual fixes, explains the items where the proposed edit was not scientifically safe, and lists the remaining data-dependent protocol-freeze gates.

## What is tested where
`python -m pytest` runs: config, split, metrics, statistics (weighted ECE/AUROC against references), operating points,
corruption, manifests, OOD manifests (these need no PyTorch). With PyTorch installed it also runs the loss tests
(numpy reference values), 12 backbone/framework forward-backward tests including Mixup/CutMix targets, augmentation tests,
checkpoint/persistence tests and end-to-end CPU tests of training, resume and the runner on a tiny synthetic dataset.

## Open items (resolve before the freeze)
- verify the R-EDL objective and the F-EDL variance expression against the original papers/code (`objective_status` in the config); the freeze gate intentionally rejects the current unresolved status strings
- review the proposed corruption severity values; record the corruption, PAD-UFES-20 and CIFAR-100 manifest hashes
- set `training.max_epochs` and `training.absolute_epoch_ceiling`; freeze `checkpoint.sync_every_n_epochs` after measuring Drive write time
- PAD-UFES-20 loading assumes the original Mendeley layout (`metadata.csv` with `patient_id, lesion_id, img_id, diagnostic`);
  the manifest builder fails loudly if the columns differ

## Data and licences
HAM10000 (CC BY-NC 4.0): cite the dataset paper and do not redistribute the images. PAD-UFES-20 and CIFAR-100: cite and follow their terms.
Code: MIT (see `LICENSE`; confirm with your institute).


### Supplemental evaluation safeguards

The ID prediction export validates its saved bundle before accepting metrics. Selective specificity is calculated on retained samples only. Corruption aggregation excludes the `clean:0` reference and requires exactly 20 corrupted conditions per run. OOD evaluation also reports the F-EDL `1 - sum(mu_k^2)` secondary score when present in both ID and OOD bundles.
