# Code Audit and Repair Report — Lightweight-EDL-Medical-v2

**Scope:** source tree supplied in `Lightweight-EDL-Medical-v2(1).zip`, compared with the consolidated thesis roadmap and three-worker Colab execution rules.

**Result:** the confirmed implementation defects listed in the 20-item review have been corrected where they were genuine. Several suggested fixes were strengthened or deliberately not applied where they would be scientifically misleading. The repository is **not yet protocol-frozen or ready to launch the primary 144 runs**: data-dependent manifests and method/corruption decisions still need to be resolved from the real data and frozen protocol.

## Verification performed

- `python -m compileall -q edl scripts`: passed.
- `python scripts/make_run_manifest.py --check`: passed; 144 rows, 144 unique run IDs, 48 per backbone; semantic design check and configured SHA-256 both pass.
- `pytest -o addopts='' -q`: **115 passed** after supplemental evaluation fixes.
- CLI `--help` smoke checks were attempted; the bulk command hit the execution timeout after the tests, so this is not claimed as a complete CLI integration test.
- The local audit runtime does **not** match `environment/colab_recorded.txt` (local Python 3.13.5, CPU Torch 2.10, torchvision 0.25 and scikit-learn 1.8 versus the recorded expected versions). This is an expected environment-check failure here, not proof of a defect in the guard. Actual T4 Colab and mounted-Drive recovery were not available for this audit.


## Supplemental post-repair audit (2026-10-12)

A follow-up review of the packaged corrected ZIP identified three real glue-code defects and one planned secondary output that was not yet wired into the OOD report:

1. **Clean condition included in corruption degradation averages — fixed.** `condition="clean:0"` is now explicitly excluded; aggregation requires exactly 20 corrupted rows per run and fails closed if the curve table is incomplete. A regression test demonstrates that the mean is divided over 20, not 21, conditions.
2. **Selective specificity calculated on all test negatives — fixed.** `apply_coverage()` now calculates specificity only among retained negative samples. The export layer no longer overwrites that result with unmasked full-test specificity. A regression test covers rejected negative samples.
3. **ID prediction bundle verifier was unused — fixed.** `export_id_predictions()` now reloads and verifies the written NPZ before writing metrics, including probability validity, dimensions, unique and ordered image IDs, finite uncertainty arrays, and expected run identity.
4. **F-EDL secondary OOD score — added.** When both the ID and OOD prediction bundles expose `total_uncertainty` (the prespecified `1 - sum(mu_k^2)` score), the evaluation now calculates its AUROC/FPR95/AUPR and balanced AUPR alongside native, MSP, and entropy scores. Non-F-EDL bundles are unaffected.

Supplemental checks: focused regression tests passed; the full automated suite passed **115 tests**. Actual dataset/T4/Google Drive acceptance tests remain outstanding as described above.

## Consolidated 20-item review

| # | Finding | Verdict and repair |
|---:|---|---|
| 1 | Wrong `build_run_manifest` import; `--check` not fatal | **Confirmed and fixed.** Import points to `edl.training.manifest`; semantic or SHA mismatch exits nonzero. A freshly generated global manifest is included. |
| 2 | Test leakage in smoke/pilot | **Confirmed and fixed.** Training never exports held-out predictions. `evaluate_runs.py` is primary-only, requires a frozen config, and checks all 144 primary statuses and their identities before opening test/OOD images. |
| 3 | BEST/LATEST can desynchronise during Drive failure or same-machine resume | **Confirmed; proposed one-line fix was insufficient.** LATEST embeds the matching best state, BEST is persisted before LATEST when dirty, each copy is hash/index checked, and a short-lived transaction backup restores the previously indexed copy if a write is interrupted before the index update. Regression tests simulate Drive/index failures. |
| 4 | Pilot report can calculate a cap with missing pilots | **Confirmed and fixed.** All five pilot records must have valid state, stop reason, epoch values and cap-consistent status; otherwise no cap is calculated. |
| 5 | `--finalize-cap` can make a cap-limited run look converged | **Confirmed and fixed with a stricter policy.** The flag is rejected; cap-limited remains `cap_limited`. The resource-ceiling case is explicitly recorded and reported, never relabelled as early stopping. |
| 6 | Mixup/CutMix CPU/GPU target-device mismatch | **Confirmed and fixed.** Soft targets are moved to the image device and floating dtype before mixing. |
| 7 | Efficiency benchmark uses random weights and lacks quality metrics | **Confirmed and fixed.** The script verifies and loads each selected seed-42, Base-flips best checkpoint, then records Macro-F1, ECE and PAD-UFES-20 native AUROC alongside latency, parameters, MACs and checkpoint size. |
| 8 | Missing required evaluation summaries/plots | **Confirmed and fixed in the CLI pipeline.** ID exports contain risk-coverage AURC and validation-selected melanoma operating points; post-training evaluation writes all three OOD score families for PAD-UFES-20/CIFAR-100; the stats script emits per-corruption ECE/ECE deltas with intervals and Macro-F1; plot generation includes reliability, risk-coverage and corruption curves. Aggregation checks deliverable completeness. |
| 9 | Corruption manifest/hash not checked during evaluation | **Confirmed and fixed.** Evaluation checks normalized-EOL manifest SHA, exact image-condition coverage, uniqueness, source hashes, derived seeds, severity parameters, and generated corrupted-pixel hashes. |
| 10 | Flips applied to every augmentation condition | **Not inherently a bug; the protocol was ambiguous.** This version explicitly chooses flips as the shared baseline: Mixup/CutMix are added on the GPU; RandAugment is added after CPU-side flips. The code, YAML and protocol text agree and bilinear interpolation is explicit. A strictly isolated augmentation comparison would be a different protocol and must not be changed silently. |
| 11 | Freeze gate accepts draft/unresolved config and nested typos | **Confirmed and fixed.** Strict nested key sets are validated; primary use requires `status: FROZEN`, filled hashes/paths, final cap and resource ceiling, and no unresolved `verify`/`PROPOSED`/placeholder markers. The current config intentionally remains DRAFT, so the gate correctly blocks launch. |
| 12 | Missing pre-pilot numerical diagnostics | **Confirmed and fixed.** Epoch logs contain loss components, maximum gradient norm, maximum evidence/alpha and concentration, F-EDL clamp-activation rate, and non-finite failures are recorded as run failures rather than silently accepted. |
| 13 | Historical-pilot text conflicts with the clean-start policy | **The suggested deletion was not safe without establishing the facts.** The code/results are clean-start and old artefacts are not reused, but historical project notes say some directional hypotheses were previously informed by pilots and old pilot training overlapped images now assigned to the new test partition. The protocol preserves an explicit provenance/limitation disclosure instead of incorrectly claiming that the historical record never existed. Reword only if documentary evidence shows that note is false. |
| 14 | Completed runs skipped before checking identity | **Confirmed and fixed.** Run identity is constructed and compared before completed/cap-limited shortcut returns. |
| 15 | Bundle, image-order, OOD/statistics provenance insufficiently checked | **Confirmed and hardened.** Active run/config/split identities are checked for statuses, checkpoints and prediction bundles; ID, PAD and CIFAR ordering is verified; analysis scripts check eligible statuses; bootstrap outputs include replicate count, RNG seed and config/split identity. Aggregation validates expected ID/OOD metric keys and curve provenance. |
| 16 | Workers can use different primary epoch caps | **Confirmed and fixed.** Primary CLI rejects `--max-epochs`; primary cap must come from the same committed YAML on all workers. Only the current cap is excluded from the identity hash; resource ceiling and all other protocol settings remain hashed. A training-source fingerprint must match for cap-only resume. |
| 17 | Persistent DataLoader workers alter augmentation RNG behavior on resume | **Confirmed as a reproducibility risk and addressed.** `persistent_workers: false`; loader/sampler RNG state is checkpointed/restored at epoch boundaries. This improves epoch-boundary resume consistency but does not promise bitwise deterministic CUDA execution across different hardware/runtime builds. |
| 18 | Drive root and split manifest filename inconsistent | **Confirmed and fixed.** Default root is `/content/drive/MyDrive/Lightweight-EDL-Medical-v2`; the split filename is `protocol/split_manifest_v1.csv` throughout. |
| 19 | OOD metadata/probability/log integrity checks incomplete | **Confirmed and fixed.** Metadata is passed to split validation when available; PAD null/duplicate IDs and duplicate basenames are rejected; CIFAR-100 must have the official 10,000 ordered test records and matching hashes; probability validation checks finite values/range/row sums; worker-log duplicates and status contradictions are checked. |
| 20 | Runtime version parity not enforced on every worker | **Confirmed and fixed in the startup guard.** Python major/minor and Torch/torchvision major/minor are checked against the recorded environment, and scikit-learn must match exactly. Each worker must pass this guard before primary training. The actual remote T4 environment still needs verification. |

## Additional safeguards added during the combined repair

- Cap-limited pilots can resume after the cap is extended instead of being silently skipped as completed; code changes to training-critical Python reject resume, while the approved cap-only config change is allowed.
- Partial OOD reruns with `--overwrite` replace only requested dataset summaries; they do not erase the other OOD dataset's existing metrics.
- Reused PAD-UFES-20 and CIFAR-100 prediction bundles are checked against the frozen manifest ordering, not just a status identity.
- Test evaluation is blocked until every row of the 144-run primary matrix is terminal and identity-compatible. Runs that hit the explicit absolute resource ceiling are allowed only with a separately recorded limitation state.
- `aggregate_results.py` checks the required metric/operating-point/OOD fields, and does not merge primary corruption curves into smoke/pilot reports.
- The training source fingerprint includes constants and shared I/O/environment helpers so changes to those training-critical semantics cannot be mistaken for a safe cap-only update.

## Protocol freeze is still deliberately blocked

The following are not code bugs and must **not** be filled with guessed values:

1. `configs/primary.yaml` remains `DRAFT`; `training.max_epochs`, `training.absolute_epoch_ceiling`, and `pilot.resource_ceiling` are unset until the smoke/pilot and resource assessment.
2. `protocol/split_manifest_v1.csv` and its SHA-256 must be generated from the actual HAM10000 metadata/images, semantically validated, committed, and recorded in the config.
3. PAD-UFES-20 and CIFAR-100 manifests/hashes must be generated from the selected releases/files.
4. The corruption manifest/hash must be generated from the actual held-out images after the severity grid has been reviewed and the chosen severities frozen.
5. R-EDL's exact training objective and the chosen F-EDL variance/objective implementation need the promised comparison with the official references. The explicit `objective_status` fields remain unresolved by design.
6. The Drive checkpoint-sync cadence must be frozen after real Colab I/O measurements.

No primary dataset download, full 144-run training, real T4 performance benchmark, or mounted-Drive failure test was possible in this audit environment. The passing unit suite is necessary but is not a substitute for those acceptance checks.

## New run manifest

`protocol/run_manifest_v1.csv` contains 144 unique rows (48 per backbone).

SHA-256: `36bc2026ffdbabe82a702b39f8c38f43f4b8bce42455603408ed211b678a1c6c`

Recheck with `python scripts/make_run_manifest.py --check` before committing the repository snapshot.
