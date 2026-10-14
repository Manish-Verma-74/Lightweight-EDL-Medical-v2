# AGENTS.md — M.Tech Thesis Context & Protocol Guardrails

Last checked against the `Lightweight-EDL-Medical-v2` repo (`protocol-v1` tag frozen at commit `60c3d21` on 14 Oct 2026 to isolate the 144 primary runs; `main` continues to receive documentation, post-primary evaluation, and subsequent stage updates). Where this file and `configs/primary.yaml` / `protocol/protocol_v1.md` ever disagree, flag it. Do not reconcile silently.

## 1. Project Identity
- **Thesis title:** Calibration Robustness of Lightweight Evidential Deep Learning under Data Augmentation and Progressive Image Degradation in Medical Image Classification.
- **Institution:** ABV-IIITM Gwalior.
- **Programme:** M.Tech CSE, 2025–2027.
- **Supervisor:** Prof. Karm Veer Arya.
- **Research approach:** An empirical comparison of lightweight CNN backbones, uncertainty frameworks, and data augmentation. This is not a proposal for a new EDL method (`configs/primary.yaml` states "empirical comparison; no new-method claim"). Do not claim universal superiority or novelty without supporting evidence.
- **Compute and workflow:** Primary GPU experiments run on Google Colab (NVIDIA T4). Source is edited in VS Code, committed and pushed with GitHub Desktop, and each Colab worker checks out the frozen tag `protocol-v1`. Persistent artifacts (checkpoints, run summaries, worker logs) go to Google Drive. Raw images are read from Colab local disk (`/content/data`), never from mounted Drive. The local machine is for editing and lightweight checks only; verify its actual hardware before recording specifications.
- **Frozen primary-run release:** `configs/primary.yaml` has `status: FROZEN` and `protocol_v1.md` records all freeze gates as complete. The git tag `protocol-v1` (commit `60c3d21`) isolates the 144 primary runs on Colab so that ongoing documentation, plotting, or next-stage development on `main` never alters active primary training runs.

## 2. Primary Experimental Matrix
The primary matrix contains 144 runs:
`3 backbones × 4 frameworks × 4 augmentation conditions × 3 seeds = 144`.

The global run manifest `protocol/run_manifest_v1.csv` (144 rows, 48 per backbone) is immutable.
- **SHA-256:** `36bc2026ffdbabe82a702b39f8c38f43f4b8bce42455603408ed211b678a1c6c`
- **Run ID pattern:** `{backbone}__{framework}__{augmentation}__seed{seed}`

### Backbones
- `efficientnet_b0`
- `mobilenet_v3_small`
- `shufflenet_v2`

All three start from ImageNet-pretrained torchvision weights (`IMAGENET1K_V1`).

### Frameworks
- `softmax`
- `edl` — Standard Evidential Deep Learning
- `r_edl` — R-EDL with configured prior weight $\lambda = 0.1$
- `f_edl` — F-EDL without spectral normalization

### Training augmentation conditions
- `base_flips`: horizontal and vertical flips, $p = 0.5$ each.
- `mixup`: Mixup with $\alpha = 0.2$.
- `cutmix`: CutMix with $\alpha = 1.0$.
- `randaugment`: two operations, magnitude 9.

**Important implementation detail:** The protocol defines horizontal and vertical flips as shared CPU-side transforms for all four conditions. Mixup and CutMix add batch-level mixing on the GPU; RandAugment adds its operations after the shared flips. A strictly isolated augmentation comparison would be a different protocol. Do not remove or add transforms silently, because that changes the experimental comparison.

### Training seeds
Use exactly `42`, `123`, and `2024` unless an explicitly approved protocol revision says otherwise.

### Primary dataset
- **Dataset:** HAM10000 (CC BY-NC 4.0: cite it, do not redistribute images).
- **Number of classes:** 7.
- **Class order:** `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- **Input resolution:** $224 \times 224$.
- **Preprocessing:** direct resize (bilinear), RGB conversion, and ImageNet normalization.
- **Known limitation:** direct resizing distorts the original image aspect ratio.

### Fixed split
Use the committed split manifest `protocol/split_manifest_v1.csv`. The specified generation method is:
- `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`, scikit-learn `1.6.1`.
- **Group column:** `lesion_id`.
- **Stratification label:** `dx`.
- **Training folds:** 0, 1, 2, 5, 6.
- **Validation fold:** 4.
- **Test fold:** 3.

Expected counts from the recorded split:
- **Train:** 7,179 images.
- **Validation:** 1,400 images.
- **Test:** 1,436 images.
- **Total:** 10,015 images across 7,470 lesions.
- **Split-manifest SHA-256:** `df7be4c42d83eb678a3eceef6772cd752a7654086787351ebef0552344ce894c`.

The training seeds are separate from the fixed split-generation seed. Any missing, unreadable, duplicated, or mismatched record aborts processing. Never regenerate the split during primary execution.

### OOD evaluation
- **ID reference:** HAM10000 held-out test partition.
- **Near-OOD/domain shift:** PAD-UFES-20, Mendeley Data v1 (original release), via `protocol/ood_pad_ufes20_manifest.csv` (2,298 images). Diagnostic classes overlap with HAM10000, so do not describe this as pure semantic OOD.
- **Far-OOD:** CIFAR-100 official test split (10,000 images), resized to $224 \times 224$, via `protocol/ood_cifar100_manifest.csv`. Record the limitation that upsampling low-resolution images can make detection artificially easy.

Manifest hashes (also in `protocol/manifest_hashes.txt` and `configs/primary.yaml`):

| Manifest | SHA-256 |
|---|---|
| `ood_pad_ufes20_manifest.csv` | `502782520af85d54abfe9a9d87c3c7666e33ee9b0f39153016549d14e44eafd3` |
| `ood_cifar100_manifest.csv` | `2f57539b242c50f182903d06c6ce8b22c278d76d95d2d8a8f524de2862fa9fa1` |
| `corruption_manifest.csv` | `19aa345452c8d889292a35ece254865bbf69c45115afad95fba5484ab424f217` |

Use the existing committed manifests and hashes. Do not silently replace dataset versions, image order, or evaluation splits.

## 3. Training Configuration
Preserve the approved experiment configuration:
- **Optimizer:** AdamW.
- **Learning rate:** `0.0001`.
- **Weight decay:** `0.01`.
- **Batch size:** `32`.
- **Scheduler:** none.
- **Precision:** FP32 (no autocast or GradScaler anywhere in the training code).
- **Class imbalance:** training-only `WeightedRandomSampler`.
- **Checkpoint selection:** best validation Macro-F1.
- **Tie-breaking:** earliest epoch.
- **Early-stopping patience:** 8 epochs.
- **Primary epoch cap:** 57, set after the convergence pilot as `max(best pilot epochs) + 8 + 15`.
- **Absolute epoch ceiling:** 100, chosen before primary training. If a run hits the cap before early stopping, the global cap may be extended by 20 only under the preregistered rule, never above 100, and all workers must use the same approved config/commit.
- The primary CLI rejects `--max-epochs`; the cap comes only from the committed YAML.
- **Data loader:** `num_workers: 2`, `persistent_workers: false`.

The cap is a maximum, not a requirement to train every run to that epoch. Early stopping is expected. A run that hits the cap stays labelled `cap_limited`; never relabel it as converged.

The convergence pilot used five configs at seed 42: EfficientNet-B0 Softmax Base Flips, EfficientNet-B0 Softmax RandAugment, EfficientNet-B0 R-EDL CutMix, MobileNetV3-Small F-EDL Mixup, ShuffleNetV2 F-EDL Mixup.

Use `best_checkpoint.pth` for the selected model and `latest_resumable.pth` to resume an interrupted run. Checkpoints are written locally, validated, then copied to Drive (synced every epoch) and accepted as persistent only after integrity checks pass. Do not delete valid checkpoints or restart completed work unnecessarily.

Do not change seeds, split indices, hyperparameters, augmentation settings, loss objectives, checkpoint selection, epoch limits, or metrics without explicit approval and the required protocol/version documentation.

## 4. Targets and Loss Compatibility
### Hard and soft targets
- Base Flips and RandAugment retain integer class labels during image transformation.
- Mixup and CutMix produce probability-distribution targets with shape `[B, K]`.
- Mixup uses the same mixing coefficient for the images and targets.
- CutMix weights must reflect the actual clipped patch area.
- Soft targets must remain distributions. Never convert them to hard class indices using `argmax()` or `.long()`.

The shared target converter accepts hard labels `[B]` or soft targets `[B, K]`. It validates class dimensions, values in `[0, 1]`, and row sums of one.

Validation, test, OOD, and corruption evaluation must use deterministic preprocessing without training-time Mixup or CutMix. Classification metrics use the actual integer ground-truth labels.

### Softmax
- **Predictive probabilities:** `softmax(logits)`.
- **Loss:** cross-entropy that supports both hard and soft targets.
- **Native rejection/OOD score:** `1 - max(probabilities)`.
Softmax does not have Dirichlet vacuity.

### Standard EDL
- **Evidence:** `relu(logits)`.
- **Concentration:** `alpha = evidence + 1`.
- **Total concentration:** `S = sum(alpha)`.
- **Predictive probabilities:** `alpha / S`.
- **Native vacuity score:** `K / S`.
- **Loss:** squared error plus expected Dirichlet variance plus an annealed KL regularizer.

### R-EDL
- **Implemented concentration:** `alpha = relu(logits) + lambda`, with configured `lambda = 0.1`.
- **Predictive probabilities:** `alpha / S`.
- **Native uncertainty mass:** `lambda * K / S` (the corrected expression, not the uncorrected `K / S` in the paper PDF).
- **Loss:** squared error between soft target and predictive mean plus the annealed KL term. It does not add the standard EDL expected-variance term.
The protocol records this as the verified objective used for this study. Report any differences in reduction or parameterization from other implementations as implementation choices, not as a new method.

### F-EDL
- **Flexible Dirichlet head:** `alpha = exp(clamp(g_alpha, -20, 20))`.
- **Allocation probabilities:** `p = softmax(g_p)`.
- **Concentration-related parameter:** `tau = softplus(g_tau)`.
- **Predictive mean:** `mu = (alpha + tau * p) / (sum(alpha) + tau)`.
- **Native uncertainty score:** sum of class-wise predictive variances.
- **Secondary uncertainty score:** `1 - sum(mu ** 2)`.
- **Loss:** batch mean of the per-example sum of `(y - mu)^2 + variance`, plus the per-example sum of `(y - p)^2`.
- Spectral normalization remains disabled for the planned controlled variant.
Document the log-alpha clamp and batch-mean loss reduction when discussing differences from the reference implementation.

### KL annealing and soft-target adjustment
For Standard EDL and R-EDL, the configured one-based epoch schedule is:
`beta = min(1.0, epoch / 10)` (`0.1` in epoch 1, `1.0` from epoch 10).

The implemented target-dependent concentration adjustment is:
`alpha_tilde = alpha * (1 - target) + target`.

The implementation applies this adjustment in the EDL and R-EDL KL terms. Preserve the frozen implementation, but distinguish target-format compatibility from verification that the objective faithfully reproduces a publication. Softmax and F-EDL do not automatically inherit EDL's KL schedule.

## 5. Evaluation Metrics
Use the metrics defined by the experiment configuration:
- **Classification:** Accuracy, Macro-F1, class-wise recall/sensitivity.
- **Calibration:** top-label ECE with 15 equal-width bins and macro class-wise ECE (15 bins).
- **Probability quality:** multiclass Brier score, mean per-image sum over classes (not divided by the number of classes).
- **Selective prediction:** risk–coverage curves and AURC. Melanoma operating thresholds (specificity 0.90 / 0.95, coverage 0.80 / 0.90) come from validation predictions only and are applied to test predictions.
- **OOD detection:** AUROC, FPR95, and AUPR with OOD as the positive class and higher score meaning more OOD. Report native, MSP, and entropy scores, plus the F-EDL secondary score where available.
- **Efficiency:** parameter count, checkpoint/model size, MACs (`Conv2d` and `Linear` layers only), and latency on a T4 (FP32, 1 CPU thread, 50 warmup and 200 timed iterations), using the seed-42 Base Flips best checkpoint as reference.

Do not silently add or substitute metrics, alter ECE binning, change OOD score directions, or change the Brier-score convention. NLL is not part of the metric list in the current written protocol; any addition must be explicitly approved and documented.

Report the mean and standard deviation across the three training seeds where appropriate. Keep seed-to-seed variation distinct from bootstrap uncertainty.

### Primary statistical contrasts
- **P1:** CutMix minus Base ECE within each framework; four co-primary contrasts with 98.75% intervals. Expected directions: Softmax negative, F-EDL negative, Standard EDL positive, R-EDL positive.
- **P2:** F-EDL native OOD score minus Softmax MSP uncertainty AUROC on PAD-UFES-20, Base Flips; expected direction positive.
- **P3:** efficiency comparison is descriptive; no unregistered composite score.
- **P4:** CutMix minus Base mean ECE degradation across the 20 corruption conditions (the clean condition is excluded); expected direction negative.

**Bootstrap:** 10,000 replicates, RNG seed `20261009`; lesion-cluster resampling for HAM10000, patient-cluster for PAD-UFES-20, image-level for CIFAR-100. Per-corruption curve intervals use 2,000 paired lesion-cluster replicates.

The P1 directions were informed by earlier validation-fold pilots. Disclose this provenance and do not claim a fully hypothesis-naive preregistration.

## 6. Progressive Image Degradation
**Frozen (do not change):**
- **Corruption families:** Gaussian noise ($\sigma \in \{0.08, 0.12, 0.18, 0.26, 0.38\}$), Gaussian blur ($\sigma \in \{1, 2, 3, 4, 6\}$), brightness shift ($0.1$ to $0.5$), and contrast factor ($0.4, 0.3, 0.2, 0.1, 0.05$).
- Five severities per family, giving 20 corrupted conditions plus the clean reference.
- The corruption manifest, exact severity parameters, source-image ordering, global seed `20261009`, and hashes.
Do not reinterpret or change corruption definitions after examining test results.

**Model scope (undecided, provisional):** `configs/primary.yaml` currently lists a default subset: EfficientNet-B0 and MobileNetV3-Small, Softmax and F-EDL, Base Flips and CutMix. The thesis plan is to decide the final corruption scope only after the 144 primary runs, choosing which configurations go to further stages or whether a full factorial is needed. Until then:
- Treat the YAML subset as a provisional default, not a final decision.
- Changing it is a documented protocol revision, not a silent edit.
- Define the selection criteria before looking at test-set corruption results. Choosing configurations post hoc from test results weakens the held-out claim.
- Keep Base Flips and CutMix in whatever scope is chosen, because contrast P4 needs both.

If the config, manifest, and written protocol disagree about severities or freeze status, flag the discrepancy rather than guessing.

## 7. Secondary Validation and Optional Directions
### Chest X-Ray external validation (core, not yet in the frozen protocol)
External validation on a Chest X-Ray dataset is a core part of the thesis. It is not yet part of the frozen protocol, config, manifests, or code. Before any Chest X-Ray run:
- Write a protocol addendum covering the exact dataset and release, split and manifest with hash, preprocessing, which backbone / framework / augmentation configurations are applied, seeds, and metrics.
- Treat it as an approved direction awaiting specification, not as permission to improvise.
- Keep it separate from the 144-run HAM10000 primary matrix and from the primary hypotheses P1 to P4.

### Optional exploratory work (only if time remains)
- SNGP pilot and DDU pilot, as comparisons with state-of-the-art methods.
- Custom EDL loss built from components of existing variants.

These are outside the 144-run matrix. If run, label them exploratory, give each a written protocol (including a protocol audit before comparison), and keep their results out of the primary claims. A custom loss would conflict with the current "no new-method claim" framing, so it needs an explicit protocol revision first. The F-EDL ablation is also optional.

## 8. Repository Structure and Validation
Use the actual repository layout:
- `configs/primary.yaml` — frozen machine-readable experiment configuration.
- `protocol/` — `protocol_v1.md`, and the split, run, OOD, and corruption manifests with `manifest_hashes.txt`.
- `edl/data/` — HAM10000 loading, split, OOD, and corruption code.
- `edl/augment.py` — Mixup and CutMix implementations.
- `edl/losses/` — framework-specific losses and the KL schedule.
- `edl/frameworks.py` — framework loss and prediction interfaces.
- `edl/models/` — backbones and model wrappers.
- `edl/training/` — engine, runner, pilot logic, checkpoints, and resume behaviour.
- `edl/evaluation/` — metrics, predictions, export, statistics, efficiency, operating points.
- `scripts/` — experiment launchers, verification, evaluation, aggregation, plotting.
- `tests/` — automated tests.
- `notebooks/colab_runner.ipynb` and `environment/colab_recorded.txt` — Colab workflow and recorded environment.
- `CODE_AUDIT_AND_REPAIR_REPORT.md` — historical audit; its DRAFT/open-gate statements are superseded by the frozen config.

Run unit tests with:
```bash
python -m pytest -q