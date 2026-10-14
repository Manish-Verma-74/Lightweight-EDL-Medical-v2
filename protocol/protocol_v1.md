# Protocol v1 — Frozen primary experimental protocol

**Status: FROZEN protocol decisions; release tag now done.** This written protocol and `configs/primary.yaml` describe the same frozen decisions. After committing the synchronized release files, create the `protocol-v1` Git tag on that exact commit before primary execution. Do not regenerate or edit the committed manifests during routine execution.

## Design

Three lightweight backbones (EfficientNet-B0, MobileNetV3-Small, ShuffleNetV2 1.0x) × four frameworks (Softmax, standard EDL, R-EDL, F-EDL without spectral normalization) × four augmentation conditions (Base flips, Mixup, CutMix, RandAugment) × three training seeds (42, 123, 2024) = **144 runs**. The global run manifest (`protocol/run_manifest_v1.csv`) is immutable. Each worker loads the same manifest and filters it locally by backbone; workers do not create per-worker copies of the manifest.

## Data and preprocessing

- **HAM10000:** seven classes; RGB; direct resize to 224 × 224; ImageNet normalization. Direct resizing changes the original aspect ratio and is a documented limitation.
- **Split:** `StratifiedGroupKFold(7, shuffle=True, random_state=42)`, grouped by `lesion_id`, stratified by `dx`, using scikit-learn 1.6.1. Train folds 0, 1, 2, 5, 6; validation fold 4; test fold 3. The committed split manifest and SHA-256 in `configs/primary.yaml` are authoritative. Reference counts are 7,179 / 1,400 / 1,436 images (10,015 total) and 7,470 lesions.
- **Data integrity:** any missing, unreadable, duplicated, or mismatched record aborts processing. Do not silently drop records or regenerate the split during primary execution.
- **Augmentations:** horizontal and vertical flips (probability 0.5 each) form the shared CPU-side baseline. `base_flips` uses the flips alone; Mixup and CutMix add GPU batch mixing; RandAugment adds two operations at magnitude 9 after the flips. Evaluation uses deterministic preprocessing without stochastic training augmentation.
- HAM10000 is CC BY-NC 4.0. Cite the dataset and do not redistribute its images.

## Training and checkpointing

AdamW, learning rate 0.0001, weight decay 0.01, batch size 32, no scheduler, FP32, and a training-only `WeightedRandomSampler`. Checkpoints are selected by validation Macro-F1; the earliest epoch wins ties. Patience is 8 epochs. The initial global epoch cap is 57, determined as `max(best pilot epochs) + 8 + 15`. If a primary run reaches the cap before early stopping, the global cap can be extended by 20, only under the preregistered rule and never above the absolute resource ceiling of 100. All workers must use the same approved configuration/commit for any cap-only extension.

Each run has two persistent checkpoint roles: `best_checkpoint.pth` (selected model weights) and `latest_resumable.pth` (full resumable state). Checkpoints are written to local Colab storage, validated, copied to Drive using the safe persistence procedure, and accepted as persistent only after integrity checks succeed. The synchronization cadence is one epoch. Colab-local `/content/` storage is temporary; the successfully persisted Drive checkpoint is the recovery point.

## Learning frameworks and documented implementation choices

- **Softmax:** `q = softmax(z)`; native uncertainty is `1 - max(q)`.
- **Standard EDL:** evidence is `relu(z)`, `alpha = evidence + 1`, `q = alpha / S`, and native uncertainty is `K / S`. The loss is the squared-error term plus the expected categorical variance term and one-based epoch annealed KL regularization. The KL soft-target rule is `alpha_tilde = alpha * (1 - t) + t`.
- **R-EDL:** `alpha = relu(z) + lambda`, with `lambda = 0.1`; probabilities are `q = alpha / S`, and the native uncertainty mass uses the corrected `lambda * K / S` expression. The implemented objective uses the squared error between the soft target and predictive mean plus the annealed KL term; it does not add the standard EDL expected-variance term. The soft-target KL rule is `alpha_tilde = alpha * (1 - t) + t`, with one-based epoch annealing. This is the verified objective implementation used for this study; any differences in reduction/parameterization from another implementation must be reported as implementation choices, not presented as a new method.
- **F-EDL without spectral normalization:** three independent heads produce positive `alpha = exp(clamp(g_alpha(z), -20, 20))`, `p = softmax(g_p(z))`, and `tau = softplus(g_tau(z))`. The predictive mean is `mu = (alpha + tau*p) / (sum(alpha) + tau)`. The class-wise variance is

  `Var(pi_k) = mu_k*(1-mu_k)/(alpha_0 + tau + 1) + tau^2*p_k*(1-p_k)/((alpha_0 + tau)*(alpha_0 + tau + 1))`.

  The native score is the sum of class-wise variances; the secondary score is `1 - sum(mu_k^2)`. The implemented loss is the batch mean of the per-example sum of `(y - mu)^2 + variance` plus the per-example sum of `(y - p)^2`. Spectral normalization is disabled. The log-alpha clamp `[-20, 20]` and batch-mean reduction are explicit implementation choices and must be disclosed when comparing to implementations with different settings.

The equations and code paths above were checked against the selected reference formulations before freezing. This statement documents the implementation selected for this experiment; it does not claim that every optional setting of the reference repositories is reproduced.

## Evaluation metrics

For each ID split, report Accuracy, Macro-F1, class-wise recall/sensitivity, top-label ECE (15 equal-width bins), macro class-wise ECE, multiclass Brier score (mean per-image sum over classes, not divided by `K`), risk–coverage curves and AURC. The code records the selected epoch using validation Macro-F1 only. Melanoma operating thresholds are selected on validation predictions and applied to test predictions.

For OOD evaluation, use HAM10000 test as ID, PAD-UFES-20 v1 as near/domain-shift OOD, and the official CIFAR-100 test split as far OOD. Report AUROC, FPR95, AUPR with OOD as positive and higher scores indicating greater OOD likelihood. PAD-UFES-20 has overlapping diagnostic classes and is interpreted as domain/acquisition shift, not pure semantic OOD. Upsampling 32 × 32 CIFAR-100 images to 224 × 224 may make far-OOD detection easier.

Corruptions are applied after resize and conversion to float `[0,1]`, before clipping and ImageNet normalization. The four fixed families are Gaussian noise, Gaussian blur, brightness and contrast, with five severities each. The committed corruption manifest binds each test image and condition to deterministic generation settings and hashes. The 20 corrupted conditions are evaluated in addition to the clean reference.

## Primary statistical contrasts

- **P1:** CutMix minus Base ECE within framework; four co-primary contrasts use 98.75% intervals. Expected directions: Softmax negative, F-EDL negative, standard EDL positive, R-EDL positive.
- **P2:** F-EDL native OOD score minus Softmax MSP uncertainty AUROC on PAD-UFES-20, using Base augmentation; expected direction positive.
- **P3:** efficiency comparison is descriptive; no unregistered composite score.
- **P4:** CutMix minus Base mean ECE degradation across the 20 corruption conditions; expected direction negative.

Use 10,000 bootstrap replicates with RNG seed `20261009` for primary contrasts. Use lesion-cluster resampling for HAM10000, patient-cluster resampling for PAD-UFES-20 where identifiers are valid, and image-level resampling for CIFAR-100. Secondary per-corruption curve intervals use the configured 2,000 paired lesion-cluster bootstrap replicates. Report training-seed variability separately from bootstrap sampling uncertainty.

## Freeze acceptance record

The following gates were completed for the frozen release:

- [x] Split manifest was generated from the selected HAM10000 source, semantically validated, hashed and committed.
- [x] The 144-row global run manifest was validated, hashed and committed.
- [x] PAD-UFES-20 v1 and CIFAR-100 manifests were generated, validated, hashed and committed.
- [x] Corruption severities were reviewed using the example grid; the generated corruption manifest was validated, hashed and committed.
- [x] The selected R-EDL objective and F-EDL equations/implementation choices were checked and documented above.
- [x] Updated reporting code, evaluation scripts, and their test suite passed in the recorded Colab environment; smoke and five-run convergence pilot completed.
- [x] The initial epoch cap is 57; patience is 8; the absolute resource ceiling and pilot resource ceiling are both 100.
- [x] Drive checkpoint synchronization cadence is one epoch, selected after the smoke/transfer checks.
The scientific and implementation freeze gates above are complete. As a release operation, commit and push this synchronized snapshot, then create and push `protocol-v1` on that exact commit before primary execution.

Routine post-freeze sessions must verify the existing hashes and use the frozen tag. Do not rerun split/OOD/corruption manifest generators or alter the configuration in a routine worker session. A genuine scientific defect requires a documented protocol revision and consistent reruns of affected experiments.

## Historical pilot provenance and limitations

The new primary experiment starts from the v2 source tree and explicit ImageNet-pretrained weights; earlier checkpoints, split manifests, predictions, and result files are not imported as primary evidence. Earlier project notes nevertheless indicate that the directional P1 hypotheses were informed by validation-fold pilots and that some images in the current test partition appeared in earlier pilot training under a previous split. Disclose these facts; do not claim that the hypothesis origins were independent of all prior pilots or that the held-out images have never appeared in any earlier project. The new held-out partition remains untouched by v2 training and model selection.

Other limitations: direct-resize aspect-ratio distortion; three training seeds; CIFAR-100 upsampling; PAD-UFES-20 diagnostic overlap/domain shift; F-EDL without spectral normalization; and lesion grouping (rather than patient grouping) for HAM10000.
