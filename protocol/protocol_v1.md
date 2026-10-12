# Protocol v1 (written protocol; the machine-readable version is `configs/primary.yaml`)

Status: DRAFT until the freeze checklist at the end is complete and `protocol-v1` is tagged.

## Design
3 backbones (EfficientNet-B0, MobileNetV3-Small, ShuffleNetV2 1.0x) x 4 frameworks (Softmax, EDL, R-EDL,
F-EDL without spectral normalization) x 4 augmentations (Base (flips), Mixup, CutMix, RandAugment) x 3 training
seeds (42, 123, 2024) = **144 runs**. The global run manifest (`protocol/run_manifest_v1.csv`) is immutable;
workers filter it locally and never edit it.

## Data
- HAM10000, seven classes, direct resize to 224 x 224 (aspect ratio distorted; documented limitation), ImageNet normalization.
- Split: `StratifiedGroupKFold(7, shuffle=True, random_state=42)`, groups = `lesion_id`, labels = `dx`,
  scikit-learn 1.6.1. Train folds 0,1,2,5,6; validation fold 4; test fold 3. Generated once, hashed, committed.
  Earlier reference counts (regression expectations only): 7,179 / 1,400 / 1,436 images, 10,015 images, 7,470 lesions.
- HAM10000 is CC BY-NC 4.0: cite the dataset, never redistribute images.
- Augmentation definition: horizontal/vertical flips (configured probability 0.5 each) are the common CPU-side baseline transforms for all four conditions. `base_flips` uses these alone; Mixup and CutMix add batch-level GPU mixing; RandAugment adds two operations at magnitude 9 after the shared flips. Resize/normalization are common to every condition.

## Training
AdamW, lr 1e-4, weight decay 0.01, batch 32, no scheduler, FP32, training-only WeightedRandomSampler.
Checkpoint = best validation macro-F1 (earliest epoch wins ties). Patience 8. Epoch cap
E_max = max(best epochs of the five pilots) + 8 + 15, with the preregistered extension rule
(+20 epochs for cap-limited runs, validation metrics only, up to an absolute ceiling set before training).
Two persistent checkpoint roles per run: `best_checkpoint.pth`, `latest_resumable.pth`.
KL annealing (EDL, R-EDL): min(1, epoch/10) with ONE-BASED epochs. Soft-target KL: alpha_tilde = alpha(1-t)+t.
Training logs include per-epoch loss parts, global gradient-norm maximum, evidence/alpha and concentration maxima,
non-finite detection, and F-EDL log-alpha clamp activation rate. F-EDL currently clamps raw log-alpha to [-20, 20]
for numerical safety and uses mean batch reduction of the stated per-example objective; this is part of the frozen
implementation and must be disclosed when comparing with implementations using other reduction conventions.

## Frameworks
Softmax: q = softmax(z); native score 1 - max q.
EDL: alpha = relu(z) + 1; q = alpha/S; native score K/S.
R-EDL: alpha = relu(z) + lambda (lambda = 0.1); q = alpha/S; native score lambda*K/S (corrected uncertainty mass).
F-EDL (no spectral norm): mu = (alpha + tau p)/(sum alpha + tau); native score = sum of class-wise variances;
secondary 1 - sum mu^2.
**Open before freeze:** verify the R-EDL objective and the F-EDL variance expression against the original papers/code.
Anything that deviates is labelled a controlled adaptation in the thesis.

## Metrics (see `edl/evaluation/metrics.py`)
ECE 15 equal-width bins, right-inclusive; macro-cwECE; Brier (not divided by K); risk-coverage/AURC
(stable order on ties); OOD AUROC/FPR95/AUPR with OOD positive, larger score = more OOD.

## Primary contrasts (preregistered)
P1 CutMix - Base ECE per framework (98.75% intervals; directions: Softmax -, F-EDL -, EDL +, R-EDL +). Earlier
project notes describe these directions as informed by validation-fold pilots; disclose this provenance and do not
claim fully hypothesis-naive preregistration. The old pilot artifacts are not reused as primary evidence. P2 AUROC(F-EDL native) - AUROC(Softmax MSP) on PAD-UFES-20, Base augmentation (+).
P4 CutMix - Base mean ECE degradation over 20 corrupted conditions (-). P3 efficiency is descriptive.
10,000 bootstrap replicates per seed, RNG seed 20261009, lesion clusters (HAM10000), patient clusters (PAD-UFES-20).
Seed variability and bootstrap uncertainty are reported separately. The secondary per-corruption curve intervals use 2,000 paired lesion-cluster bootstrap replicates (RNG seed 20261009); the preregistered P1/P2/P4 contrasts use 10,000.

## Freeze checklist
- [ ] split manifest generated; hash in `configs/primary.yaml`
- [ ] run manifest hash in `configs/primary.yaml`
- [ ] PAD-UFES-20 and CIFAR-100 manifests hashed in the config
- [ ] corruption severities inspected on the example grid, manifest hashed
- [ ] R-EDL objective and F-EDL equations verified
- [ ] CPU tests pass locally; smoke test and five-run pilot done; `training.max_epochs` and the absolute ceiling set
- [ ] Drive sync cadence frozen from measured checkpoint-write time
- [ ] commit, tag `protocol-v1`

## Historical pilot provenance and limitations to disclose
The new primary experiment starts from the new source tree and explicitly initialized ImageNet weights; no earlier
project checkpoint, split, manifest, prediction, or result file is imported as primary evidence. Earlier project notes
nevertheless say that the directional P1 hypotheses were informed by validation-fold pilots, and that some images in
the current held-out partition were used in earlier pilot training under a previous split. Those facts must be disclosed;
do not claim that the hypothesis origins were independent of all historical pilots or that the held-out images have never
been seen in any earlier project. The new held-out partition remains untouched by the new primary training and model
selection. Other limitations: direct-resize distortion; three training seeds; CIFAR-100 upsampling; PAD-UFES-20
diagnostic overlap and domain shift; F-EDL without spectral normalization; lesion IDs (not patient IDs) for HAM10000.
