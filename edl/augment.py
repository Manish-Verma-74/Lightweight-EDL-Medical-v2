"""Batch-level Mixup and CutMix with soft targets. Base and RandAugment are image transforms.

The random numbers come from a NumPy Generator owned by the training engine, so its state can be
saved in the resumable checkpoint.
"""
import numpy as np


def cutmix_box_and_weight(h: int, w: int, lam: float, cx: int, cy: int):
    """Box (x1, y1, x2, y2) clipped to the image, and the label weight of the ORIGINAL image
    computed from the actual clipped area: lam_adj = 1 - area / (h * w)."""
    ratio = float(np.sqrt(1.0 - lam))
    cut_w, cut_h = int(w * ratio), int(h * ratio)
    x1, x2 = int(np.clip(cx - cut_w // 2, 0, w)), int(np.clip(cx + cut_w // 2, 0, w))
    y1, y2 = int(np.clip(cy - cut_h // 2, 0, h)), int(np.clip(cy + cut_h // 2, 0, h))
    area = (x2 - x1) * (y2 - y1)
    return (x1, y1, x2, y2), 1.0 - area / float(w * h)


def mixup_weight(alpha: float, rng: np.random.Generator) -> float:
    return float(rng.beta(alpha, alpha))


def apply_mixup(images, onehot, alpha: float, rng: np.random.Generator):
    """images [B,C,H,W], onehot [B,K] -> mixed images and soft targets (same lambda for both)."""
    import torch
    onehot = onehot.to(device=images.device, dtype=torch.float32)
    lam = mixup_weight(alpha, rng)
    perm = torch.from_numpy(rng.permutation(images.shape[0])).to(images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    targets = lam * onehot + (1.0 - lam) * onehot[perm]
    return mixed, targets


def apply_cutmix(images, onehot, alpha: float, rng: np.random.Generator):
    """Paste a box from a shuffled copy of the batch; the target weight follows the clipped area."""
    import torch
    onehot = onehot.to(device=images.device, dtype=torch.float32)
    b, _, h, w = images.shape
    lam = float(rng.beta(alpha, alpha))
    perm_np = rng.permutation(b)
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    (x1, y1, x2, y2), lam_adj = cutmix_box_and_weight(h, w, lam, cx, cy)
    perm = torch.from_numpy(perm_np).to(images.device)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    targets = lam_adj * onehot + (1.0 - lam_adj) * onehot[perm]
    return mixed, targets
