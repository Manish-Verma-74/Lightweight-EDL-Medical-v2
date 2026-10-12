import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edl.augment import apply_cutmix, apply_mixup, cutmix_box_and_weight  # noqa: E402


def test_mixup_uses_the_same_lambda_for_images_and_targets():
    x = torch.randn(6, 3, 16, 16)
    onehot = torch.eye(7)[torch.tensor([0, 1, 2, 3, 4, 5])]
    mixed, t = apply_mixup(x, onehot, 0.2, np.random.default_rng(3))
    ref = np.random.default_rng(3)
    lam, perm = float(ref.beta(0.2, 0.2)), torch.from_numpy(ref.permutation(6))
    assert torch.allclose(mixed, lam * x + (1 - lam) * x[perm], atol=1e-6)
    assert torch.allclose(t, lam * onehot + (1 - lam) * onehot[perm], atol=1e-6)
    assert torch.allclose(t.sum(1), torch.ones(6), atol=1e-6)


def test_cutmix_target_weight_matches_the_pasted_area():
    x = torch.randn(6, 3, 32, 32)
    onehot = torch.eye(7)[torch.tensor([0, 1, 2, 3, 4, 5])]
    mixed, t = apply_cutmix(x, onehot, 1.0, np.random.default_rng(11))
    ref = np.random.default_rng(11)
    lam, perm = float(ref.beta(1.0, 1.0)), ref.permutation(6)
    cx, cy = int(ref.integers(0, 32)), int(ref.integers(0, 32))
    (x1, y1, x2, y2), lam_adj = cutmix_box_and_weight(32, 32, lam, cx, cy)
    inside = mixed[:, :, y1:y2, x1:x2]
    assert torch.equal(inside, x[torch.from_numpy(perm)][:, :, y1:y2, x1:x2])
    keep = torch.ones(32, 32, dtype=torch.bool)
    keep[y1:y2, x1:x2] = False
    assert torch.equal(mixed[:, :, keep], x[:, :, keep])
    assert torch.allclose(t, lam_adj * onehot + (1 - lam_adj) * onehot[torch.from_numpy(perm)], atol=1e-6)
    assert torch.allclose(t.sum(1), torch.ones(6), atol=1e-6)
    assert not torch.equal(mixed, x) or lam_adj == 1.0
