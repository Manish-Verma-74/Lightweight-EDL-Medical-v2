import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.special import digamma, gammaln  # noqa: E402

from edl.frameworks import EDLFramework, FEDLFramework, REDLFramework, SoftmaxFramework  # noqa: E402
from edl.losses.evidential import edl_loss, redl_loss, softmax_cross_entropy, to_target_matrix  # noqa: E402
from edl.losses.fedl import compute_moments  # noqa: E402

K = 7


def kl_ref(alpha):
    s = alpha.sum()
    return gammaln(s) - gammaln(alpha).sum() - gammaln(K) + ((alpha - 1) * (digamma(alpha) - digamma(s))).sum()


def edl_ref(output, t, epoch, step=10):
    alpha = np.maximum(output, 0) + 1.0
    s = alpha.sum()
    a = ((t - alpha / s) ** 2).sum()
    b = (alpha * (s - alpha) / (s * s * (s + 1))).sum()
    return a + b + min(1.0, epoch / step) * kl_ref(alpha * (1 - t) + t)


def redl_ref(output, t, epoch, lam=0.1, step=10):
    alpha = np.maximum(output, 0) + lam
    s = alpha.sum()
    return ((t - alpha / s) ** 2).sum() + min(1.0, epoch / step) * kl_ref(alpha * (1 - t) + t)


OUT = np.array([1.0, 0.0, 2.0, 0.0, -1.0, 0.0, 3.0])
HARD = np.eye(K)[0]
SOFT = np.array([0, 0.7, 0.3, 0, 0, 0, 0.0])


def tt(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)


def test_softmax_cross_entropy_hard_and_soft():
    logits = torch.zeros(2, K)
    assert softmax_cross_entropy(logits, torch.tensor([3, 5])).item() == pytest.approx(np.log(K), rel=1e-6)
    soft = torch.tensor(np.eye(K)[[3, 5]], dtype=torch.float32)
    assert softmax_cross_entropy(logits, soft).item() == pytest.approx(np.log(K), rel=1e-6)


def test_edl_zero_evidence_is_the_known_constant_for_every_epoch():
    expected = 42 / 49 + 42 / 392       # squared error + expected variance; KL is zero at alpha_tilde = 1
    for epoch in (1, 5, 10, 30):
        assert edl_loss(torch.zeros(1, K), torch.tensor([2]), epoch, 10).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("target", [HARD, SOFT])
@pytest.mark.parametrize("epoch", [1, 3, 10, 25])
def test_edl_and_redl_match_independent_numpy_references(target, epoch):
    assert edl_loss(tt(OUT), tt(target), epoch, 10).item() == pytest.approx(edl_ref(OUT, target, epoch), rel=1e-4, abs=1e-5)
    assert redl_loss(tt(OUT), tt(target), epoch, 10, 0.1).item() == pytest.approx(redl_ref(OUT, target, epoch), rel=1e-4, abs=1e-5)


def test_kl_weight_follows_the_one_based_schedule():
    _, p1 = edl_loss(tt(OUT), tt(HARD), 1, 10, return_parts=True)
    _, p10 = edl_loss(tt(OUT), tt(HARD), 10, 10, return_parts=True)
    _, p20 = redl_loss(tt(OUT), tt(HARD), 20, 10, 0.1, return_parts=True)
    assert p1["coef"] == pytest.approx(0.1) and p10["coef"] == 1.0 and p20["coef"] == 1.0


def test_integer_labels_equal_one_hot_soft_targets_for_every_loss():
    out = torch.randn(4, K)
    labels = torch.tensor([0, 2, 4, 6])
    onehot = torch.eye(K)[labels]
    assert softmax_cross_entropy(out, labels).item() == pytest.approx(softmax_cross_entropy(out, onehot).item())
    assert edl_loss(out, labels, 4, 10).item() == pytest.approx(edl_loss(out, onehot, 4, 10).item())
    assert redl_loss(out, labels, 4, 10, 0.1).item() == pytest.approx(redl_loss(out, onehot, 4, 10, 0.1).item())


def test_soft_targets_are_validated_not_silently_fixed():
    with pytest.raises(ValueError):
        to_target_matrix(torch.full((2, K), 0.2), K)          # rows sum to 1.4
    with pytest.raises(ValueError):
        to_target_matrix(torch.full((2, K), -0.1), K)
    with pytest.raises(ValueError):
        to_target_matrix(torch.zeros(2, K + 1), K)


def test_softmax_native_uncertainty_and_probabilities():
    fw = SoftmaxFramework()
    p = fw.predict({"logits": torch.tensor([[2.0, 0, 0, 0, 0, 0, 0], [0.0] * 7])})
    assert p["probs"].sum(1).tolist() == pytest.approx([1.0, 1.0])
    assert p["native_uncertainty"][1].item() == pytest.approx(1 - 1 / K)
    assert p["native_uncertainty"][0] < p["native_uncertainty"][1]
    assert p["entropy"][1].item() == pytest.approx(np.log(K), rel=1e-5)
    assert torch.allclose(p["msp_uncertainty"], p["native_uncertainty"])


def test_edl_and_redl_native_uncertainty_are_vacuity_and_point_the_right_way():
    zeros, strong = torch.zeros(1, K), torch.full((1, K), 50.0)
    edl = EDLFramework(10).predict({"logits": torch.cat([zeros, strong])})
    assert edl["native_uncertainty"][0].item() == pytest.approx(1.0)               # K / S with S = K
    assert edl["native_uncertainty"][0] > edl["native_uncertainty"][1]
    redl = REDLFramework(0.1, 10).predict({"logits": torch.cat([zeros, strong])})
    assert redl["native_uncertainty"][0].item() == pytest.approx(1.0, rel=1e-5)    # lambda*K/S, S = lambda*K
    assert redl["native_uncertainty"][0] > redl["native_uncertainty"][1]
    manual = 0.1 * K / (50.0 * K + 0.1 * K)
    assert redl["native_uncertainty"][1].item() == pytest.approx(manual, rel=1e-5)
    assert edl["probs"].sum(1).tolist() == pytest.approx([1.0, 1.0])


def test_fedl_moments_and_scores_hand_calculated():
    alpha, p, tau = torch.ones(1, K), torch.full((1, K), 1 / K), torch.ones(1, 1)
    mu, var = compute_moments(alpha, p, tau)
    assert mu.sum().item() == pytest.approx(1.0) and mu[0, 0].item() == pytest.approx(1 / K)
    assert var.sum().item() == pytest.approx(3 / 28, rel=1e-5)
    pred = FEDLFramework().predict({"alpha": alpha, "p": p, "tau": tau})
    assert pred["native_uncertainty"].item() == pytest.approx(3 / 28, rel=1e-5)
    assert pred["total_uncertainty"].item() == pytest.approx(1 - K / K ** 2, rel=1e-5)
    assert (var >= 0).all()


def test_fedl_loss_is_finite_for_hard_and_soft_targets():
    fw = FEDLFramework()
    out = {"alpha": torch.rand(3, K) + 0.5, "p": torch.softmax(torch.randn(3, K), 1), "tau": torch.rand(3, 1) + 0.5}
    assert torch.isfinite(fw.loss(out, torch.tensor([0, 1, 2]), 1))
    assert torch.isfinite(fw.loss(out, torch.softmax(torch.randn(3, K), 1), 1))
