import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score  # noqa: E402

from edl.evaluation import metrics as M  # noqa: E402
from edl.evaluation import stats as S  # noqa: E402


def make_probs(rng, n=200, k=7, sharp=0.5):
    p = rng.dirichlet(np.ones(k) * sharp, size=n)
    return p, rng.integers(0, k, size=n)


def test_weighted_ece_matches_plain_ece_and_repeated_samples():
    rng = np.random.default_rng(0)
    p, y = make_probs(rng)
    point, boots = S.ece_with_point(p, y, np.ones((3, len(y))))
    assert point == pytest.approx(M.ece(p, y)) and np.allclose(boots, point)
    counts = rng.integers(0, 3, size=(1, len(y))).astype(float)
    rep = np.repeat(np.arange(len(y)), counts[0].astype(int))
    got = S.ece_from_counts(p.max(1), (p.argmax(1) == y).astype(float), counts)[0]
    assert got == pytest.approx(M.ece(p[rep], y[rep]))


def test_weighted_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    ids, oods = rng.integers(0, 10, 120).astype(float), rng.integers(3, 14, 90).astype(float)
    ci, co = rng.integers(0, 3, (1, 120)).astype(float), rng.integers(0, 3, (1, 90)).astype(float)
    got = S.auroc_from_counts(ids, oods, ci, co)[0]
    ri, ro = np.repeat(ids, ci[0].astype(int)), np.repeat(oods, co[0].astype(int))
    want = roc_auc_score(np.r_[np.zeros(len(ri)), np.ones(len(ro))], np.r_[ri, ro])
    assert got == pytest.approx(want)
    full = S.auroc_from_counts(ids, oods, np.ones((1, 120)), np.ones((1, 90)))[0]
    assert full == pytest.approx(M.ood_metrics(ids, oods)["auroc"])


def test_cluster_bootstrap_keeps_clusters_together_and_is_reproducible():
    clusters = np.repeat(np.arange(50), 4)
    a = S.cluster_bootstrap_counts(clusters, 20, np.random.default_rng(5))
    b = S.cluster_bootstrap_counts(clusters, 20, np.random.default_rng(5))
    assert np.array_equal(a, b) and a.shape == (20, 200)
    for i in range(50):
        assert (a[:, i * 4:(i + 1) * 4].std(axis=1) == 0).all()     # all members of a cluster share one count
    assert (a.sum(axis=1) == 200).all()                              # equal cluster sizes -> constant total


def test_p1_sign_when_cutmix_is_better_calibrated():
    rng = np.random.default_rng(2)
    n, k = 400, 7
    y = rng.integers(0, k, n)
    sharp = np.eye(k)[y] * 0.5 + 0.5 / k                      # always correct, under-confident
    overconf = np.full((n, k), 0.01)
    overconf[np.arange(n), rng.integers(0, k, n)] = 0.94      # random class, over-confident
    overconf /= overconf.sum(1, keepdims=True)
    preds = {("b1", "cutmix"): (sharp, y), ("b1", "base_flips"): (overconf, y)}
    counts = S.cluster_bootstrap_counts(np.arange(n), 200, rng)
    res = S.p1_one_seed(preds, counts)
    base, cut = M.ece(overconf, y), M.ece(sharp, y)
    assert res["point"] == pytest.approx(cut - base)
    assert res["point"] < 0 and (res["boot"] < 0).mean() > 0.95


def test_p2_and_p4_shapes_and_signs():
    rng = np.random.default_rng(3)
    id_s, ood_good, ood_bad = rng.random(100), rng.random(80) + 0.5, rng.random(80)
    ci, co = S.cluster_bootstrap_counts(np.arange(100), 50, rng), S.cluster_bootstrap_counts(np.arange(80), 50, rng)
    res = S.p2_one_seed({"b": (id_s, ood_good)}, {"b": (id_s, ood_bad)}, ci, co)
    assert res["point"] > 0 and res["boot"].shape == (50,)

    n, k = 120, 7
    y = rng.integers(0, k, n)
    good = np.eye(k)[y] * 0.6 + 0.4 / k

    def noisy(strength):
        q = good + strength * rng.random((n, k))
        return q / q.sum(1, keepdims=True), y

    models = {("b", "softmax", "cutmix"): (noisy(0.0), [noisy(0.05) for _ in range(20)]),
              ("b", "softmax", "base_flips"): (noisy(0.0), [noisy(0.9) for _ in range(20)])}
    counts = S.cluster_bootstrap_counts(np.arange(n), 40, rng)
    r4 = S.p4_one_seed(models, counts)
    assert np.isfinite(r4["point"]) and r4["boot"].shape == (40,)


def test_summarize_seeds_keeps_seed_spread_separate_from_bootstrap():
    boots = {s: {"point": 0.1 * (i + 1), "boot": np.random.default_rng(i).normal(0.1 * (i + 1), 0.01, 500)}
             for i, s in enumerate([42, 123, 2024])}
    out = S.summarize_seeds(boots, 0.95)
    assert out["mean"] == pytest.approx(0.2) and out["sd"] == pytest.approx(0.1)
    assert out["ci_seed42"][0] < 0.1 < out["ci_seed42"][1]
    assert S.percentile_ci(np.arange(1001), 0.95) == pytest.approx((25.0, 975.0))
