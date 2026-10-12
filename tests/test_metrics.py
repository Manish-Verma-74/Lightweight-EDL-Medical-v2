import numpy as np
import pytest

from edl.evaluation import metrics as M

P = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
Y = np.array([0, 1, 1, 2])


def test_accuracy_brier_ece_hand_calculated():
    assert M.accuracy(P, Y) == 0.75
    assert M.brier(P, Y) == pytest.approx(0.34)            # (0.06 + 0.86 + 0.38 + 0.06) / 4
    assert M.ece(P, Y, n_bins=10) == pytest.approx(0.375)  # 0.5*0.2 + 0.25*0.6 + 0.25*0.5


def test_bin_boundaries_are_right_inclusive():
    e = M.bin_edges(15)
    idx = M.bin_index(np.array([0.0, e[3], e[3] + 1e-9, 1.0]), 15)
    assert idx.tolist() == [0, 2, 3, 14]


def test_perfect_and_worst_calibration():
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert M.ece(probs, np.array([0, 1, 2, 0])) == 0.0
    assert M.ece(probs, np.array([1, 2, 0, 1])) == pytest.approx(1.0)


def test_cwece_is_mean_of_class_eces():
    ce = M.class_ece(P, Y, 10)
    assert M.macro_cwece(P, Y, 10) == pytest.approx(ce.mean())
    assert len(ce) == 3


def test_macro_f1_and_recall():
    pred = np.array([0, 0, 1, 2])
    assert M.macro_f1(Y, pred, 3) == pytest.approx(7 / 9)    # F1 = 2/3, 2/3, 1
    rec = M.per_class_recall(P, Y, 3)
    assert rec[0] == 1.0 and rec[1] == 0.5 and rec[2] == 1.0
    assert M.macro_f1(np.array([0, 1]), np.array([0, 1]), 3) == pytest.approx(2 / 3)  # class 2 absent -> F1 0


def test_risk_coverage_and_aurc():
    errors = np.array([0, 1, 0, 1])
    unc = np.array([0.1, 0.4, 0.2, 0.9])
    cov, risk = M.risk_coverage(errors, unc)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])   # order: 0.1,0.2,0.4,0.9 -> errors 0,0,1,1
    assert M.aurc(errors, unc) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert M.aurc(errors, -unc) > M.aurc(errors, unc)                 # a bad ordering has higher AURC


def test_ties_use_a_stable_order():
    c1, r1 = M.risk_coverage(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5]))
    c2, r2 = M.risk_coverage(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5]))
    assert np.array_equal(r1, r2)


def test_ood_metrics_orientation_and_fpr95():
    id_s = np.array([0.1, 0.2, 0.3, 0.4])
    ood_s = np.array([0.5, 0.6, 0.7, 0.8])
    m = M.ood_metrics(id_s, ood_s)
    assert m["auroc"] == 1.0 and m["fpr95"] == 0.0 and m["ood_prevalence"] == 0.5
    flipped = M.ood_metrics(ood_s, id_s)          # larger-score-is-OOD convention: reversed gives AUROC 0
    assert flipped["auroc"] == 0.0
    # one OOD below all ID scores: 95% recall needs every OOD, so FPR95 = 1.0
    m2 = M.ood_metrics(np.array([0.2, 0.3, 0.4, 0.5]), np.array([0.1, 0.6, 0.7, 0.8]))
    assert m2["fpr95"] == 1.0


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        M.ece(np.array([[0.5, np.nan]]), np.array([0]))
    with pytest.raises(ValueError):
        M.risk_coverage(np.array([1.0]), np.array([np.inf]))
    with pytest.raises(ValueError):
        M.ood_metrics([], [1.0])
