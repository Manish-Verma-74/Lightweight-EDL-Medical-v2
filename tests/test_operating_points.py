import numpy as np
import pytest

from edl.evaluation import operating_points as O


def test_threshold_for_specificity_uses_validation_only_and_meets_target():
    val_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    val_pos = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    t = O.threshold_for_specificity(val_scores, val_pos, 0.9)
    neg = val_scores[~val_pos]
    assert (neg < t).mean() >= 0.9
    res = O.apply_sensitivity(np.array([0.05, 0.85, 0.9]), np.array([0, 1, 1], dtype=bool), t)
    assert res["sensitivity"] == 1.0 and res["specificity"] == 1.0


def test_coverage_threshold_and_selective_risk():
    val_u = np.linspace(0, 1, 101)
    t = O.threshold_for_coverage(val_u, 0.8)
    assert t == pytest.approx(0.8)
    test_u = np.array([0.1, 0.5, 0.9, 0.95])
    res = O.apply_coverage(test_u, np.array([0, 0, 1, 1]), np.array([1, 0, 1, 1], dtype=bool),
                           np.array([1, 0, 0, 1], dtype=bool), t)
    assert res["coverage"] == 0.5 and res["selective_risk"] == 0.0 and res["sensitivity"] == 1.0


def test_coverage_specificity_is_calculated_only_on_retained_negative_samples():
    # The two low-uncertainty samples are retained. The retained negative is
    # correctly classified; two rejected negatives are misclassified. Selective
    # specificity must therefore be 1.0, not 1/3 computed over all negatives.
    test_u = np.array([0.1, 0.2, 0.8, 0.9])
    is_pos = np.array([1, 0, 0, 0], dtype=bool)
    pred_pos = np.array([1, 0, 1, 1], dtype=bool)
    errors = np.array([0, 0, 1, 1], dtype=float)
    res = O.apply_coverage(test_u, errors, is_pos, pred_pos, threshold=0.5)
    assert res["coverage"] == 0.5
    assert res["sensitivity"] == 1.0
    assert res["achieved_specificity"] == 1.0
