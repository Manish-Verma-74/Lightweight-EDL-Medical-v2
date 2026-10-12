"""Melanoma operating points. Thresholds come from VALIDATION predictions only and are then applied
unchanged to the test predictions."""
import numpy as np


def threshold_for_specificity(val_scores, val_is_pos, target_spec: float) -> float:
    """Smallest score threshold t (predict positive when score >= t) whose validation specificity >= target."""
    scores = np.asarray(val_scores, dtype=np.float64)
    pos = np.asarray(val_is_pos, dtype=bool)
    neg_scores = np.sort(scores[~pos])
    if len(neg_scores) == 0:
        raise ValueError("no negative samples")
    # specificity(t) = fraction of negatives with score < t
    cands = np.unique(np.concatenate([scores, [scores.max() + 1.0]]))
    for t in cands:  # ascending: the first feasible t maximises sensitivity
        spec = np.searchsorted(neg_scores, t, side="left") / len(neg_scores)
        if spec >= target_spec:
            return float(t)
    return float(cands[-1])


def apply_sensitivity(test_scores, test_is_pos, threshold: float) -> dict:
    scores = np.asarray(test_scores, dtype=np.float64)
    pos = np.asarray(test_is_pos, dtype=bool)
    pred = scores >= threshold
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((~pred[~pos]).mean()) if (~pos).any() else float("nan")
    return {"threshold": float(threshold), "sensitivity": sens, "specificity": spec}


def threshold_for_coverage(val_uncertainty, target_coverage: float) -> float:
    """Reject samples with uncertainty above this value to keep about `target_coverage` of validation data."""
    return float(np.quantile(np.asarray(val_uncertainty, dtype=np.float64), target_coverage))


def apply_coverage(test_uncertainty, test_errors, test_is_pos, test_pred_pos, threshold: float) -> dict:
    """Selective metrics, including sensitivity/specificity, among retained test samples."""
    u = np.asarray(test_uncertainty, dtype=np.float64)
    err = np.asarray(test_errors, dtype=np.float64)
    pos = np.asarray(test_is_pos, dtype=bool)
    pred = np.asarray(test_pred_pos, dtype=bool)
    if not (u.ndim == err.ndim == pos.ndim == pred.ndim == 1):
        raise ValueError("coverage inputs must be one-dimensional")
    if not (len(u) == len(err) == len(pos) == len(pred)):
        raise ValueError("coverage inputs must have equal lengths")
    if len(u) == 0 or not np.isfinite(u).all() or not np.isfinite(err).all():
        raise ValueError("coverage inputs must be non-empty and finite")
    keep = u <= threshold
    kept_pos = keep & pos
    kept_neg = keep & (~pos)
    return {"threshold": float(threshold), "coverage": float(keep.mean()),
            "selective_risk": float(err[keep].mean()) if keep.any() else float("nan"),
            "sensitivity": float(pred[kept_pos].mean()) if kept_pos.any() else float("nan"),
            "achieved_specificity": float((~pred[kept_neg]).mean()) if kept_neg.any() else float("nan")}
