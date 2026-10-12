"""All metric definitions, in one place. Pure NumPy (plus scikit-learn for ROC/PR curves).

Conventions (frozen in the protocol):
- ECE: top-label, 15 equal-width bins on [0, 1]; bin i is (lo, hi], the first bin is [0, hi];
  empty bins are skipped; each bin is weighted by its fraction of samples.
- Macro-cwECE: one-vs-rest ECE per class using that class's probability, then the unweighted mean.
- Brier: mean over samples of the sum over classes of squared errors (not divided by K).
- Risk-coverage: samples sorted by increasing uncertainty (stable order on ties); the risk at
  coverage k/n is the error rate among the k most certain samples; AURC is the mean of those risks.
- OOD: OOD is the positive class and a larger score means more likely OOD. FPR95 is the false
  positive rate at the first ROC point whose OOD recall is at least 0.95.
"""
import numpy as np


def _check_probs(probs, labels=None):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must have shape (n, K)")
    if not np.isfinite(probs).all():
        raise ValueError("probs contain non-finite values")
    if probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError("probs must contain at least one sample and one class")
    if (probs < -1e-7).any() or (probs > 1.0 + 1e-7).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("each probability row must sum to 1")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (probs.shape[0],):
            raise ValueError("labels must have shape (n,)")
        if labels.min() < 0 or labels.max() >= probs.shape[1]:
            raise ValueError("labels out of range")
    return probs, labels


def accuracy(probs, labels) -> float:
    probs, labels = _check_probs(probs, labels)
    return float((probs.argmax(1) == labels).mean())


def per_class_recall(probs, labels, num_classes=None) -> np.ndarray:
    """Recall (sensitivity) per class; NaN for classes with no support."""
    probs, labels = _check_probs(probs, labels)
    k = num_classes or probs.shape[1]
    pred = probs.argmax(1)
    out = np.full(k, np.nan)
    for c in range(k):
        m = labels == c
        if m.any():
            out[c] = float((pred[m] == c).mean())
    return out


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """Unweighted mean of per-class F1. A class with no support and no predictions scores 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(num_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def bin_edges(n_bins: int = 15) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def bin_index(conf: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """Bin of each confidence: bin i covers (edge_i, edge_{i+1}], the first bin also contains 0."""
    conf = np.asarray(conf, dtype=np.float64)
    # side="left": a value equal to an edge belongs to the bin that ends at that edge.
    idx = np.searchsorted(bin_edges(n_bins), conf, side="left") - 1
    return np.clip(idx, 0, n_bins - 1)


def _ece_from_conf(conf, correct, n_bins=15) -> float:
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(conf)
    if n == 0:
        raise ValueError("empty input")
    idx = bin_index(conf, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():  # empty bins are skipped
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ece(probs, labels, n_bins: int = 15) -> float:
    probs, labels = _check_probs(probs, labels)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    return _ece_from_conf(conf, correct, n_bins)


def class_ece(probs, labels, n_bins: int = 15) -> np.ndarray:
    """One-vs-rest ECE for every class (confidence = that class's probability)."""
    probs, labels = _check_probs(probs, labels)
    return np.array([_ece_from_conf(probs[:, k], (labels == k).astype(np.float64), n_bins)
                     for k in range(probs.shape[1])])


def macro_cwece(probs, labels, n_bins: int = 15) -> float:
    return float(class_ece(probs, labels, n_bins).mean())


def brier(probs, labels) -> float:
    probs, labels = _check_probs(probs, labels)
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def risk_coverage(errors, uncertainty):
    """Return (coverage, risk) arrays for k = 1..n most-certain samples."""
    errors = np.asarray(errors, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if errors.shape != uncertainty.shape or errors.ndim != 1 or len(errors) == 0:
        raise ValueError("errors and uncertainty must be non-empty 1-D arrays of equal length")
    if not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty contains non-finite values")
    order = np.argsort(uncertainty, kind="stable")
    cum = np.cumsum(errors[order])
    k = np.arange(1, len(errors) + 1)
    return k / len(errors), cum / k


def aurc(errors, uncertainty) -> float:
    """Area under the risk-coverage curve = mean risk over k = 1..n. Lower is better."""
    _, risk = risk_coverage(errors, uncertainty)
    return float(risk.mean())


def ood_metrics(id_scores, ood_scores, recall_target: float = 0.95) -> dict:
    """AUROC, FPR95 and AUPR with OOD as the positive class (larger score = more OOD)."""
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    if len(id_scores) == 0 or len(ood_scores) == 0:
        raise ValueError("need both ID and OOD scores")
    if not (np.isfinite(id_scores).all() and np.isfinite(ood_scores).all()):
        raise ValueError("scores contain non-finite values")
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    first = int(np.argmax(tpr >= recall_target))
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "fpr95": float(fpr[first]),
        "ood_prevalence": float(len(ood_scores) / len(s)),
        "n_id": int(len(id_scores)),
        "n_ood": int(len(ood_scores)),
    }


def classification_summary(probs, labels, n_bins: int = 15) -> dict:
    probs, labels = _check_probs(probs, labels)
    k = probs.shape[1]
    return {
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(labels, probs.argmax(1), k),
        "ece": ece(probs, labels, n_bins),
        "macro_cwece": macro_cwece(probs, labels, n_bins),
        "brier": brier(probs, labels),
        "per_class_recall": per_class_recall(probs, labels, k).tolist(),
        "support": np.bincount(labels, minlength=k).tolist(),
    }
