"""Cluster bootstrap and the preregistered contrasts P1, P2 and P4.

Bootstrap samples are represented as COUNT matrices: counts[r, i] is how many times sample i appears
in replicate r when whole clusters (lesions / patients) are resampled with replacement. All matched
models and methods use the SAME counts, which keeps comparisons paired, and ECE / AUROC for thousands
of replicates are computed with matrix products instead of Python loops.

Training-seed variability is kept separate: every contrast is computed per seed (with its own bootstrap
interval); the three seed-specific estimates are then summarised by their mean and SD.
"""
import numpy as np

from edl.evaluation.metrics import bin_index


# ----------------------------------------------------------------------------- resampling
def cluster_bootstrap_counts(cluster_ids, n_rep: int, rng: np.random.Generator) -> np.ndarray:
    """counts [n_rep, n]: resample clusters with replacement; every member of a drawn cluster is included."""
    cluster_ids = np.asarray(cluster_ids)
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    c = len(uniq)
    draws = rng.integers(0, c, size=(n_rep, c))
    flat = (np.arange(n_rep)[:, None] * c + draws).ravel()
    cluster_counts = np.bincount(flat, minlength=n_rep * c).reshape(n_rep, c)
    return cluster_counts[:, inv].astype(np.float32)


def percentile_ci(samples, level: float):
    lo = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(samples, lo)), float(np.percentile(samples, 100.0 - lo))


# ----------------------------------------------------------------------------- weighted metrics
def ece_from_counts(conf, correct, counts, n_bins: int = 15, chunk: int = 2000) -> np.ndarray:
    """Top-label ECE for every bootstrap replicate. ECE = sum_b |sum(correct_b) - sum(conf_b)| / N."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(conf)
    idx = bin_index(conf, n_bins)
    member = np.zeros((n, n_bins))
    member[np.arange(n), idx] = 1.0
    conf_bin, corr_bin = member * conf[:, None], member * correct[:, None]
    out = []
    for s in range(0, counts.shape[0], chunk):
        c = np.asarray(counts[s:s + chunk], dtype=np.float64)
        out.append(np.abs(c @ corr_bin - c @ conf_bin).sum(axis=1) / c.sum(axis=1))
    return np.concatenate(out)


def auroc_from_counts(id_scores, ood_scores, id_counts, ood_counts, chunk: int = 1000) -> np.ndarray:
    """AUROC with OOD as the positive class for each replicate; ties count one half."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    n1, n2 = len(id_scores), len(ood_scores)
    scores = np.concatenate([id_scores, ood_scores])
    order = np.argsort(scores, kind="stable")
    s_sorted = scores[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    out = []
    for s in range(0, id_counts.shape[0], chunk):
        ci = np.asarray(id_counts[s:s + chunk], dtype=np.float64)
        co = np.asarray(ood_counts[s:s + chunk], dtype=np.float64)
        w_id = np.concatenate([ci, np.zeros_like(co)], axis=1)[:, order]
        w_ood = np.concatenate([np.zeros_like(ci), co], axis=1)[:, order]
        g_id = np.add.reduceat(w_id, starts, axis=1)
        g_ood = np.add.reduceat(w_ood, starts, axis=1)
        before = np.cumsum(g_id, axis=1) - g_id
        num = (g_ood * (before + 0.5 * g_id)).sum(axis=1)
        out.append(num / (ci.sum(axis=1) * co.sum(axis=1)))
    assert n1 + n2 == len(scores)
    return np.concatenate(out)


# ----------------------------------------------------------------------------- contrasts
def _top_label(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    return probs.max(1), (probs.argmax(1) == np.asarray(labels)).astype(np.float64)


def ece_with_point(probs, labels, counts, n_bins=15):
    """(point estimate on the full sample, bootstrap replicates)."""
    conf, correct = _top_label(probs, labels)
    point = ece_from_counts(conf, correct, np.ones((1, len(conf))), n_bins)[0]
    return float(point), ece_from_counts(conf, correct, counts, n_bins)


def p1_one_seed(preds: dict, counts, n_bins: int = 15) -> dict:
    """CutMix minus Base ECE for ONE framework and ONE seed, averaged over backbones.

    preds[(backbone, augmentation)] = (probs, labels) with augmentation in {"base_flips", "cutmix"}.
    """
    backbones = sorted({b for b, _ in preds})
    points, boots = [], []
    for b in backbones:
        pc, bc = ece_with_point(*preds[(b, "cutmix")], counts, n_bins)
        pb, bb = ece_with_point(*preds[(b, "base_flips")], counts, n_bins)
        points.append(pc - pb)
        boots.append(bc - bb)
    return {"point": float(np.mean(points)), "boot": np.mean(boots, axis=0), "per_backbone": dict(zip(backbones, points))}


def p2_one_seed(fedl_by_backbone: dict, softmax_by_backbone: dict, id_counts, ood_counts) -> dict:
    """AUROC(F-EDL native score) - AUROC(Softmax MSP uncertainty) on a near-OOD set, averaged over backbones.

    Each argument maps backbone -> (id_scores, ood_scores) (larger = more OOD).
    """
    backbones = sorted(fedl_by_backbone)
    points, boots = [], []
    for b in backbones:
        f_id, f_ood = fedl_by_backbone[b]
        s_id, s_ood = softmax_by_backbone[b]
        ones_i, ones_o = np.ones((1, len(f_id))), np.ones((1, len(f_ood)))
        pf = auroc_from_counts(f_id, f_ood, ones_i, ones_o)[0]
        ps = auroc_from_counts(s_id, s_ood, ones_i, ones_o)[0]
        points.append(pf - ps)
        boots.append(auroc_from_counts(f_id, f_ood, id_counts, ood_counts)
                     - auroc_from_counts(s_id, s_ood, id_counts, ood_counts))
    return {"point": float(np.mean(points)), "boot": np.mean(boots, axis=0), "per_backbone": dict(zip(backbones, points))}


def ece_degradation(clean, corrupted: list, counts, n_bins: int = 15):
    """Mean over the corrupted conditions of (ECE_corrupted - ECE_clean): (point, replicates).

    clean and every entry of corrupted are (probs, labels). Equal weights; clean is the reference only.
    """
    pc, bc = ece_with_point(*clean, counts, n_bins)
    pts, bts = [], []
    for probs, labels in corrupted:
        p, b = ece_with_point(probs, labels, counts, n_bins)
        pts.append(p - pc)
        bts.append(b - bc)
    return float(np.mean(pts)), np.mean(bts, axis=0)


def p4_one_seed(models: dict, counts, n_bins: int = 15) -> dict:
    """CutMix minus Base ECE degradation, averaged over the matched (backbone, framework) pairs.

    models[(backbone, framework, augmentation)] = (clean, [corrupted ...]) with 20 corrupted conditions.
    """
    pairs = sorted({(b, f) for b, f, _ in models})
    points, boots = [], []
    for b, f in pairs:
        pc, bc = ece_degradation(*models[(b, f, "cutmix")], counts, n_bins)
        pb, bb = ece_degradation(*models[(b, f, "base_flips")], counts, n_bins)
        points.append(pc - pb)
        boots.append(bc - bb)
    return {"point": float(np.mean(points)), "boot": np.mean(boots, axis=0), "per_pair": dict(zip(pairs, points))}


def summarize_seeds(per_seed: dict, level: float) -> dict:
    """per_seed[seed] = result dict with 'point' and 'boot'. Seed spread and bootstrap intervals stay separate."""
    seeds = sorted(per_seed)
    pts = np.array([per_seed[s]["point"] for s in seeds])
    out = {"seeds": seeds, "points": pts.tolist(), "mean": float(pts.mean()),
           "sd": float(pts.std(ddof=1)) if len(pts) > 1 else float("nan"), "level": level}
    for s in seeds:
        lo, hi = percentile_ci(per_seed[s]["boot"], level)
        out[f"ci_seed{s}"] = (lo, hi)
    return out
