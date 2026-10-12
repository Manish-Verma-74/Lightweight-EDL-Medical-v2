"""Preregistered contrasts P1, P2 and P4 from the saved prediction bundles (no retraining).

P1: CutMix - Base ECE, per framework (98.75% intervals, Bonferroni for four co-primary contrasts)
P2: AUROC(F-EDL native epistemic) - AUROC(Softmax MSP) on PAD-UFES-20, Base augmentation
P4: CutMix - Base mean ECE degradation over the 20 corrupted conditions (Softmax/F-EDL, two backbones)
Each contrast is computed per training seed with its own bootstrap interval; the three seed-specific
estimates are then summarised by mean and SD. Seed spread and bootstrap uncertainty are never merged.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edl.cli import REPO, add_common_args, expected_run_identity, load_protocol  # noqa: E402
from edl.config import config_hash, require_frozen, resolve  # noqa: E402
from edl.data.split import load_verified_manifest  # noqa: E402
from edl.evaluation.metrics import ece, macro_f1  # noqa: E402
from edl.training.checkpoint import check_identity  # noqa: E402
from edl.training.manifest import RunStatus  # noqa: E402
from edl.utils.env import git_commit  # noqa: E402
from edl.constants import BACKBONES, FRAMEWORKS, SEEDS  # noqa: E402
from edl.evaluation.predict import load_bundle  # noqa: E402
from edl.data.ood import PAD_COLUMNS, load_ood_manifest  # noqa: E402
from edl.evaluation.stats import cluster_bootstrap_counts, ece_from_counts, p1_one_seed, p2_one_seed, p4_one_seed, percentile_ci, summarize_seeds  # noqa: E402


def _flat(summary, name, extra):
    row = {"contrast": name, **extra, "mean_over_seeds": summary["mean"], "sd_over_seeds": summary["sd"],
           "level": summary["level"]}
    for s, p in zip(summary["seeds"], summary["points"]):
        row[f"point_seed{s}"] = p
        row[f"ci_lo_seed{s}"], row[f"ci_hi_seed{s}"] = summary[f"ci_seed{s}"]
    return row


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--phase", default="primary")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--replicates", type=int, default=None, help="override only for quick tests")
    args = ap.parse_args()
    if args.phase != "primary":
        raise SystemExit("P1-P4 statistical analyses are only defined for the frozen primary study")
    cfg, split_df, run_df, split_sha, commit = load_protocol(args)
    require_frozen(cfg)
    st = cfg["statistics"]
    n_rep = args.replicates or st["bootstrap_replicates"]
    root = os.path.join(args.drive_root, args.phase, "runs")
    out_dir = args.out_dir or os.path.join(args.drive_root, "results", "stats")
    os.makedirs(out_dir, exist_ok=True)
    test = split_df[split_df["split"] == "test"].reset_index(drop=True)

    def rid(b, f, a, s):
        return f"{b}__{f}__{a}__seed{s}"

    test_ids = test["image_id"].astype(str).tolist()
    pad_manifest = load_ood_manifest(resolve(cfg["ood"]["pad_ufes_20"]["manifest"], REPO),
                                     cfg["ood"]["pad_ufes_20"]["sha256"], PAD_COLUMNS)
    def bundle(b, f, a, s, name):
        run_id = rid(b, f, a, s)
        path = os.path.join(root, run_id, name)
        arrays, meta = load_bundle(path)
        expected = expected_run_identity(run_id, cfg, split_sha, commit, args.phase, False)
        check_identity(meta.get("identity", {}), expected, allow_commit_change=True)
        status = RunStatus(os.path.join(root, run_id)).read()
        check_identity(status.get("identity", {}), expected, allow_commit_change=True)
        eligible = status.get("state") == "completed" or (
            status.get("state") == "cap_limited" and status.get("cap_limited_at_resource_ceiling") is True)
        if not eligible:
            raise ValueError(f"{run_id}: statistical analysis requires an eligible primary run; state={status.get('state')}")
        if name in ("predictions_id.npz", "predictions_corruption.npz"):
            got_ids = arrays.get("test_image_ids", arrays.get("image_ids"))
            if got_ids is None or np.asarray(got_ids).astype(str).tolist() != test_ids:
                raise ValueError(f"{run_id}: {name} test image order does not match the frozen split manifest")
        elif name == "predictions_pad_ufes20.npz":
            if arrays.get("image_ids", np.array([])).astype(str).tolist() != pad_manifest["img_id"].astype(str).tolist():
                raise ValueError(f"{run_id}: PAD-UFES-20 image order differs from its hash-verified frozen manifest")
            if arrays.get("cluster_ids", np.array([])).astype(str).tolist() != pad_manifest["patient_id"].astype(str).tolist():
                raise ValueError(f"{run_id}: PAD-UFES-20 patient-cluster order differs from its frozen manifest")
        return arrays

    rows = []
    for fw in FRAMEWORKS:  # ---- P1
        per_seed = {}
        for s in SEEDS:
            rng = np.random.default_rng([st["bootstrap_rng_seed"], s])
            counts = cluster_bootstrap_counts(test["lesion_id"].values, n_rep, rng)
            preds = {}
            for b in BACKBONES:
                for a in ["base_flips", "cutmix"]:
                    z = bundle(b, fw, a, s, "predictions_id.npz")
                    preds[(b, a)] = (z["test_probs"], z["test_labels"])
            per_seed[s] = p1_one_seed(preds, counts, cfg["evaluation"]["ece_bins"])
        rows.append(_flat(summarize_seeds(per_seed, st["p1_interval_level"]), "P1_cutmix_minus_base_ece", {"framework": fw, "replicates": n_rep, "rng_seed": st["bootstrap_rng_seed"], "split_sha256": split_sha,
                                                                 "config_hash": config_hash(cfg)}))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "p1.csv"), index=False)

    per_seed = {}  # ---- P2
    for s in SEEDS:
        rng = np.random.default_rng([st["bootstrap_rng_seed"], s, 2])
        id_counts = cluster_bootstrap_counts(test["lesion_id"].values, n_rep, rng)
        fedl, soft, ood_counts = {}, {}, None
        for b in BACKBONES:
            zf, zs = bundle(b, "f_edl", "base_flips", s, "predictions_id.npz"), bundle(b, "softmax", "base_flips", s, "predictions_id.npz")
            pf, ps = bundle(b, "f_edl", "base_flips", s, "predictions_pad_ufes20.npz"), bundle(b, "softmax", "base_flips", s, "predictions_pad_ufes20.npz")
            fedl[b] = (zf["test_native_uncertainty"], pf["native_uncertainty"])
            soft[b] = (zs["test_msp_uncertainty"], ps["msp_uncertainty"])
            if ood_counts is None:
                ood_counts = cluster_bootstrap_counts(pf["cluster_ids"], n_rep, rng)
        per_seed[s] = p2_one_seed(fedl, soft, id_counts, ood_counts)
    pd.DataFrame([_flat(summarize_seeds(per_seed, st["default_interval_level"]), "P2_fedl_native_minus_softmax_msp_auroc_pad_ufes20", {"replicates": n_rep, "rng_seed": st["bootstrap_rng_seed"],
                                                                      "split_sha256": split_sha, "config_hash": config_hash(cfg)})]
                 ).to_csv(os.path.join(out_dir, "p2.csv"), index=False)

    per_seed = {}  # ---- P4
    sub = cfg["corruption"]["subset"]
    for s in SEEDS:
        rng = np.random.default_rng([st["bootstrap_rng_seed"], s, 4])
        counts = cluster_bootstrap_counts(test["lesion_id"].values, n_rep, rng)
        models = {}
        for b in sub["backbones"]:
            for fw in sub["frameworks"]:
                for a in sub["augmentations"]:
                    z = bundle(b, fw, a, s, "predictions_corruption.npz")
                    names = list(z["conditions"])
                    clean = (z["probs"][names.index("clean:0")], z["labels"])
                    corrupted = [(z["probs"][i], z["labels"]) for i, n in enumerate(names) if n != "clean:0"]
                    if len(corrupted) != 20:
                        raise SystemExit(f"{rid(b, fw, a, s)}: expected 20 corrupted conditions, found {len(corrupted)}")
                    models[(b, fw, a)] = (clean, corrupted)
        per_seed[s] = p4_one_seed(models, counts, cfg["evaluation"]["ece_bins"])
    pd.DataFrame([_flat(summarize_seeds(per_seed, st["default_interval_level"]), "P4_cutmix_minus_base_ece_degradation", {"replicates": n_rep, "rng_seed": st["bootstrap_rng_seed"],
                                                              "split_sha256": split_sha, "config_hash": config_hash(cfg)})]
                 ).to_csv(os.path.join(out_dir, "p4.csv"), index=False)
    # Secondary per-corruption curves for the 24 preselected model/seed bundles.
    curve_rows = []
    curve_rep = int(st.get("curve_bootstrap_replicates", 2000))
    sub_cfg = cfg["corruption"]["subset"]
    for seed in SEEDS:
        rng = np.random.default_rng([st["bootstrap_rng_seed"], seed, 405])
        counts = cluster_bootstrap_counts(test["lesion_id"].values, curve_rep, rng)
        for backbone in sub_cfg["backbones"]:
            for framework in sub_cfg["frameworks"]:
                for aug in sub_cfg["augmentations"]:
                    run_id = rid(backbone, framework, aug, seed)
                    z = bundle(backbone, framework, aug, seed, "predictions_corruption.npz")
                    labels = z["labels"].astype(int)
                    conditions = list(z["conditions"].astype(str))
                    clean_idx = conditions.index("clean:0")
                    clean_probs = z["probs"][clean_idx]
                    clean_ece = ece(clean_probs, labels, cfg["evaluation"]["ece_bins"])
                    clean_f1 = macro_f1(labels, clean_probs.argmax(1), 7)
                    clean_conf = clean_probs.max(1)
                    clean_correct = (clean_probs.argmax(1) == labels).astype(float)
                    clean_boot = ece_from_counts(clean_conf, clean_correct, counts, cfg["evaluation"]["ece_bins"])
                    for idx, condition in enumerate(conditions):
                        probs = z["probs"][idx]
                        current_ece = ece(probs, labels, cfg["evaluation"]["ece_bins"])
                        current_f1 = macro_f1(labels, probs.argmax(1), 7)
                        errors = (probs.argmax(1) != labels).astype(float)
                        conf, correct = probs.max(1), 1.0 - errors
                        b_ece = ece_from_counts(conf, correct, counts, cfg["evaluation"]["ece_bins"])
                        if idx == clean_idx:
                            delta_boot = np.zeros_like(b_ece)
                        else:
                            delta_boot = b_ece - clean_boot
                        lo, hi = percentile_ci(delta_boot, st["default_interval_level"])
                        curve_rows.append({"run_id": run_id, "backbone": backbone, "framework": framework,
                                           "augmentation": aug, "seed": seed, "condition": condition,
                                           "ece": current_ece, "ece_clean": clean_ece,
                                           "ece_delta": current_ece-clean_ece,
                                           "ece_delta_ci_low": lo, "ece_delta_ci_high": hi,
                                           "macro_f1": current_f1, "macro_f1_clean": clean_f1,
                                           "macro_f1_delta": current_f1-clean_f1,
                                           "curve_bootstrap_replicates": curve_rep,
                                           "rng_seed": st["bootstrap_rng_seed"], "split_sha256": split_sha,
                                           "config_hash": config_hash(cfg)})
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, "corruption_curves.csv"), index=False)
    print("wrote p1.csv, p2.csv, p4.csv and corruption_curves.csv to", out_dir)


if __name__ == "__main__":
    main()
