"""Build final_factorial_v1.csv, include ID/OOD/efficiency outputs, and validate worker-log consistency."""
import argparse
import glob
import json
import os
import sys

import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from edl.cli import add_common_args, expected_run_identity, load_protocol  # noqa: E402
from edl.config import config_hash, require_frozen  # noqa: E402
from edl.training.checkpoint import check_identity  # noqa: E402
from edl.utils.io import read_json  # noqa: E402


def flatten_operating_points(row, metrics):
    op = metrics.get("melanoma_operating_points", {})
    for key, value in op.get("probability", {}).items():
        prefix = f"melanoma_{key}"
        for metric in ("threshold", "sensitivity", "specificity"):
            row[f"{prefix}_{metric}"] = value.get(metric)
    for score, levels in op.get("selective", {}).items():
        for key, value in levels.items():
            prefix = f"selective_{score}_{key}"
            for metric in ("threshold", "coverage", "selective_risk", "sensitivity", "achieved_specificity"):
                row[f"{prefix}_{metric}"] = value.get(metric)


def aggregate_corruption_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Average degradation over the 20 corrupted conditions, excluding the clean reference."""
    required = {"run_id", "condition", "ece_delta", "macro_f1_delta"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"corruption_curves.csv missing required columns: {sorted(missing)}")
    conditions = curves["condition"].astype(str)
    corrupted = curves.loc[~conditions.isin({"clean", "clean:0"})].copy()
    counts = corrupted.groupby("run_id").size()
    bad = counts[counts != 20]
    if not bad.empty:
        details = {str(k): int(v) for k, v in bad.items()}
        raise ValueError(f"each corruption-subset run must have exactly 20 corrupted conditions; got {details}")
    if corrupted.empty:
        raise ValueError("corruption_curves.csv contains no corrupted-condition rows")
    return corrupted.groupby("run_id").agg(
        corruption_mean_ece_delta=("ece_delta", "mean"),
        corruption_mean_macro_f1_delta=("macro_f1_delta", "mean"),
    ).reset_index()


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--phase", default="primary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-evaluation", action="store_true", help="fail unless each finished run has ID and requested OOD evaluation outputs")
    args = ap.parse_args()
    cfg, _, run_df, split_sha, commit = load_protocol(args)
    if args.phase == "primary":
        require_frozen(cfg)
    root = os.path.join(args.drive_root, args.phase, "runs")
    rows = []
    for r in run_df.itertuples():
        d = os.path.join(root, r.run_id)
        sp = os.path.join(d, "status.json")
        row = {"run_id": r.run_id, "backbone": r.backbone, "framework": r.framework,
               "augmentation": r.augmentation, "seed": int(r.seed), "state": "missing"}
        if os.path.exists(sp):
            st = read_json(sp)
            expected_identity = expected_run_identity(r.run_id, cfg, split_sha, commit, args.phase, False)
            if st.get("state") in {"completed", "cap_limited", "failed", "running"} and not st.get("identity"):
                raise ValueError(f"{r.run_id}: status.json lacks the run identity")
            if st.get("identity"):
                check_identity(st["identity"], expected_identity, allow_commit_change=True)
            row.update({"state": st.get("state"), "stop_reason": st.get("stop_reason"), "last_epoch": st.get("last_epoch"),
                        "best_epoch": st.get("best_epoch"), "best_val_macro_f1": st.get("best_val_macro_f1"),
                        "gpu_model": st.get("gpu_model"), "attempts": st.get("attempts"),
                        "cap_limited_at_resource_ceiling": st.get("cap_limited_at_resource_ceiling", False)})
            eligible_for_evaluation = st.get("state") == "completed" or (
                st.get("state") == "cap_limited" and st.get("cap_limited_at_resource_ceiling") is True)
            tm = os.path.join(d, "test_metrics.json")
            if os.path.exists(tm):
                m = read_json(tm)
                if not m.get("identity"):
                    raise ValueError(f"{r.run_id}: test_metrics.json lacks the run identity")
                check_identity(m["identity"], expected_identity, allow_commit_change=True)
                required_metrics = {"accuracy", "macro_f1", "ece", "macro_cwece", "brier", "per_class_recall", "support"}
                metrics_ok = all(required_metrics.issubset(set(m.get(part, {}))) and
                                 {"native", "msp", "entropy"}.issubset(set(m.get(part, {}).get("aurc", {})))
                                 for part in ("val", "test"))
                op = m.get("melanoma_operating_points", {})
                probability = op.get("probability", {})
                selective = op.get("selective", {})
                operating_ok = (set(probability) == {"specificity_90", "specificity_95"}
                                and all({"threshold", "sensitivity", "specificity"}.issubset(set(v))
                                        for v in probability.values())
                                and set(selective) == {"native", "msp", "entropy"}
                                and all(set(levels) == {"coverage_80", "coverage_90"} and
                                        all({"threshold", "coverage", "selective_risk", "sensitivity", "achieved_specificity"}.issubset(set(v))
                                            for v in levels.values())
                                        for levels in selective.values()))
                if args.require_evaluation and eligible_for_evaluation and (not metrics_ok or not operating_ok):
                    row["evaluation_missing"] = True
                for part in ["val", "test"]:
                    for k in ["accuracy", "macro_f1", "ece", "macro_cwece", "brier"]:
                        if k in m.get(part, {}): row[f"{part}_{k}"] = m[part][k]
                    for score, value in m.get(part, {}).get("aurc", {}).items(): row[f"{part}_aurc_{score}"] = value
                    for i, value in enumerate(m.get(part, {}).get("per_class_recall", [])):
                        row[f"{part}_recall_class_{i}"] = value
                    for i, value in enumerate(m.get(part, {}).get("support", [])):
                        row[f"{part}_support_class_{i}"] = value
                flatten_operating_points(row, m)
            elif args.require_evaluation and eligible_for_evaluation:
                row["evaluation_missing"] = True
            om = os.path.join(d, "ood_metrics.json")
            if os.path.exists(om):
                ood_record = read_json(om)
                if not ood_record.get("identity"):
                    raise ValueError(f"{r.run_id}: ood_metrics.json lacks the run identity")
                check_identity(ood_record["identity"], expected_identity, allow_commit_change=True)
                ood = ood_record.get("scores", {})
                for dataset, score_set in ood.items():
                    for score, values in score_set.items():
                        for metric in ["auroc", "fpr95", "aupr", "ood_prevalence", "n_id", "n_ood", "balanced_subsample_aupr", "balanced_subsample_n_per_class"]:
                            if metric in values: row[f"{dataset}_{score}_{metric}"] = values[metric]
                if args.require_evaluation and eligible_for_evaluation:
                    required_scores = {"native", "msp", "entropy"}
                    required_ood_metrics = {"auroc", "fpr95", "aupr", "ood_prevalence", "n_id", "n_ood",
                                            "balanced_subsample_aupr", "balanced_subsample_n_per_class"}
                    for required_dataset in ("pad_ufes20", "cifar100"):
                        dataset_scores = ood.get(required_dataset, {})
                        if not required_scores.issubset(set(dataset_scores)) or any(
                                not required_ood_metrics.issubset(set(dataset_scores.get(score, {})))
                                for score in required_scores):
                            row["ood_evaluation_missing"] = True
            elif args.require_evaluation and eligible_for_evaluation:
                row["ood_evaluation_missing"] = True
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df["state"].value_counts().to_string())
    cap = df[df["state"] == "cap_limited"]
    if len(cap):
        print(f"\n{len(cap)} cap-limited run(s) -> apply the preregistered extension rule; resource-ceiling-limited runs remain disclosed as such:")
        print(cap["run_id"].to_string(index=False))
    failed = df[df["state"] == "failed"]
    if len(failed):
        print("\nfailed runs (resume the SAME run; never change the seed):")
        print(failed["run_id"].to_string(index=False))

    logs = glob.glob(os.path.join(args.drive_root, "worker_logs", f"logs_*_{args.phase}.csv"))
    log_errors = []
    if logs:
        parts = []
        file_owner = {}
        for path in logs:
            part = pd.read_csv(path)
            if "run_id" not in part or "state" not in part or "finished_at" not in part:
                log_errors.append(f"invalid worker log schema: {path}"); continue
            for run_id in part["run_id"].dropna().astype(str).unique():
                if run_id in file_owner and file_owner[run_id] != path:
                    log_errors.append(f"run {run_id} appears in multiple worker logs: {file_owner[run_id]} and {path}")
                file_owner[run_id] = path
            part["_log_path"] = path; parts.append(part)
        if parts:
            seen = pd.concat(parts, ignore_index=True)
            for run_id, group in seen.groupby("run_id"):
                if int((group["state"] == "completed").sum()) > 1:
                    log_errors.append(f"run {run_id} has more than one completed worker-log entry")
                latest = group.sort_values("finished_at").iloc[-1]
                row = df[df["run_id"] == run_id]
                if len(row) and row.iloc[0]["state"] in {"completed", "cap_limited", "failed"}:
                    if str(latest["state"]) != str(row.iloc[0]["state"]):
                        log_errors.append(f"worker-log/status contradiction for {run_id}: latest log={latest['state']}, status.json={row.iloc[0]['state']}")
            print(f"\nworker logs: {len(logs)} file(s), {seen['run_id'].nunique()} unique run IDs")
    if log_errors:
        print("\nWORKER LOG INTEGRITY ERRORS:")
        for e in log_errors: print(" -", e)

    curves_path = os.path.join(args.drive_root, "results", "stats", "corruption_curves.csv")
    if args.phase == "primary" and os.path.exists(curves_path):
        curves = pd.read_csv(curves_path)
        if not {"config_hash", "split_sha256"}.issubset(curves.columns):
            raise ValueError("corruption_curves.csv lacks config/split identity columns")
        if not (curves["config_hash"].astype(str) == config_hash(cfg)).all():
            raise ValueError("corruption_curves.csv was generated from a different config hash")
        if not (curves["split_sha256"].astype(str) == split_sha).all():
            raise ValueError("corruption_curves.csv was generated from a different split manifest")
        agg = aggregate_corruption_curves(curves)
        df = df.merge(agg, on="run_id", how="left")
    out = args.out or os.path.join(args.drive_root, "results", "final_factorial_v1.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    early_stopped = int((df["state"] == "completed").sum())
    ceiling_flag = df.get("cap_limited_at_resource_ceiling", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ceiling_limited = int(((df["state"] == "cap_limited") & ceiling_flag).sum())
    eligible_terminal = (df["state"] == "completed") | ((df["state"] == "cap_limited") & ceiling_flag)
    nonterminal = int((~eligible_terminal).sum())
    evaluation_missing = bool(args.require_evaluation and (
        ("evaluation_missing" in df.columns and df["evaluation_missing"].fillna(False).astype(bool).any())
        or ("ood_evaluation_missing" in df.columns and df["ood_evaluation_missing"].fillna(False).astype(bool).any())
    ))
    print(f"\nPrimary matrix summary: {early_stopped} early-stopped; {ceiling_limited} reached the resource ceiling; "
          f"{nonterminal} remain missing/failed/running/not-yet-eligible. Table saved to {out}")
    if nonterminal or log_errors or evaluation_missing: raise SystemExit(1)

if __name__ == "__main__": main()
