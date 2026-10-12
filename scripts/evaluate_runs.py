"""Post-training evaluation of ID, near/far OOD, and corruption conditions. Never run before protocol freeze."""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from edl.cli import REPO, add_common_args, check_gpu, check_primary_environment, expected_run_identity, load_protocol  # noqa: E402
from edl.config import require_frozen, resolve  # noqa: E402
from edl.data.corruption import condition_list, corrupted_dataset, load_corruption_manifest  # noqa: E402
from edl.data.ham10000 import build_eval_transform, find_image_paths, make_dataset, make_loader  # noqa: E402
from edl.data.ood import CIFAR_COLUMNS, PAD_COLUMNS, cifar100_dataset, load_ood_manifest, pad_dataset  # noqa: E402
from edl.evaluation.export import ID_BUNDLE, TEST_METRICS, export_id_predictions  # noqa: E402
from edl.evaluation.metrics import ood_metrics  # noqa: E402
from edl.evaluation.predict import load_bundle, predict_arrays, save_bundle  # noqa: E402
from edl.frameworks import build_framework  # noqa: E402
from edl.training.checkpoint import BEST, RunCheckpointer, check_identity, sync_file  # noqa: E402
from edl.training.engine import RunSpec, load_best_model  # noqa: E402
from edl.training.manifest import RunStatus, filter_runs  # noqa: E402
from edl.utils.hashing import sha256_file  # noqa: E402
from edl.utils.io import write_json_atomic  # noqa: E402


def _predict(model, framework, ds, cfg, device, seed):
    loader, _ = make_loader(ds, cfg["training"]["batch_size"], cfg["data_loader"], seed=seed)
    return predict_arrays(model, framework, loader, device, bool(cfg["data_loader"]["non_blocking_transfer"]))


def _save(name, arrays, meta, work_dir, drive_dir):
    path = os.path.join(work_dir, name)
    save_bundle(path, arrays, meta)
    sync_file(path, drive_dir)


def _balanced_aupr(id_scores, ood_scores, seed):
    n = min(len(id_scores), len(ood_scores))
    if n == 0: raise ValueError("balanced AUPR requires nonempty ID and OOD samples")
    rng = np.random.default_rng(int(seed))
    ii = rng.choice(len(id_scores), n, replace=False)
    oi = rng.choice(len(ood_scores), n, replace=False)
    y = np.r_[np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)]
    scores = np.r_[np.asarray(id_scores)[ii], np.asarray(ood_scores)[oi]]
    return float(average_precision_score(y, scores)), n


def _ood_score_map(id_arrays, ood_arrays):
    """Return prespecified score pairs; include F-EDL's secondary total-uncertainty score when present."""
    score_map = {"native": ("test_native_uncertainty", "native_uncertainty"),
                 "msp": ("test_msp_uncertainty", "msp_uncertainty"),
                 "entropy": ("test_entropy", "entropy")}
    if "test_total_uncertainty" in id_arrays and "total_uncertainty" in ood_arrays:
        score_map["total_uncertainty"] = ("test_total_uncertainty", "total_uncertainty")
    return score_map


def _check_identity(bundle_meta, expected, run_id):
    identity = bundle_meta.get("identity")
    if not isinstance(identity, dict): raise ValueError(f"{run_id}: bundle metadata has no identity")
    check_identity(identity, expected, allow_commit_change=True)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--phase", default="primary", choices=["smoke", "pilot", "primary"])
    ap.add_argument("--sets", default="id,pad_ufes20,cifar100,corruption",
                    help="comma-separated: id,pad_ufes20,cifar100,corruption")
    ap.add_argument("--backbone"); ap.add_argument("--framework"); ap.add_argument("--augmentation")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--pad-root", default="/content/data/pad_ufes_20")
    ap.add_argument("--cifar-root", default="/content/data/cifar100")
    ap.add_argument("--require-gpu", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.phase != "primary": raise SystemExit("held-out test/OOD/corruption evaluation is only allowed after all primary training runs and protocol freeze")
    cfg, split_df, run_df, split_sha, commit = load_protocol(args)
    require_frozen(cfg)
    try: check_primary_environment()
    except Exception as exc: raise SystemExit(str(exc))
    check_gpu(args.require_gpu)
    sets = set(args.sets.split(","))
    unknown = sets - {"id", "pad_ufes20", "cifar100", "corruption"}
    if unknown: raise SystemExit(f"unknown evaluation sets: {sorted(unknown)}")
    if not sets: raise SystemExit("no evaluation sets selected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = filter_runs(run_df, args.backbone, args.framework, args.augmentation, args.seed)
    out_root = os.path.join(args.drive_root, "primary", "runs")
    work_root = os.path.join(args.work_root, "primary", "runs")
    # The frozen roadmap places held-out test evaluation after the full primary matrix. Check
    # every status before opening test/OOD images, including when CLI filters select a subset.
    not_ready = []
    for run in run_df.itertuples():
        status = RunStatus(os.path.join(out_root, run.run_id)).read()
        eligible = status.get("state") == "completed" or (
            status.get("state") == "cap_limited" and status.get("cap_limited_at_resource_ceiling") is True)
        if not eligible:
            not_ready.append(f"{run.run_id}={status.get('state', 'missing')}"
                             + ("" if status.get("state") != "cap_limited" else " (not at resource ceiling)"))
            continue
        expected_status_identity = expected_run_identity(run.run_id, cfg, split_sha, commit, "primary", False)
        try:
            check_identity(status.get("identity", {}), expected_status_identity, allow_commit_change=True)
        except Exception as exc:
            not_ready.append(f"{run.run_id}=identity mismatch ({exc})")
    if not_ready:
        examples = ", ".join(not_ready[:10])
        raise SystemExit(f"held-out evaluation is blocked until all 144 primary runs are eligible; "
                         f"{len(not_ready)} run(s) remain incomplete: {examples}")
    pad = cifar = corruption_manifest = paths = None
    if "pad_ufes20" in sets:
        m = load_ood_manifest(resolve(cfg["ood"]["pad_ufes_20"]["manifest"], REPO), cfg["ood"]["pad_ufes_20"]["sha256"], PAD_COLUMNS)
        pad = pad_dataset(m, args.pad_root, build_eval_transform())
    if "cifar100" in sets:
        m = load_ood_manifest(resolve(cfg["ood"]["cifar100"]["manifest"], REPO), cfg["ood"]["cifar100"]["sha256"], CIFAR_COLUMNS)
        cifar = cifar100_dataset(m, args.cifar_root, build_eval_transform())
    test = split_df[split_df["split"] == "test"].reset_index(drop=True)
    if sets.intersection({"id", "corruption"}):
        paths = find_image_paths(args.data_root)
    if "corruption" in sets:
        corruption_manifest = load_corruption_manifest(resolve(cfg["corruption"]["manifest"], REPO),
                                                         cfg["corruption"]["sha256"], test, paths, cfg)
    sub = cfg["corruption"]["subset"]
    for r in runs.itertuples():
        run_id = r.run_id
        drive_dir, work_dir = os.path.join(out_root, run_id), os.path.join(work_root, run_id)
        st = RunStatus(drive_dir).read()
        evaluable = st.get("state") == "completed" or (st.get("state") == "cap_limited" and
                    st.get("cap_limited_at_resource_ceiling") is True)
        if not evaluable: continue
        os.makedirs(work_dir, exist_ok=True)
        spec = RunSpec(run_id, r.backbone, r.framework, r.augmentation, int(r.seed))
        expected = expected_run_identity(run_id, cfg, split_sha, commit, "primary", False)
        saved_identity = st.get("identity", {})
        check_identity(saved_identity, expected, allow_commit_change=True)
        best_drive = os.path.join(drive_dir, BEST)
        RunCheckpointer(work_dir, drive_dir, expected).verify_drive_checkpoint(BEST)
        best_local = os.path.join(work_dir, BEST)
        if not os.path.exists(best_local) or sha256_file(best_local) != sha256_file(best_drive):
            shutil.copyfile(best_drive, best_local)
        model, ck = load_best_model(spec, cfg, best_local, device, expected)
        fw = build_framework(r.framework, cfg)
        meta = {"identity": expected, "run": spec.__dict__, "best_epoch": int(ck["epoch"])}

        if "id" in sets:
            target_path = os.path.join(drive_dir, ID_BUNDLE)
            test_metrics_path = os.path.join(drive_dir, TEST_METRICS)
            if args.overwrite or not os.path.exists(target_path) or not os.path.exists(test_metrics_path):
                export_id_predictions(spec, cfg, split_df, paths or find_image_paths(args.data_root), work_dir, device, expected)
                for name in [ID_BUNDLE, TEST_METRICS]:
                    sync_file(os.path.join(work_dir, name), drive_dir)
                print(f"[{run_id}] exported ID validation/test metrics")
            else:
                id_arrays, id_meta = load_bundle(target_path); _check_identity(id_meta, expected, run_id)
                expected_test_ids = test["image_id"].astype(str).tolist()
                expected_val_ids = split_df.loc[split_df["split"] == "val", "image_id"].astype(str).tolist()
                if id_arrays.get("test_image_ids", np.array([])).astype(str).tolist() != expected_test_ids:
                    raise ValueError(f"{run_id}: existing ID bundle test image order mismatch")
                if id_arrays.get("val_image_ids", np.array([])).astype(str).tolist() != expected_val_ids:
                    raise ValueError(f"{run_id}: existing ID bundle validation image order mismatch")
                from edl.utils.io import read_json
                tm = read_json(test_metrics_path)
                _check_identity(tm, expected, run_id)

        if "pad_ufes20" in sets and pad is not None:
            fname = "predictions_pad_ufes20.npz"; target = os.path.join(drive_dir, fname)
            if args.overwrite or not os.path.exists(target):
                arrays = _predict(model, fw, pad, cfg, device, spec.seed)
                arrays["image_ids"] = np.array(pad.image_ids)
                arrays["cluster_ids"] = np.array(pad.cluster_ids)
                _save(fname, arrays, meta, work_dir, drive_dir); print(f"[{run_id}] saved {fname}")
            else:
                old_arrays, m = load_bundle(target); _check_identity(m, expected, run_id)
                if old_arrays.get("image_ids", np.array([])).astype(str).tolist() != pad.image_ids:
                    raise ValueError(f"{run_id}: existing PAD-UFES-20 prediction image order differs from its manifest")
                if old_arrays.get("cluster_ids", np.array([])).astype(str).tolist() != pad.cluster_ids:
                    raise ValueError(f"{run_id}: existing PAD-UFES-20 patient-cluster order differs from its manifest")
        if "cifar100" in sets and cifar is not None:
            fname = "predictions_cifar100.npz"; target = os.path.join(drive_dir, fname)
            if args.overwrite or not os.path.exists(target):
                arrays = _predict(model, fw, cifar, cfg, device, spec.seed)
                arrays["image_ids"] = np.array(cifar.image_ids)
                _save(fname, arrays, meta, work_dir, drive_dir); print(f"[{run_id}] saved {fname}")
            else:
                old_arrays, m = load_bundle(target); _check_identity(m, expected, run_id)
                if old_arrays.get("image_ids", np.array([])).astype(str).tolist() != cifar.image_ids:
                    raise ValueError(f"{run_id}: existing CIFAR-100 prediction image order differs from its manifest")

        in_subset = (r.backbone in sub["backbones"] and r.framework in sub["frameworks"]
                     and r.augmentation in sub["augmentations"])
        fname = "predictions_corruption.npz"; target = os.path.join(drive_dir, fname)
        if "corruption" in sets and in_subset and (args.overwrite or not os.path.exists(target)):
            conds = [("clean", 0, 0.0)] + condition_list(cfg)
            stack = {}; labels = None
            for name, level, param in conds:
                ds = (make_dataset(split_df, "test", paths, build_eval_transform()) if name == "clean"
                      else corrupted_dataset(test, paths, name, level, param, cfg["corruption"]["global_seed"], corruption_manifest))
                pred = _predict(model, fw, ds, cfg, device, spec.seed)
                for k in ["probs", "native_uncertainty", "msp_uncertainty", "entropy"]:
                    stack.setdefault(k, []).append(pred[k])
                if labels is None: labels = pred["labels"]
                elif not np.array_equal(labels, pred["labels"]): raise ValueError(f"{run_id}: labels/order differ across corruptions")
            arrays = {k: np.stack(v) for k, v in stack.items()}
            arrays["labels"] = labels
            arrays["image_ids"] = np.array(test["image_id"])
            arrays["lesion_ids"] = np.array(test["lesion_id"])
            arrays["conditions"] = np.array([f"{n}:{l}" for n, l, _ in conds])
            _save(fname, arrays, meta, work_dir, drive_dir); print(f"[{run_id}] saved {fname} ({len(conds)} conditions)")
        elif "corruption" in sets and in_subset and os.path.exists(target):
            _, m = load_bundle(target); _check_identity(m, expected, run_id)

        # Compute full-prevalence OOD metrics and fixed-seed balanced AUPR for all three score families.
        if sets.intersection({"pad_ufes20", "cifar100"}):
            id_path = os.path.join(drive_dir, ID_BUNDLE)
            if not os.path.exists(id_path):
                raise FileNotFoundError(f"{run_id}: OOD evaluation requires {ID_BUNDLE}; include --sets id first")
            id_arrays, id_meta = load_bundle(id_path); _check_identity(id_meta, expected, run_id)
            id_ids = id_arrays["test_image_ids"]
            if id_ids.tolist() != test["image_id"].astype(str).tolist():
                raise ValueError(f"{run_id}: ID prediction image order differs from frozen test manifest")
            ood_out_path = os.path.join(drive_dir, "ood_metrics.json")
            all_metrics = {"identity": expected, "scores": {}}
            if os.path.exists(ood_out_path):
                import json
                with open(ood_out_path, "r", encoding="utf-8") as f: all_metrics = json.load(f)
                _check_identity(all_metrics, expected, run_id)
                all_metrics.setdefault("scores", {})
            # --overwrite replaces only the requested dataset summaries; it must not erase metrics
            # for the other OOD dataset when a single-set repair is being run.
            if args.overwrite:
                for selected_dataset in sets.intersection({"pad_ufes20", "cifar100"}):
                    all_metrics["scores"].pop(selected_dataset, None)
            for dataset_name in ["pad_ufes20", "cifar100"]:
                if dataset_name not in sets: continue
                name = f"predictions_{dataset_name}.npz"; pth = os.path.join(drive_dir, name)
                if not os.path.exists(pth): continue
                ood_arrays, ometa = load_bundle(pth); _check_identity(ometa, expected, run_id)
                score_map = _ood_score_map(id_arrays, ood_arrays)
                all_metrics["scores"][dataset_name] = {}
                for score_name, (ik, ok) in score_map.items():
                    full = ood_metrics(id_arrays[ik], ood_arrays[ok], cfg["evaluation"]["fpr_recall_target"])
                    bal_seed = int(cfg["statistics"]["bootstrap_rng_seed"])
                    balanced, n_bal = _balanced_aupr(id_arrays[ik], ood_arrays[ok], bal_seed)
                    full["balanced_subsample_aupr"] = balanced
                    full["balanced_subsample_n_per_class"] = n_bal
                    full["balanced_subsample_seed"] = bal_seed
                    all_metrics["scores"][dataset_name][score_name] = full
            out_json = os.path.join(work_dir, "ood_metrics.json")
            write_json_atomic(all_metrics, out_json); sync_file(out_json, drive_dir)

    print("Evaluation pass finished. Confirm all required predictions and OOD summaries exist before aggregation/plots.")

if __name__ == "__main__": main()
