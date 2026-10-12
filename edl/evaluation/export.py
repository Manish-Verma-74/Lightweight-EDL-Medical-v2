"""Export validation/test predictions and the preregistered metrics after primary training is complete."""
import os
import numpy as np
from edl.constants import NUM_CLASSES
from edl.data.ham10000 import build_eval_transform, make_dataset, make_loader
from edl.evaluation.metrics import aurc, classification_summary
from edl.evaluation.operating_points import (apply_coverage, apply_sensitivity, threshold_for_coverage,
                                              threshold_for_specificity)
from edl.evaluation.predict import load_bundle, predict_arrays, save_bundle
from edl.frameworks import build_framework
from edl.training.checkpoint import BEST
from edl.training.engine import RunSpec, load_best_model
from edl.utils.io import write_json_atomic

ID_BUNDLE = "predictions_id.npz"
TEST_METRICS = "test_metrics.json"
MELANOMA_IDX = 4
UNCERTAINTY_KEYS = {"native": "native_uncertainty", "msp": "msp_uncertainty", "entropy": "entropy"}


def _with_risk_coverage(summary, pred):
    errors = (pred["probs"].argmax(1) != pred["labels"]).astype(float)
    summary["aurc"] = {name: aurc(errors, pred[key]) for name, key in UNCERTAINTY_KEYS.items()}
    return summary


def _melanoma_operating_points(val, test, cfg):
    val_probs, test_probs = val["probs"], test["probs"]
    val_labels, test_labels = val["labels"], test["labels"]
    val_pos, test_pos = val_labels == MELANOMA_IDX, test_labels == MELANOMA_IDX
    test_pred_pos = test_probs.argmax(1) == MELANOMA_IDX
    test_errors = (test_probs.argmax(1) != test_labels).astype(float)
    out = {"positive_class_index": MELANOMA_IDX, "positive_class_name": "mel",
           "thresholds_selected_from": "validation predictions only", "probability": {}, "selective": {}}
    for target in cfg["evaluation"]["melanoma"]["specificity_targets"]:
        thr = threshold_for_specificity(val_probs[:, MELANOMA_IDX], val_pos, float(target))
        res = apply_sensitivity(test_probs[:, MELANOMA_IDX], test_pos, thr)
        res["target_specificity"] = float(target)
        out["probability"][f"specificity_{int(round(target*100))}"] = res
    for score_name, key in UNCERTAINTY_KEYS.items():
        out["selective"][score_name] = {}
        for target in cfg["evaluation"]["melanoma"]["coverage_targets"]:
            thr = threshold_for_coverage(val[key], float(target))
            res = apply_coverage(test[key], test_errors, test_pos, test_pred_pos, thr)
            res.update({"target_coverage": float(target), "uncertainty_threshold": float(thr)})
            out["selective"][score_name][f"coverage_{int(round(target*100))}"] = res
    return out


def export_id_predictions(spec: RunSpec, cfg: dict, manifest, paths: dict, work_dir: str, device, identity: dict):
    """Export validation and test bundles. Caller must invoke only in the post-training evaluation phase."""
    model, ck = load_best_model(spec, cfg, os.path.join(work_dir, BEST), device, identity)
    framework = build_framework(spec.framework, cfg)
    arrays, summaries, raw = {}, {}, {}
    expected_image_ids = {}
    for part in ["val", "test"]:
        ds = make_dataset(manifest, part, paths, build_eval_transform())
        loader, _ = make_loader(ds, cfg["training"]["batch_size"], cfg["data_loader"], seed=spec.seed)
        pred = predict_arrays(model, framework, loader, device, bool(cfg["data_loader"]["non_blocking_transfer"]))
        raw[part] = pred
        for k, v in pred.items(): arrays[f"{part}_{k}"] = v
        arrays[f"{part}_image_ids"] = np.array(ds.image_ids)
        arrays[f"{part}_lesion_ids"] = np.array(ds.lesion_ids)
        expected_image_ids[part] = [str(x) for x in ds.image_ids]
        summaries[part] = _with_risk_coverage(
            classification_summary(pred["probs"], pred["labels"], cfg["evaluation"]["ece_bins"]), pred)
    operating = _melanoma_operating_points(raw["val"], raw["test"], cfg)
    meta = {"identity": identity, "best_epoch": ck["epoch"], "best_val_macro_f1": ck["val_macro_f1"],
            "run": spec.__dict__, "split": "validation and held-out test in post-training evaluation"}
    bundle_path = os.path.join(work_dir, ID_BUNDLE)
    save_bundle(bundle_path, arrays, meta)
    verify_id_bundle(bundle_path, len(raw["val"]["labels"]), len(raw["test"]["labels"]),
                     expected_val_ids=expected_image_ids["val"], expected_test_ids=expected_image_ids["test"],
                     expected_identity=identity)
    write_json_atomic({"val": summaries["val"], "test": summaries["test"],
                       "melanoma_operating_points": operating, "best_epoch": ck["epoch"], "identity": identity},
                      os.path.join(work_dir, TEST_METRICS))
    return summaries["val"], summaries["test"]


def verify_id_bundle(path: str, n_val: int, n_test: int, *, expected_val_ids=None,
                     expected_test_ids=None, expected_identity=None) -> None:
    """Reload and validate the saved ID bundle before metrics are accepted or persisted."""
    arrays, meta = load_bundle(path)
    for part, n, expected_ids in [
        ("val", n_val, expected_val_ids), ("test", n_test, expected_test_ids)
    ]:
        required = [f"{part}_probs", f"{part}_labels", f"{part}_image_ids",
                    f"{part}_native_uncertainty", f"{part}_msp_uncertainty", f"{part}_entropy"]
        missing = [k for k in required if k not in arrays]
        if missing:
            raise ValueError(f"ID bundle is missing {part} arrays: {missing}")
        probs = arrays[f"{part}_probs"]
        labels = arrays[f"{part}_labels"]
        image_ids = arrays[f"{part}_image_ids"].astype(str).tolist()
        if probs.shape != (n, NUM_CLASSES):
            raise ValueError(f"{part}_probs has shape {probs.shape}, expected {(n, NUM_CLASSES)}")
        if labels.shape != (n,):
            raise ValueError(f"{part}_labels has shape {labels.shape}, expected {(n,)}")
        if not np.isfinite(probs).all() or (probs < -1e-6).any() or (probs > 1+1e-6).any():
            raise ValueError(f"{part}_probs contains invalid probabilities")
        if not np.allclose(probs.sum(1), 1.0, atol=1e-4):
            raise ValueError(f"{part}_probs rows do not sum to 1")
        if len(set(image_ids)) != n:
            raise ValueError(f"{part}_image_ids are not unique")
        if expected_ids is not None and image_ids != [str(x) for x in expected_ids]:
            raise ValueError(f"{part}_image_ids differ from the expected frozen-manifest order")
        for k in ["native_uncertainty", "msp_uncertainty", "entropy"]:
            if arrays[f"{part}_{k}"].shape != (n,):
                raise ValueError(f"{part}_{k} has the wrong shape")
            if not np.isfinite(arrays[f"{part}_{k}"]).all():
                raise ValueError(f"{part}_{k} contains non-finite values")
    if "identity" not in meta:
        raise ValueError("bundle metadata lacks the run identity")
    if expected_identity is not None and meta["identity"] != expected_identity:
        raise ValueError("saved ID bundle metadata identity does not match the requested run identity")
