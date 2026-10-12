"""Lesion-grouped train/validation/test split: generation, saving and verification.

The manifest is generated ONCE, hashed, committed and then only loaded. No training,
OOD or corruption script may regenerate it.
"""
import os

import numpy as np
import pandas as pd

from edl.constants import CLASS_NAMES, CLASS_TO_IDX
from edl.utils.hashing import sha256_file

MANIFEST_COLUMNS = ["image_id", "lesion_id", "dx", "fold", "split"]
METADATA_COLUMNS = ["lesion_id", "image_id", "dx"]


class SplitError(RuntimeError):
    """Raised when the split or its manifest fails an integrity check."""


def read_metadata(path) -> pd.DataFrame:
    """Read HAM10000_metadata.csv in its original row order (the order matters for reproducibility)."""
    meta = pd.read_csv(path)
    missing = [c for c in METADATA_COLUMNS if c not in meta.columns]
    if missing:
        raise SplitError(f"metadata is missing columns {missing}")
    if meta["image_id"].duplicated().any():
        raise SplitError("duplicate image_id in metadata")
    bad = sorted(set(meta["dx"]) - set(CLASS_NAMES))
    if bad:
        raise SplitError(f"unrecognized diagnosis labels: {bad}")
    if meta[["image_id", "lesion_id", "dx"]].isna().any().any():
        raise SplitError("metadata contains missing values")
    return meta


def generate_split_manifest(meta: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    """Deterministic StratifiedGroupKFold split. Returns the manifest DataFrame."""
    from sklearn.model_selection import StratifiedGroupKFold  # imported late so tests can skip it

    g = split_cfg["generator"]
    roles = split_cfg["roles"]
    y = meta[g["label_column"]].map(CLASS_TO_IDX).astype(int).values
    groups = meta[g["group_column"]].values
    sgkf = StratifiedGroupKFold(n_splits=g["n_splits"], shuffle=g["shuffle"], random_state=g["random_state"])
    fold = np.full(len(meta), -1, dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(np.arange(len(meta)), y, groups=groups)):
        fold[test_idx] = k
    if (fold < 0).any():
        raise SplitError("some images were not assigned to a fold")

    role_of_fold = {f: "train" for f in roles["train_folds"]}
    role_of_fold[roles["validation_fold"]] = "val"
    role_of_fold[roles["test_fold"]] = "test"
    out = meta[["image_id", "lesion_id", "dx"]].copy()
    out["fold"] = fold
    out["split"] = pd.Series(fold).map(role_of_fold).values
    return out[MANIFEST_COLUMNS]


def save_manifest(df: pd.DataFrame, path) -> str:
    """Write the manifest with a fixed serialization (LF endings, fixed column order) and return its SHA-256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def verify_manifest(df: pd.DataFrame, meta: pd.DataFrame = None, expected: dict = None) -> dict:
    """Semantic checks. Raises SplitError on any failure and returns a summary.

    expected: optional regression expectations (images, lesions, counts). A mismatch there is
    reported in the summary under 'regression_mismatches' and does not raise.
    """
    if list(df.columns) != MANIFEST_COLUMNS:
        raise SplitError(f"manifest columns must be {MANIFEST_COLUMNS}, got {list(df.columns)}")
    if df["image_id"].duplicated().any():
        raise SplitError("an image appears more than once in the manifest")
    if not df["split"].isin(["train", "val", "test"]).all():
        raise SplitError("every image must belong to exactly one of train/val/test")
    if not df["dx"].isin(CLASS_NAMES).all():
        raise SplitError("unrecognized class label in manifest")
    if df.isna().any().any():
        raise SplitError("manifest contains missing values")

    lesions = {s: set(df.loc[df["split"] == s, "lesion_id"]) for s in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = lesions[a] & lesions[b]
        if shared:
            raise SplitError(f"{len(shared)} lesion(s) appear in both {a} and {b}")
    # one fold per lesion is implied by the checks above, but verify the fold/split link too
    if (df.groupby("fold")["split"].nunique() > 1).any():
        raise SplitError("a fold maps to more than one partition")

    if meta is not None:
        if set(meta["image_id"]) != set(df["image_id"]):
            raise SplitError("manifest and metadata contain different image IDs")
        m = meta.set_index("image_id").loc[df["image_id"]]
        if not (m["lesion_id"].values == df["lesion_id"].values).all():
            raise SplitError("lesion_id differs between metadata and manifest")
        if not (m["dx"].values == df["dx"].values).all():
            raise SplitError("dx differs between metadata and manifest")

    summary = {
        "images": int(len(df)),
        "lesions": int(df["lesion_id"].nunique()),
        "counts": {k: int(v) for k, v in df.groupby("split").size().to_dict().items()},
        "class_counts": {s: {c: int(n) for c, n in g["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).items()}
                         for s, g in df.groupby("split")},
        "regression_mismatches": [],
    }
    if expected:
        if "images" in expected and expected["images"] != summary["images"]:
            summary["regression_mismatches"].append(("images", expected["images"], summary["images"]))
        if "lesions" in expected and expected["lesions"] != summary["lesions"]:
            summary["regression_mismatches"].append(("lesions", expected["lesions"], summary["lesions"]))
        if "counts" in expected and dict(expected["counts"]) != summary["counts"]:
            summary["regression_mismatches"].append(("counts", expected["counts"], summary["counts"]))
    return summary


def load_verified_manifest(path, expected_sha256: str, meta: pd.DataFrame = None) -> pd.DataFrame:
    """Load the frozen manifest: check the file hash (line-ending safe), then the semantics."""
    if not expected_sha256 or str(expected_sha256).startswith("TO_FILL"):
        raise SplitError("the split hash is not recorded in the config yet")
    got = sha256_file(path, normalize_eol=True)
    if got != expected_sha256:
        raise SplitError(f"split manifest hash mismatch: got {got}, expected {expected_sha256}")
    df = pd.read_csv(path)
    verify_manifest(df, meta)
    return df
