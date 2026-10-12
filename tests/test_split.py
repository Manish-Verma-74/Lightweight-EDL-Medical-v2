import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from edl.data.split import (MANIFEST_COLUMNS, SplitError, generate_split_manifest, load_verified_manifest,  # noqa: E402
                            save_manifest, verify_manifest)
from edl.constants import CLASS_NAMES  # noqa: E402
from edl.utils.hashing import sha256_file  # noqa: E402


def make_meta(n_lesions=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for l in range(n_lesions):
        dx = CLASS_NAMES[rng.choice(len(CLASS_NAMES), p=[.03, .05, .11, .01, .11, .67, .02])]
        for k in range(int(rng.integers(1, 4))):
            rows.append({"lesion_id": f"HAM_{l:05d}", "image_id": f"ISIC_{l:05d}_{k}", "dx": dx})
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_lesion_grouped(cfg):
    meta = make_meta()
    a = generate_split_manifest(meta, cfg["data"]["split"])
    b = generate_split_manifest(meta, cfg["data"]["split"])
    assert a.equals(b)
    assert list(a.columns) == MANIFEST_COLUMNS
    summary = verify_manifest(a, meta)
    assert sum(summary["counts"].values()) == len(meta)
    assert set(a["split"]) == {"train", "val", "test"}


def test_every_fold_has_one_role(cfg):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    assert (a.groupby("fold")["split"].nunique() == 1).all()
    roles = a.groupby("fold")["split"].first().to_dict()
    assert roles[4] == "val" and roles[3] == "test" and all(roles[f] == "train" for f in [0, 1, 2, 5, 6])


def test_lesion_overlap_is_caught(cfg):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    test_lesion = a.loc[a["split"] == "test", "lesion_id"].iloc[0]
    a.loc[a["split"] == "train", "lesion_id"] = a.loc[a["split"] == "train", "lesion_id"].where(
        a.loc[a["split"] == "train", "lesion_id"] != a.loc[a["split"] == "train", "lesion_id"].iloc[0], test_lesion)
    with pytest.raises(SplitError):
        verify_manifest(a)


def test_duplicate_image_and_bad_label_are_caught(cfg):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    dup = pd.concat([a, a.iloc[[0]]])
    with pytest.raises(SplitError):
        verify_manifest(dup)
    bad = a.copy()
    bad.loc[0, "dx"] = "unknown"
    with pytest.raises(SplitError):
        verify_manifest(bad)


def test_metadata_mismatch_is_caught(cfg):
    meta = make_meta()
    a = generate_split_manifest(meta, cfg["data"]["split"])
    meta2 = meta.copy()
    meta2.loc[0, "dx"] = "mel" if meta2.loc[0, "dx"] != "mel" else "nv"
    with pytest.raises(SplitError):
        verify_manifest(a, meta2)


def test_regression_expectations_are_reported_not_forced(cfg):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    s = verify_manifest(a, expected=cfg["data"]["split"]["regression_expectations"])
    assert s["regression_mismatches"], "a synthetic split cannot match the HAM10000 counts"


def test_hash_roundtrip_and_crlf_tolerance(cfg, tmp_path):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    p = tmp_path / "split.csv"
    digest = save_manifest(a, p)
    assert digest == sha256_file(p)
    load_verified_manifest(str(p), digest)
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))        # simulate a Windows checkout
    load_verified_manifest(str(p), digest)
    with open(p, "ab") as f:
        f.write(b"x,y,nv,0,train\n")
    with pytest.raises(SplitError):
        load_verified_manifest(str(p), digest)


def test_placeholder_hash_is_refused(cfg, tmp_path):
    a = generate_split_manifest(make_meta(), cfg["data"]["split"])
    p = tmp_path / "split.csv"
    save_manifest(a, p)
    with pytest.raises(SplitError):
        load_verified_manifest(str(p), "TO_FILL_AFTER_SPLIT_GENERATION")
