import numpy as np
import pandas as pd
import pytest

from edl.data.ood import OODDataError, PAD_COLUMNS, build_pad_manifest, load_ood_manifest, save_ood_manifest


def make_pad(tmp_path, n=6, drop=None):
    from PIL import Image
    root = tmp_path / "pad"
    (root / "imgs_part_1").mkdir(parents=True)
    rows = []
    for i in range(n):
        img_id = f"PAT_{i // 2}_{i}_1.png"
        rows.append({"patient_id": f"PAT_{i // 2}", "lesion_id": i, "img_id": img_id, "diagnostic": "BCC"})
        if img_id != drop:
            Image.fromarray(np.full((8, 8, 3), i * 10, dtype=np.uint8)).save(root / "imgs_part_1" / img_id)
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root


def test_manifest_has_stable_order_clusters_and_hashes(tmp_path):
    m = build_pad_manifest(str(make_pad(tmp_path)))
    assert list(m.columns) == PAD_COLUMNS and m["img_id"].is_monotonic_increasing
    assert m["patient_id"].nunique() == 3 and m["sha256"].str.len().eq(64).all()


def test_missing_listed_image_is_an_error(tmp_path):
    with pytest.raises(OODDataError):
        build_pad_manifest(str(make_pad(tmp_path, drop="PAT_1_3_1.png")))


def test_manifest_hash_and_columns_are_enforced(tmp_path):
    m = build_pad_manifest(str(make_pad(tmp_path)))
    out = tmp_path / "pad_manifest.csv"
    h = save_ood_manifest(m, str(out))
    assert len(load_ood_manifest(str(out), h, PAD_COLUMNS)) == 6
    with pytest.raises(OODDataError):
        load_ood_manifest(str(out), "f" * 64, PAD_COLUMNS)
    with pytest.raises(OODDataError):
        load_ood_manifest(str(out), h, ["img_id"])
    with pytest.raises(OODDataError):
        load_ood_manifest(str(out), "TO_FILL_AFTER_MANIFEST_GENERATION", PAD_COLUMNS)
