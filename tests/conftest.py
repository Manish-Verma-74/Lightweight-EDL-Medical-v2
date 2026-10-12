import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edl.config import load_config  # noqa: E402
from edl.constants import CLASS_NAMES  # noqa: E402

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "primary.yaml"))


@pytest.fixture()
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture()
def synthetic_ham(tmp_path):
    """A tiny fake HAM10000: random JPEGs + a manifest. Test images are deliberately NOT written to disk
    for the training tests that must never touch the test partition (see paths_without_test)."""
    from PIL import Image
    rng = np.random.default_rng(0)
    rows, paths = [], {}
    n = 0
    plan = {"train": 42, "val": 14, "test": 14}
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for split, count in plan.items():
        for j in range(count):
            image_id, lesion = f"ISIC_{n:07d}", f"HAM_{split}_{j // 2}"
            dx = CLASS_NAMES[j % len(CLASS_NAMES)]
            arr = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
            p = img_dir / f"{image_id}.jpg"
            Image.fromarray(arr).save(p)
            rows.append({"image_id": image_id, "lesion_id": lesion, "dx": dx, "fold": {"train": 0, "val": 4, "test": 3}[split],
                         "split": split})
            paths[image_id] = str(p)
            n += 1
    return pd.DataFrame(rows), paths


@pytest.fixture()
def tiny_cfg(cfg):
    """Config tuned for fast CPU tests (no pretrained download, no worker processes)."""
    cfg = dict(cfg)
    cfg["models"] = dict(cfg["models"], pretrained=False)
    cfg["data_loader"] = dict(cfg["data_loader"], num_workers=0, pin_memory=False, persistent_workers=False)
    cfg["training"] = dict(cfg["training"], batch_size=8, patience=2)
    cfg["checkpoint"] = dict(cfg["checkpoint"], sync_every_n_epochs=1, retry_backoff_seconds=0.0)
    return cfg
