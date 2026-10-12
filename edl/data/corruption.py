"""Deterministic corruption generation plus manifest/hash verification."""
import hashlib
import os

import numpy as np
import pandas as pd

from edl.utils.hashing import sha256_bytes, sha256_file

CORRUPTION_ORDER = ["gaussian_noise", "gaussian_blur", "brightness", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTION_COLUMNS = ["image_id", "corruption", "severity", "param", "seed", "source_sha256", "output_sha256"]


def derive_seed(global_seed: int, image_id: str, corruption: str, severity: int) -> int:
    msg = f"{global_seed}|{image_id}|{corruption}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def corrupt_image(x: np.ndarray, corruption: str, param: float, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) array")
    if not np.isfinite(x).all() or x.min() < 0.0 or x.max() > 1.0:
        raise ValueError("input must be finite and lie in [0, 1]")
    if corruption == "gaussian_noise":
        rng = np.random.default_rng(seed)
        y = x + rng.normal(0.0, param, size=x.shape).astype(np.float32)
    elif corruption == "gaussian_blur":
        from scipy.ndimage import gaussian_filter
        y = gaussian_filter(x, sigma=(param, param, 0.0), mode="reflect", truncate=4.0)
    elif corruption == "brightness":
        y = x + np.float32(param)
    elif corruption == "contrast":
        mean = x.mean(axis=(0, 1), keepdims=True)
        y = (x - mean) * np.float32(param) + mean
    else:
        raise ValueError(f"unknown corruption {corruption}")
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def load_resized(path: str) -> np.ndarray:
    import torchvision.transforms as T
    from PIL import Image
    from edl.constants import INPUT_SIZE
    resize = T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR)
    with Image.open(path) as im:
        img = resize(im.convert("RGB"))
    return np.asarray(img, dtype=np.float32) / 255.0


def condition_list(cfg: dict):
    out = []
    for name in CORRUPTION_ORDER:
        for level, param in enumerate(cfg["corruption"]["types"][name]["severities"], start=1):
            out.append((name, level, float(param)))
    return out


def load_corruption_manifest(path: str, expected_sha256: str, test_manifest, paths: dict, cfg: dict) -> pd.DataFrame:
    """Verify file hash, exact image/condition coverage, source hash, severity/seed settings and unique keys."""
    if not expected_sha256 or str(expected_sha256).startswith("TO_FILL"):
        raise ValueError("corruption manifest SHA-256 is not frozen in configs/primary.yaml")
    digest = sha256_file(path, normalize_eol=True)
    if digest != expected_sha256:
        raise ValueError(f"corruption manifest SHA-256 mismatch: got {digest}, expected {expected_sha256}")
    df = pd.read_csv(path)
    if list(df.columns) != CORRUPTION_COLUMNS:
        raise ValueError(f"corruption manifest columns must be {CORRUPTION_COLUMNS}, got {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("corruption manifest contains missing values")
    keys = ["image_id", "corruption", "severity"]
    if df.duplicated(keys).any():
        raise ValueError("duplicate image/corruption/severity rows in corruption manifest")
    image_ids = test_manifest["image_id"].astype(str).tolist()
    conds = condition_list(cfg)
    expected_keys = {(img, name, int(level)) for img in image_ids for name, level, _ in conds}
    actual_keys = set(zip(df.image_id.astype(str), df.corruption.astype(str), df.severity.astype(int)))
    if actual_keys != expected_keys or len(df) != len(expected_keys):
        raise ValueError(f"corruption manifest coverage mismatch: expected {len(expected_keys)} unique rows, got {len(df)}")
    lookup = df.set_index(keys)
    global_seed = int(cfg["corruption"]["global_seed"])
    params = {(n, int(level)): float(param) for n, level, param in conds}
    source_hashes = {}
    for image_id in image_ids:
        if image_id not in paths:
            raise ValueError(f"missing source image {image_id} for corruption manifest")
        source_hashes[image_id] = sha256_file(paths[image_id])
    for image_id in image_ids:
        source_hash = source_hashes[image_id]
        for name, level, _ in conds:
            row = lookup.loc[(image_id, name, int(level))]
            want_seed = derive_seed(global_seed, image_id, name, int(level))
            if int(row["seed"]) != want_seed:
                raise ValueError(f"derived seed mismatch for {(image_id, name, level)}")
            if not np.isclose(float(row["param"]), params[(name, int(level))], rtol=0, atol=1e-12):
                raise ValueError(f"severity parameter mismatch for {(image_id, name, level)}")
            if str(row["source_sha256"]) != source_hash:
                raise ValueError(f"source image hash mismatch for {image_id}")
            if not re_full_sha(str(row["output_sha256"])):
                raise ValueError(f"invalid output hash for {(image_id, name, level)}")
    return df


def re_full_sha(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def corrupted_dataset(manifest_part, paths: dict, corruption: str, level: int, param: float, global_seed: int,
                      corruption_manifest: pd.DataFrame = None):
    """Dataset for one corruption condition. If a manifest is supplied, source and output hashes are enforced."""
    import torch
    import torchvision.transforms as T
    from torch.utils.data import Dataset
    from edl.constants import CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    lookup = None
    if corruption_manifest is not None:
        lookup = corruption_manifest.set_index(["image_id", "corruption", "severity"])

    class Corrupted(Dataset):
        def __init__(self):
            self.image_ids = manifest_part["image_id"].astype(str).tolist()
            self.lesion_ids = manifest_part["lesion_id"].tolist()
            self.labels = manifest_part["dx"].map(CLASS_TO_IDX).astype(int).tolist()
        def __len__(self): return len(self.image_ids)
        def __getitem__(self, i):
            image_id = self.image_ids[i]
            path = paths[image_id]
            x = load_resized(path)
            seed = derive_seed(global_seed, image_id, corruption, level)
            if lookup is not None:
                row = lookup.loc[(image_id, corruption, int(level))]
                source_hash = sha256_file(path)
                if str(row["source_sha256"]) != source_hash or int(row["seed"]) != seed:
                    raise ValueError(f"corruption manifest source/seed mismatch for {(image_id, corruption, level)}")
                if not np.isclose(float(row["param"]), float(param), rtol=0, atol=1e-12):
                    raise ValueError(f"corruption manifest parameter mismatch for {(image_id, corruption, level)}")
            y = corrupt_image(x, corruption, param, seed)
            if lookup is not None:
                got = sha256_bytes(np.ascontiguousarray(y).tobytes())
                expected = str(lookup.loc[(image_id, corruption, int(level)), "output_sha256"])
                if got != expected:
                    raise ValueError(f"corrupted image hash mismatch for {(image_id, corruption, level)}: {got} != {expected}")
            t = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)))
            return norm(t), self.labels[i]
    return Corrupted()
