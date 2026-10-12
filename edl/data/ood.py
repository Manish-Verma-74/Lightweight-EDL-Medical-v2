"""OOD datasets: PAD-UFES-20 (near-OOD / domain shift) and CIFAR-100 test set (far-OOD).

Both are described by a manifest (stable ids, labels, cluster ids, SHA-256 per file) that is generated
once, hashed and then only loaded. A missing file, hash mismatch or duplicate id raises an error.
"""
import os

import numpy as np
import pandas as pd

from edl.utils.hashing import sha256_bytes, sha256_file

PAD_COLUMNS = ["img_id", "patient_id", "lesion_id", "diagnostic", "rel_path", "sha256"]
CIFAR_COLUMNS = ["img_id", "label", "sha256"]


class OODDataError(RuntimeError):
    pass


def build_pad_manifest(root: str) -> pd.DataFrame:
    """Scan an extracted PAD-UFES-20 release. Selection rule: every image listed in metadata.csv whose file exists;
    a listed but missing image is an error. Images are sorted by img_id for a stable order."""
    meta_path = None
    for folder, _, files in os.walk(root):
        if "metadata.csv" in files:
            meta_path = os.path.join(folder, "metadata.csv")
            break
    if meta_path is None:
        raise OODDataError(f"metadata.csv not found under {root}")
    meta = pd.read_csv(meta_path)
    need = ["patient_id", "lesion_id", "img_id", "diagnostic"]
    if any(c not in meta.columns for c in need):
        raise OODDataError(f"metadata.csv must contain {need}; found {list(meta.columns)}")
    if meta["img_id"].isna().any() or meta[["patient_id", "lesion_id", "diagnostic"]].isna().any().any():
        raise OODDataError("PAD-UFES-20 metadata contains missing image/patient/lesion/diagnosis identifiers")
    if meta["img_id"].duplicated().any():
        raise OODDataError("duplicate img_id in PAD-UFES-20 metadata")
    files = {}
    for folder, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                if f in files:
                    raise OODDataError(f"duplicate image basename {f!r} across PAD-UFES-20 folders")
                files[f] = os.path.join(folder, f)
    rows, missing = [], []
    for r in meta.sort_values("img_id").itertuples():
        if r.img_id not in files:
            missing.append(r.img_id)
            continue
        rows.append({"img_id": r.img_id, "patient_id": r.patient_id, "lesion_id": r.lesion_id,
                     "diagnostic": r.diagnostic, "rel_path": os.path.relpath(files[r.img_id], root).replace("\\", "/"),
                     "sha256": sha256_file(files[r.img_id])})
    if missing:
        raise OODDataError(f"{len(missing)} PAD-UFES-20 image(s) listed in metadata but not found, e.g. {missing[:5]}")
    return pd.DataFrame(rows, columns=PAD_COLUMNS)


def build_cifar100_manifest(root: str) -> pd.DataFrame:
    """Manifest of the official CIFAR-100 test set (10,000 images). Downloads through torchvision if needed."""
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=root, train=False, download=True)
    if len(ds) != 10000:
        raise OODDataError(f"CIFAR-100 test set must have 10000 images, found {len(ds)}")
    rows = [{"img_id": f"cifar100_test_{i:05d}", "label": int(ds.targets[i]),
             "sha256": sha256_bytes(np.ascontiguousarray(ds.data[i]).tobytes())} for i in range(len(ds))]
    return pd.DataFrame(rows, columns=CIFAR_COLUMNS)


def save_ood_manifest(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def load_ood_manifest(path: str, expected_sha256: str, columns) -> pd.DataFrame:
    if not expected_sha256 or str(expected_sha256).startswith("TO_FILL"):
        raise OODDataError("OOD manifest hash is not recorded in the config yet")
    got = sha256_file(path, normalize_eol=True)
    if got != expected_sha256:
        raise OODDataError(f"OOD manifest hash mismatch: got {got}, expected {expected_sha256}")
    df = pd.read_csv(path)
    if list(df.columns) != list(columns):
        raise OODDataError(f"unexpected manifest columns {list(df.columns)}")
    if df.empty or df["img_id"].isna().any() or df["img_id"].duplicated().any():
        raise OODDataError("OOD manifest is empty or contains null/duplicate img_id values")
    if df["sha256"].isna().any():
        raise OODDataError("OOD manifest contains missing image hashes")
    if columns == PAD_COLUMNS and df[["patient_id", "lesion_id", "diagnostic", "rel_path"]].isna().any().any():
        raise OODDataError("PAD-UFES-20 manifest contains missing required identifiers")
    if columns == CIFAR_COLUMNS and len(df) != 10000:
        raise OODDataError(f"CIFAR-100 official test manifest must contain 10000 rows, found {len(df)}")
    return df


def pad_dataset(manifest: pd.DataFrame, root: str, transform, verify_hashes: bool = True):
    """PAD-UFES-20 dataset in manifest order. Returns (img, 0); the label is unused for OOD scoring."""
    from PIL import Image
    from torch.utils.data import Dataset

    paths = [os.path.join(root, p) for p in manifest["rel_path"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise OODDataError(f"{len(missing)} PAD-UFES-20 file(s) missing, e.g. {missing[:3]}")
    if verify_hashes:
        for p, h in zip(paths, manifest["sha256"]):
            if sha256_file(p) != h:
                raise OODDataError(f"hash mismatch for {p}")

    class PadUfes(Dataset):
        image_ids = manifest["img_id"].tolist()
        cluster_ids = manifest["patient_id"].tolist()

        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                return transform(im.convert("RGB")), 0

    return PadUfes()


def cifar100_dataset(manifest: pd.DataFrame, root: str, transform):
    """Official CIFAR-100 test images, checked against the manifest hashes, in manifest order."""
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.datasets import CIFAR100

    ds = CIFAR100(root=root, train=False, download=True)
    if len(manifest) != 10000 or len(ds) != 10000:
        raise OODDataError(f"CIFAR-100 manifest and official test set must both contain 10000 images; "
                           f"manifest={len(manifest)}, dataset={len(ds)}")
    expected_ids = [f"cifar100_test_{i:05d}" for i in range(10000)]
    if manifest["img_id"].tolist() != expected_ids:
        raise OODDataError("CIFAR-100 manifest image IDs/order do not match the official test-set ordering")
    for i in range(len(ds)):
        if sha256_bytes(np.ascontiguousarray(ds.data[i]).tobytes()) != manifest["sha256"].iloc[i]:
            raise OODDataError(f"CIFAR-100 image {i} does not match the manifest hash")

    class Cifar(Dataset):
        image_ids = manifest["img_id"].tolist()

        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return transform(Image.fromarray(ds.data[i])), 0

    return Cifar()
