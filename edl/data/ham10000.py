"""HAM10000 image index, transforms, dataset, sampler and loaders.

Images are read from LOCAL disk (e.g. /content/data/ham10000), never from mounted Drive.
Missing or unreadable images raise errors; nothing is silently dropped.
"""
import os

import numpy as np
import pandas as pd

from edl.constants import CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


class DataError(RuntimeError):
    """Raised when images are missing, duplicated or unreadable."""


def find_image_paths(data_root: str) -> dict:
    """Map image_id -> file path for every .jpg under data_root (Kaggle layout: two image folders)."""
    paths = {}
    for folder, _, files in os.walk(data_root):
        for fname in files:
            if fname.lower().endswith(".jpg"):
                image_id = os.path.splitext(fname)[0]
                if image_id in paths:
                    raise DataError(f"duplicate image file for {image_id}: {paths[image_id]} and {folder}")
                paths[image_id] = os.path.join(folder, fname)
    return paths


def verify_images(manifest: pd.DataFrame, paths: dict, check_readable: bool = False) -> None:
    """Every manifest image must exist; optionally decode each one once."""
    missing = [i for i in manifest["image_id"] if i not in paths]
    if missing:
        raise DataError(f"{len(missing)} manifest image(s) not found on disk, e.g. {missing[:5]}")
    if check_readable:
        from PIL import Image
        for image_id in manifest["image_id"]:
            try:
                with Image.open(paths[image_id]) as im:
                    im.convert("RGB").load()
            except Exception as e:  # noqa: BLE001
                raise DataError(f"unreadable image {image_id}: {e}") from e


def build_train_transform(augmentation: str, aug_cfg: dict):
    """The common CPU pipeline is direct resize + horizontal/vertical flips for every condition.
    RandAugment adds two operations after those common transforms; Mixup/CutMix are batch-level GPU ops."""
    import torchvision.transforms as T
    base = aug_cfg["base_flips"]
    steps = [T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
             T.RandomHorizontalFlip(p=base["horizontal_flip_p"]),
             T.RandomVerticalFlip(p=base["vertical_flip_p"])]
    if augmentation == "randaugment":
        ra = aug_cfg["randaugment"]
        steps.append(T.RandAugment(num_ops=ra["num_ops"], magnitude=ra["magnitude"],
                                  interpolation=T.InterpolationMode.BILINEAR))
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def build_eval_transform():
    """Deterministic preprocessing used for validation, test, OOD and corruption inputs."""
    import torchvision.transforms as T
    return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BILINEAR),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def make_dataset(manifest: pd.DataFrame, partition: str, paths: dict, transform):
    """Dataset over one partition, matched by image_id. __getitem__ returns (tensor, label)."""
    from PIL import Image
    from torch.utils.data import Dataset

    sub = manifest[manifest["split"] == partition].reset_index(drop=True)
    if len(sub) == 0:
        raise DataError(f"partition '{partition}' is empty")

    class HAM10000Partition(Dataset):
        def __init__(self):
            self.partition = partition
            self.image_ids = sub["image_id"].tolist()
            self.lesion_ids = sub["lesion_id"].tolist()
            self.labels = sub["dx"].map(CLASS_TO_IDX).astype(int).tolist()

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, i):
            path = paths[self.image_ids[i]]
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
                    img.load()
            except Exception as e:  # noqa: BLE001
                raise DataError(f"cannot read {path}: {e}") from e
            return transform(img), self.labels[i]

    return HAM10000Partition()


def class_weights_sampler(labels, seed: int):
    """Inverse-class-frequency WeightedRandomSampler with its own seeded generator (training only)."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=len(CLASS_TO_IDX)).astype(np.float64)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(labels), replacement=True, generator=gen)
    return sampler, gen


def make_loader(dataset, batch_size: int, loader_cfg: dict, sampler=None, seed: int = 0, drop_last: bool = False):
    """DataLoader with the frozen num_workers/pin_memory settings and a seeded generator."""
    import torch
    from torch.utils.data import DataLoader
    from edl.utils.seed import worker_init_fn

    n_workers = int(loader_cfg["num_workers"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
        num_workers=n_workers, pin_memory=bool(loader_cfg["pin_memory"]) and torch.cuda.is_available(),
        persistent_workers=bool(loader_cfg["persistent_workers"]) and n_workers > 0,
        worker_init_fn=worker_init_fn if n_workers > 0 else None,
        generator=gen, drop_last=drop_last,
    )
    return loader, gen


def debug_subset(manifest: pd.DataFrame, per_class: int, seed: int = 0) -> pd.DataFrame:
    """Small deterministic subset for SMOKE TESTS only (never used for the primary study)."""
    rng = np.random.default_rng(seed)
    parts = []
    for (_, _), g in manifest.groupby(["split", "dx"]):
        take = min(per_class, len(g))
        parts.append(g.iloc[np.sort(rng.choice(len(g), size=take, replace=False))])
    return pd.concat(parts).reset_index(drop=True)
