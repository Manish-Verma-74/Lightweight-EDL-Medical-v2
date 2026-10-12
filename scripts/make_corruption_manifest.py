"""Generate the corruption manifest (one row per test image x corrupted condition) and an example grid.

Run after the split manifest is frozen. It records the derived seed and the SHA-256 of every corrupted
image so the identical corrupted inputs can be verified in any session.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edl.cli import REPO, add_common_args, load_protocol  # noqa: E402
from edl.config import resolve  # noqa: E402
from edl.data.corruption import condition_list, corrupt_image, derive_seed, load_resized  # noqa: E402
from edl.data.ham10000 import find_image_paths, verify_images  # noqa: E402
from edl.utils.hashing import sha256_bytes, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--grid-out", default=os.path.join(REPO, "protocol", "corruption_example_grid.png"))
    args = ap.parse_args()
    cfg, split_df, _, _, _ = load_protocol(args)
    test = split_df[split_df["split"] == "test"].reset_index(drop=True)
    paths = find_image_paths(args.data_root)
    verify_images(test, paths)
    g = cfg["corruption"]["global_seed"]
    conds = condition_list(cfg)

    rows = []
    for r in test.itertuples():
        x = load_resized(paths[r.image_id])
        src_hash = sha256_file(paths[r.image_id])
        for name, level, param in conds:
            seed = derive_seed(g, r.image_id, name, level)
            y = corrupt_image(x, name, param, seed)
            rows.append({"image_id": r.image_id, "corruption": name, "severity": level, "param": param, "seed": seed,
                         "source_sha256": src_hash, "output_sha256": sha256_bytes(np.ascontiguousarray(y).tobytes())})
    out = resolve(cfg["corruption"]["manifest"], REPO)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(len(rows), "rows;", len(conds), "conditions x", len(test), "images")
    print("corruption manifest SHA-256:", sha256_file(out))

    # example grid for the first test image (inspect before freezing the severity values)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = load_resized(paths[test["image_id"].iloc[0]])
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for row, name in enumerate(["gaussian_noise", "gaussian_blur", "brightness", "contrast"]):
        axes[row, 0].imshow(x)
        axes[row, 0].set_title("clean", fontsize=8)
        for lvl in range(1, 6):
            param = cfg["corruption"]["types"][name]["severities"][lvl - 1]
            seed = derive_seed(g, test["image_id"].iloc[0], name, lvl)
            axes[row, lvl].imshow(corrupt_image(x, name, param, seed))
            axes[row, lvl].set_title(f"{name} {lvl}\n({param})", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.grid_out, dpi=110)
    print("example grid saved to", args.grid_out)
    print("-> record the manifest hash in configs/primary.yaml (corruption.sha256)")


if __name__ == "__main__":
    main()
