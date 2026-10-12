"""Generate the lesion-grouped split manifest once, verify it and print its SHA-256.

Usage (Colab, after downloading HAM10000):
    python scripts/make_split.py --data-root /content/data/ham10000
Then copy the printed SHA-256 into configs/primary.yaml (data.split.sha256), commit the
manifest and the config, and never regenerate the split.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import sklearn  # noqa: E402

from edl.config import load_config, resolve  # noqa: E402
from edl.data.split import generate_split_manifest, read_metadata, save_manifest, verify_manifest  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder containing HAM10000_metadata.csv")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "primary.yaml"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest (not for the frozen study)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    split_cfg = cfg["data"]["split"]
    out_path = resolve(split_cfg["manifest"], REPO)
    want = split_cfg["generator"]["sklearn_version"]
    if sklearn.__version__ != want:
        print(f"WARNING: scikit-learn {sklearn.__version__} differs from the pinned {want}; fold assignment may differ.")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} already exists. The split is frozen; use --force only for a deliberate new version.")

    meta = read_metadata(os.path.join(args.data_root, "HAM10000_metadata.csv"))
    df = generate_split_manifest(meta, split_cfg)
    summary = verify_manifest(df, meta, split_cfg["regression_expectations"])
    digest = save_manifest(df, out_path)

    print("counts:", summary["counts"], "| images:", summary["images"], "| lesions:", summary["lesions"])
    for part, cc in summary["class_counts"].items():
        print(f"  {part}: {cc}")
    if summary["regression_mismatches"]:
        print("REGRESSION MISMATCHES (investigate; do not force):", summary["regression_mismatches"])
    else:
        print("counts match the regression expectations")
    ref = split_cfg["regression_expectations"].get("reference_sha256")
    print("SHA-256:", digest)
    print("matches reference hash:", digest == ref)
    print(f"-> put this hash in configs/primary.yaml (data.split.sha256) and commit {out_path}")


if __name__ == "__main__":
    main()
