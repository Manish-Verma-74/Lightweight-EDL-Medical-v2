"""Build the PAD-UFES-20 and CIFAR-100 selected-image manifests and print their SHA-256 hashes.

PAD-UFES-20: download the ORIGINAL Mendeley Data release (version 1) once, extract it, and pass the folder.
CIFAR-100: downloaded through torchvision into --cifar-root.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edl.cli import DEFAULT_CONFIG, REPO  # noqa: E402
from edl.config import load_config, resolve  # noqa: E402
from edl.data.ood import build_cifar100_manifest, build_pad_manifest, save_ood_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pad-root", help="extracted PAD-UFES-20 folder")
    ap.add_argument("--cifar-root", default="/content/data/cifar100")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.pad_root:
        df = build_pad_manifest(args.pad_root)
        h = save_ood_manifest(df, resolve(cfg["ood"]["pad_ufes_20"]["manifest"], REPO))
        print(f"PAD-UFES-20: {len(df)} images, {df['patient_id'].nunique()} patients, SHA-256 {h}")
    df = build_cifar100_manifest(args.cifar_root)
    h = save_ood_manifest(df, resolve(cfg["ood"]["cifar100"]["manifest"], REPO))
    print(f"CIFAR-100 test: {len(df)} images, SHA-256 {h}")
    print("-> record the hashes in configs/primary.yaml (ood.*.sha256) and commit the manifests")


if __name__ == "__main__":
    main()
