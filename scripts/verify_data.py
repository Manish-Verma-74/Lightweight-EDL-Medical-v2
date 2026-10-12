"""Check HAM10000 on local disk against the frozen split manifest (every image present and decodable)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edl.cli import REPO, add_common_args  # noqa: E402
from edl.config import load_config, resolve  # noqa: E402
from edl.data.ham10000 import find_image_paths, verify_images  # noqa: E402
from edl.data.split import load_verified_manifest, read_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--no-decode", action="store_true", help="skip decoding every image (faster)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = read_metadata(os.path.join(args.data_root, "HAM10000_metadata.csv"))
    df = load_verified_manifest(resolve(cfg["data"]["split"]["manifest"], REPO), cfg["data"]["split"]["sha256"], meta)
    verify_images(df, find_image_paths(args.data_root), check_readable=not args.no_decode)
    print("OK:", len(df), "images verified;", df.groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
