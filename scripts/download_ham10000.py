"""Download HAM10000 from Kaggle to LOCAL disk (never read images from Drive).

Credentials (any one):
  - environment variable KAGGLE_API_TOKEN, or
  - KAGGLE_USERNAME and KAGGLE_KEY, or
  - ~/.kaggle/kaggle.json
In Colab, set the token in a cell with getpass() (see notebooks/colab_runner.ipynb); never commit it.
"""
import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

SLUG = "kmader/skin-cancer-mnist-ham10000"


def counts(dest):
    meta = os.path.join(dest, "HAM10000_metadata.csv")
    n_meta = len(pd.read_csv(meta)) if os.path.exists(meta) else 0
    n_jpg = len(glob.glob(os.path.join(dest, "**", "*.jpg"), recursive=True))
    return n_meta, n_jpg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/content/data/ham10000")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    n_meta, n_jpg = counts(args.dest)
    if n_meta == 10015 and n_jpg == 10015:
        print("HAM10000 already present:", n_meta, "metadata rows,", n_jpg, "images")
        return
    have_cred = ("KAGGLE_API_TOKEN" in os.environ or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)
                 or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")))
    if not have_cred:
        sys.exit("No Kaggle credentials found. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY).")
    subprocess.run(["kaggle", "datasets", "download", "-d", SLUG, "-p", args.dest, "--unzip"], check=True)
    n_meta, n_jpg = counts(args.dest)
    print("metadata rows:", n_meta, "| images:", n_jpg)
    if n_meta != 10015 or n_jpg != 10015:
        sys.exit("Unexpected HAM10000 contents (expected 10015 metadata rows and 10015 images).")


if __name__ == "__main__":
    main()
