"""Generate (or check) the immutable 144-run manifest and print its SHA-256."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from edl.cli import REPO  # noqa: E402
from edl.training.manifest import build_run_manifest  # noqa: E402
from edl.utils.hashing import sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "protocol", "run_manifest_v1.csv"))
    ap.add_argument("--check", action="store_true", help="only compare against the existing file")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "primary.yaml"))
    args = ap.parse_args()
    from edl.config import load_config
    cfg = load_config(args.config)
    df = build_run_manifest()
    if args.check:
        import pandas as pd
        if not os.path.exists(args.out):
            print(f"manifest does not exist: {args.out}")
            raise SystemExit(1)
        actual = pd.read_csv(args.out)
        semantic_ok = actual.equals(df)
        digest = sha256_file(args.out, normalize_eol=True)
        expected_hash = cfg["design"]["run_manifest_sha256"]
        hash_ok = bool(expected_hash and not str(expected_hash).startswith("TO_FILL") and digest == expected_hash)
        print("matches the factorial design:", semantic_ok)
        print("SHA-256:", digest)
        print("matches configured SHA-256:", hash_ok)
        if not semantic_ok or not hash_ok:
            raise SystemExit(1)
        return
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(len(df), "runs written to", args.out)
    print("SHA-256:", sha256_file(args.out))
    print("-> make sure configs/primary.yaml design.run_manifest_sha256 matches this value; --check verifies it")


if __name__ == "__main__":
    main()
