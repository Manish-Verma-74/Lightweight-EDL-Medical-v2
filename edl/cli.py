"""Shared setup for the command-line scripts (config, verified manifests, environment checks)."""
import os
import sys

from edl.config import load_config, resolve
from edl.data.split import load_verified_manifest, read_metadata
from edl.training.manifest import load_run_manifest
from edl.utils.env import git_commit, training_source_fingerprint

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_CONFIG = os.path.join(REPO, "configs", "primary.yaml")


def add_common_args(ap):
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Lightweight-EDL-Medical-v2",
                    help="persistent root on Drive (runs, logs, results)")
    ap.add_argument("--work-root", default="/content/work", help="local scratch for checkpoints being written")
    ap.add_argument("--data-root", default="/content/data/ham10000", help="LOCAL folder with HAM10000 images")


def load_protocol(args):
    """Load the config and the two hash-verified manifests. Returns (cfg, split_df, run_df, split_sha, commit)."""
    cfg = load_config(args.config)
    split_path = resolve(cfg["data"]["split"]["manifest"], REPO)
    split_sha = cfg["data"]["split"]["sha256"]
    meta_path = os.path.join(args.data_root, "HAM10000_metadata.csv")
    meta = read_metadata(meta_path) if os.path.exists(meta_path) else None
    split_df = load_verified_manifest(split_path, split_sha, meta=meta)
    run_df = load_run_manifest(resolve(cfg["design"]["run_manifest"], REPO), cfg["design"]["run_manifest_sha256"])
    return cfg, split_df, run_df, split_sha, git_commit(REPO)


def expected_run_identity(run_id: str, cfg: dict, split_sha: str, commit: str, phase: str,
                          debug_subset: bool = False) -> dict:
    return {"run_id": run_id, "protocol_version": cfg["protocol_version"],
            "config_hash": __import__("edl.config", fromlist=["config_hash"]).config_hash(cfg),
            "split_sha256": split_sha, "code_commit": commit, "phase": phase,
            "debug_subset": bool(debug_subset), "source_fingerprint": training_source_fingerprint(REPO)}


def check_primary_environment():
    from edl.utils.env import check_runtime_environment
    path = os.path.join(REPO, "environment", "colab_recorded.txt")
    return check_runtime_environment({}, path, require_exact=True)


def check_gpu(require: str):
    """Abort if the GPU model does not contain the required text (e.g. 'T4')."""
    import torch
    if not require:
        return
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    if require.lower() not in name.lower():
        sys.exit(f"GPU check failed: required '{require}', found '{name}'. "
                 "Reconnect to a matching runtime so that all workers use the same GPU model.")
