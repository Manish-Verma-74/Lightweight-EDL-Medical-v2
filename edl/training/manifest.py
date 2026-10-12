"""Run manifest and bookkeeping that need no torch: manifest build/load/filter, per-run status and
per-worker logs. The global 144-row manifest is immutable; workers only filter it locally."""
import csv
import datetime
import os

import pandas as pd

from edl.config import run_id as make_run_id
from edl.constants import AUGMENTATIONS, BACKBONES, FRAMEWORKS, SEEDS
from edl.utils.hashing import sha256_file
from edl.utils.io import read_json, write_json_atomic


class RunnerError(RuntimeError):
    pass


MANIFEST_COLUMNS = ["run_id", "backbone", "framework", "augmentation", "seed"]
WORKER_LOG_FIELDS = [
    "run_id",
    "backbone",
    "framework",
    "augmentation",
    "seed",
    "state",
    "stop_reason",
    "last_epoch",
    "best_epoch",
    "best_val_macro_f1",
    "best_val_accuracy",
    "best_val_ece",
    "best_val_macro_cwece",
    "best_val_brier",
    "best_val_aurc_native",
    "best_val_aurc_msp",
    "best_val_aurc_entropy",
    "seconds_this_session",
    "gpu_model",
    "code_commit",
    "finished_at",
]


# ----------------------------------------------------------------------------- manifest
def build_run_manifest() -> pd.DataFrame:
    rows = [{"run_id": make_run_id(b, f, a, s), "backbone": b, "framework": f, "augmentation": a, "seed": s}
            for b in BACKBONES for f in FRAMEWORKS for a in AUGMENTATIONS for s in SEEDS]
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(df) != 144 or not df["run_id"].is_unique:
        raise RunnerError("the run manifest must contain exactly 144 unique runs")
    return df


def load_run_manifest(path: str, expected_sha256: str) -> pd.DataFrame:
    """Load the immutable global manifest: hash check, then structure check."""
    if not expected_sha256 or str(expected_sha256).startswith("TO_FILL"):
        raise RunnerError("the run-manifest hash is not recorded in the config yet")
    got = sha256_file(path, normalize_eol=True)
    if got != expected_sha256:
        raise RunnerError(f"run manifest hash mismatch: got {got}, expected {expected_sha256}")
    df = pd.read_csv(path)
    if list(df.columns) != MANIFEST_COLUMNS:
        raise RunnerError(f"run manifest columns must be {MANIFEST_COLUMNS}; the manifest is immutable "
                          "(no status column)")
    if len(df) != 144 or not df["run_id"].is_unique:
        raise RunnerError("run manifest must have 144 unique run IDs")
    expected = build_run_manifest()
    if not df.equals(expected):
        raise RunnerError("run manifest differs from the factorial design")
    return df


def filter_runs(df: pd.DataFrame, backbone=None, framework=None, augmentation=None, seed=None,
                run_ids=None, shard=None, pilot_configs=None, pilot_seed=None) -> pd.DataFrame:
    """Local filtering of the global manifest. The manifest itself is never edited."""
    out = df
    if backbone:
        out = out[out["backbone"] == backbone]
    if framework:
        out = out[out["framework"] == framework]
    if augmentation:
        out = out[out["augmentation"] == augmentation]
    if seed is not None:
        out = out[out["seed"] == int(seed)]
    if run_ids:
        out = out[out["run_id"].isin(list(run_ids))]
    if pilot_configs is not None:
        keep = {(c["backbone"], c["framework"], c["augmentation"]) for c in pilot_configs}
        out = out[[(r.backbone, r.framework, r.augmentation) in keep for r in out.itertuples()]]
        out = out[out["seed"] == int(pilot_seed)]
    if shard:
        i, n = shard
        out = out.iloc[[k for k in range(len(out)) if k % n == i]]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- status / logs
class RunStatus:
    """status.json inside the run's own Drive folder (single writer: the worker that owns the run)."""

    def __init__(self, drive_dir: str):
        self.path = os.path.join(drive_dir, "status.json")

    def read(self) -> dict:
        return read_json(self.path) if os.path.exists(self.path) else {}

    def update(self, **fields) -> dict:
        st = self.read()
        st.update(fields)
        st["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json_atomic(st, self.path)
        return st


def append_worker_log(path: str, row: dict) -> None:
    """Append to this worker's own CSV. A header mismatch is an error, never a silent new file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != WORKER_LOG_FIELDS:
            raise RunnerError(f"worker log schema mismatch in {path}: {header} != {WORKER_LOG_FIELDS}")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WORKER_LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in WORKER_LOG_FIELDS})
        f.flush()
        os.fsync(f.fileno())


