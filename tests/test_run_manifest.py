import csv

import pytest

from edl.training.manifest import (MANIFEST_COLUMNS, RunnerError, WORKER_LOG_FIELDS, append_worker_log, build_run_manifest,
                                 filter_runs, load_run_manifest)
from edl.utils.hashing import sha256_file


def test_manifest_is_the_full_factorial():
    df = build_run_manifest()
    assert len(df) == 144 and df["run_id"].is_unique
    assert list(df.columns) == MANIFEST_COLUMNS
    assert set(df["framework"]) == {"softmax", "edl", "r_edl", "f_edl"}
    assert set(df["augmentation"]) == {"base_flips", "mixup", "cutmix", "randaugment"}
    assert df.groupby("backbone").size().tolist() == [48, 48, 48]


def test_local_filtering_never_edits_the_global_manifest():
    df = build_run_manifest()
    before = df.copy()
    eff = filter_runs(df, backbone="efficientnet_b0")
    assert len(eff) == 48 and set(eff["backbone"]) == {"efficientnet_b0"}
    assert df.equals(before)
    parts = [filter_runs(df, shard=(i, 3)) for i in range(3)]
    ids = [set(p["run_id"]) for p in parts]
    assert sum(len(i) for i in ids) == 144 and set().union(*ids) == set(df["run_id"])


def test_pilot_filter_returns_the_five_configs(cfg):
    df = build_run_manifest()
    p = filter_runs(df, pilot_configs=cfg["pilot"]["configs"], pilot_seed=cfg["pilot"]["seed"])
    assert len(p) == 5 and set(p["seed"]) == {42}


def test_manifest_loading_checks_hash_columns_and_design(tmp_path):
    path = tmp_path / "run_manifest.csv"
    build_run_manifest().to_csv(path, index=False, lineterminator="\n")
    load_run_manifest(str(path), sha256_file(path))
    with pytest.raises(RunnerError):
        load_run_manifest(str(path), "0" * 64)
    with pytest.raises(RunnerError):
        load_run_manifest(str(path), "TO_FILL_AFTER_MANIFEST_GENERATION")
    df = build_run_manifest()
    df["status"] = "pending"                                   # a mutable status column is not allowed
    df.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(RunnerError):
        load_run_manifest(str(path), sha256_file(path))


def test_worker_log_schema_mismatch_is_an_error_not_a_new_file(tmp_path):
    log = tmp_path / "logs_a.csv"
    append_worker_log(str(log), {"run_id": "r1", "state": "completed"})
    append_worker_log(str(log), {"run_id": "r2", "state": "failed"})
    rows = list(csv.DictReader(open(log)))
    assert [r["run_id"] for r in rows] == ["r1", "r2"] and list(rows[0].keys()) == WORKER_LOG_FIELDS
    bad = tmp_path / "logs_b.csv"
    bad.write_text("run_id,other\n1,2\n")
    with pytest.raises(RunnerError):
        append_worker_log(str(bad), {"run_id": "r3"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["logs_a.csv", "logs_b.csv"]   # no silent *_v2 file


def test_repository_run_manifest_matches_frozen_config_hash():
    from pathlib import Path
    import yaml
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "primary.yaml").read_text(encoding="utf-8"))
    path = repo / cfg["design"]["run_manifest"]
    assert path.is_file()
    got = sha256_file(str(path), normalize_eol=True)
    assert got == cfg["design"]["run_manifest_sha256"]
    loaded = load_run_manifest(str(path), got)
    assert len(loaded) == 144 and loaded["run_id"].is_unique
