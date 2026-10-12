import os

import pytest

torch = pytest.importorskip("torch")

from edl.training.checkpoint import (BEST, INDEX, LATEST, CheckpointError, PersistenceError, RunCheckpointer,  # noqa: E402
                                     atomic_torch_save, check_identity, load_checkpoint, sync_file)

IDENT = {"run_id": "r1", "protocol_version": "v1", "config_hash": "c", "split_sha256": "s", "code_commit": "abc",
         "phase": "smoke", "debug_subset": False}


def state(epoch):
    best_epoch = min(int(epoch), 2)
    best_model = {"w": torch.ones(2) * best_epoch}
    return {"model": {"w": torch.ones(2) * epoch}, "optimizer": {}, "best_f1": 0.5,
            "best_epoch": best_epoch, "best_model_state": best_model, "best_extra": {}, "no_improve": 1,
            "history": [], "stop_reason": None, "rng": {}, "aug_rng": {}, "sampler_gen": None, "loader_gen": None}


def make(tmp_path, **kw):
    return RunCheckpointer(str(tmp_path / "local"), str(tmp_path / "drive"), IDENT, backoff=0.0, **kw)


def test_atomic_save_leaves_no_temp_file(tmp_path):
    p = str(tmp_path / "x.pth")
    atomic_torch_save({"a": 1}, p)
    assert load_checkpoint(p) == {"a": 1} and os.listdir(tmp_path) == ["x.pth"]


def test_sync_retries_then_succeeds_or_raises(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"data")
    calls = []

    def flaky(attempt):
        calls.append(attempt)
        if attempt < 3:
            raise OSError("drive hiccup")

    sync_file(str(src), str(tmp_path / "d"), retries=3, backoff=0.0, fail_hook=flaky)
    assert calls == [1, 2, 3] and (tmp_path / "d" / "f.bin").read_bytes() == b"data"

    def always(attempt):
        raise OSError("down")

    with pytest.raises(PersistenceError):
        sync_file(str(src), str(tmp_path / "d2"), retries=2, backoff=0.0, fail_hook=always)
    assert not os.path.exists(tmp_path / "d2" / "f.bin")
    assert not [n for n in os.listdir(tmp_path / "d2") if n.startswith(".tmp_")]


def test_only_two_checkpoint_roles_exist_and_drive_index_is_written(tmp_path):
    ck = make(tmp_path)
    for epoch in range(1, 5):
        if epoch == 2:
            ck.save_best({"w": torch.ones(2)}, 2, 0.6)
        ck.save_latest(state(epoch), epoch)
    assert sorted(os.listdir(tmp_path / "local")) == [BEST, LATEST]
    assert sorted(os.listdir(tmp_path / "drive")) == sorted([BEST, INDEX, LATEST])
    assert load_checkpoint(str(tmp_path / "drive" / LATEST))["epoch"] == 4


def test_resume_restores_from_drive_when_local_disk_is_gone(tmp_path):
    ck = make(tmp_path)
    ck.save_best({"w": torch.ones(2)}, 2, 0.6)
    ck.save_latest(state(3), 3)
    for f in os.listdir(tmp_path / "local"):
        os.remove(tmp_path / "local" / f)                      # a new Colab runtime
    fresh = make(tmp_path)
    got = fresh.load_resume()
    assert got["epoch"] == 3 and got["_source"] == "drive"
    assert os.path.exists(tmp_path / "local" / LATEST) and os.path.exists(tmp_path / "local" / BEST)


def test_a_half_written_or_tampered_drive_copy_is_not_trusted(tmp_path):
    ck = make(tmp_path)
    ck.save_latest(state(3), 3)
    os.remove(tmp_path / "local" / LATEST)
    with open(tmp_path / "drive" / LATEST, "ab") as f:
        f.write(b"garbage")                                     # hash no longer matches persist_index.json
    assert make(tmp_path).load_resume() is None


def test_cadence_and_force(tmp_path):
    ck = make(tmp_path, sync_every=2)
    assert ck.save_latest(state(1), 1) is False
    assert not os.path.exists(tmp_path / "drive" / LATEST)
    assert ck.save_latest(state(2), 2) is True and os.path.exists(tmp_path / "drive" / LATEST)
    assert ck.save_latest(state(3), 3, force=True) is True
    assert load_checkpoint(str(tmp_path / "drive" / LATEST))["epoch"] == 3


def test_newest_valid_copy_wins(tmp_path):
    ck = make(tmp_path, sync_every=2)
    ck.save_latest(state(2), 2)         # persisted
    ck.save_latest(state(3), 3)         # local only (cadence)
    got = make(tmp_path, sync_every=2).load_resume()
    assert got["epoch"] == 3 and got["_source"] == "local"


def test_identity_mismatch_refuses_to_resume(tmp_path):
    ck = make(tmp_path)
    ck.save_latest(state(1), 1)
    other = RunCheckpointer(str(tmp_path / "local"), str(tmp_path / "drive"), dict(IDENT, config_hash="DIFFERENT"))
    with pytest.raises(CheckpointError):
        other.load_resume()
    newer_code = RunCheckpointer(str(tmp_path / "local"), str(tmp_path / "drive"), dict(IDENT, code_commit="def"))
    with pytest.raises(CheckpointError):
        newer_code.load_resume()
    assert newer_code.load_resume(allow_commit_change=True)["epoch"] == 1


def test_check_identity_names_the_field():
    with pytest.raises(CheckpointError, match="split_sha256"):
        check_identity(IDENT, dict(IDENT, split_sha256="other"))


def test_drive_latest_failure_keeps_previous_resume_and_matching_best(tmp_path):
    calls = {"n": 0}

    enabled = {"value": False}
    def fail_latest_after_best(attempt):
        if not enabled["value"]:
            return
        calls["n"] += 1
        # first new copy is BEST and succeeds; the LATEST write then fails on every retry.
        if calls["n"] >= 2:
            raise OSError("simulated disconnect while persisting latest")

    ck = make(tmp_path, retries=2, fail_hook=fail_latest_after_best)
    ck.save_best({"w": torch.ones(2) * 1}, 1, 0.4)
    ck.save_latest(state(1), 1, force=True)  # establishes a valid old/latest pair
    enabled["value"] = True
    # Change the best and try to publish a later resumable state; BEST reaches Drive, LATEST fails.
    ck.save_best({"w": torch.ones(2) * 2}, 2, 0.6)
    before_calls = calls["n"]
    with pytest.raises(PersistenceError):
        ck.save_latest(state(2), 2, force=True)
    assert calls["n"] > before_calls
    # Simulate a new Colab runtime: local storage is lost, so the persisted Drive LATEST is authoritative.
    for name in os.listdir(tmp_path / "local"):
        os.remove(tmp_path / "local" / name)
    fresh = make(tmp_path)
    got = fresh.load_resume()
    assert got is not None and got["epoch"] == 1
    best = load_checkpoint(str(tmp_path / "local" / BEST))
    assert best["epoch"] == 1
    assert torch.equal(best["model"]["w"], torch.ones(2))


def test_index_update_failure_recovers_previous_checkpoint_pair(tmp_path, monkeypatch):
    import edl.training.checkpoint as module

    ck = make(tmp_path)
    ck.save_best({"w": torch.ones(2)}, 1, 0.4)
    first = state(1)
    first["best_epoch"] = 1
    first["best_model_state"] = {"w": torch.ones(2)}
    ck.save_latest(first, 1, force=True)

    later = state(2)
    later["best_epoch"] = 1
    later["best_model_state"] = {"w": torch.ones(2)}
    original_write = module.write_json_atomic

    def fail_index_write(value, path):
        if str(path).endswith(INDEX):
            raise OSError("simulated failure committing persist index")
        return original_write(value, path)

    monkeypatch.setattr(module, "write_json_atomic", fail_index_write)
    with pytest.raises(OSError, match="persist index"):
        ck.save_latest(later, 2, force=True)
    # Simulate loss of local Colab disk and return to the last complete checkpoint on Drive.
    for name in os.listdir(tmp_path / "local"):
        os.remove(tmp_path / "local" / name)
    monkeypatch.setattr(module, "write_json_atomic", original_write)
    fresh = make(tmp_path)
    got = fresh.load_resume()
    assert got is not None and got["epoch"] == 1
    assert load_checkpoint(str(tmp_path / "drive" / LATEST))["epoch"] == 1
    assert not os.path.exists(tmp_path / "drive" / f".txn_backup_{LATEST}")
