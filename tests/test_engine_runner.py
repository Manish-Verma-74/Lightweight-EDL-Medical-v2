"""End-to-end tests on a tiny synthetic dataset (CPU, a few seconds each)."""
import os
import csv
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from edl.training.checkpoint import BEST, LATEST, PersistenceError  # noqa: E402
from edl.training.engine import RunSpec, train_run  # noqa: E402
from edl.training.manifest import RunStatus, WORKER_LOG_FIELDS  # noqa: E402
from edl.training.runner import RunContext, execute_run  # noqa: E402
from edl.utils.io import read_json  # noqa: E402

IDENT = {"run_id": "mobilenet_v3_small__softmax__base_flips__seed42", "protocol_version": "v1", "config_hash": "c",
         "split_sha256": "s", "code_commit": "test", "phase": "smoke", "debug_subset": False}
SPEC = RunSpec(IDENT["run_id"], "mobilenet_v3_small", "softmax", "base_flips", 42)
DEVICE = torch.device("cpu")


def run(spec, cfg, manifest, paths, tmp, max_epochs, sub="a", **kw):
    return train_run(spec, cfg, manifest, paths, str(tmp / "work" / sub), str(tmp / "drive" / sub),
                     dict(IDENT, run_id=spec.run_id), DEVICE, max_epochs, log=lambda *_: None, **kw)


def test_training_never_touches_test_images(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham
    no_test = {i: p for i, p in paths.items() if i in set(manifest.loc[manifest["split"] != "test", "image_id"])}
    res = run(SPEC, tiny_cfg, manifest, no_test, tmp_path, 2)       # would raise KeyError if a test image were read
    assert res["last_epoch"] == 2 and res["stop_reason"] == "cap"
    assert sorted(os.listdir(tmp_path / "work" / "a")) == sorted([BEST, LATEST, "epochs.csv"])


def test_interrupted_run_resumes_to_the_same_stopping_point(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham
    full = run(SPEC, tiny_cfg, manifest, paths, tmp_path, 12, sub="full")
    part = run(SPEC, tiny_cfg, manifest, paths, tmp_path, 2, sub="resumed")
    assert part["stop_reason"] == "cap" and part["last_epoch"] == 2
    for f in os.listdir(tmp_path / "work" / "resumed"):                 # a fresh runtime: local disk is lost
        os.remove(tmp_path / "work" / "resumed" / f)
    done = run(SPEC, tiny_cfg, manifest, paths, tmp_path, 12, sub="resumed")
    assert done["last_epoch"] == full["last_epoch"]
    assert done["best_epoch"] == full["best_epoch"] and done["stop_reason"] == full["stop_reason"]
    assert [h["epoch"] for h in done["history"]] == list(range(1, done["last_epoch"] + 1))
    assert done["history"][:2] == part["history"]                       # the pre-interruption epochs are kept verbatim


def test_early_stopping_uses_patience_and_earliest_best_epoch(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham
    res = run(SPEC, tiny_cfg, manifest, paths, tmp_path, 40)
    if res["stop_reason"] == "early_stop":
        assert res["last_epoch"] - res["best_epoch"] == tiny_cfg["training"]["patience"]
    assert 1 <= res["best_epoch"] <= res["last_epoch"]
    f1s = [h["val_macro_f1"] for h in res["history"]]
    assert res["best_epoch"] == f1s.index(max(f1s)) + 1               # ties keep the earliest epoch


def test_persistence_failure_stops_the_run_but_a_transient_one_is_retried(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham

    def always(attempt):
        raise OSError("drive down")

    with pytest.raises(PersistenceError):
        run(SPEC, tiny_cfg, manifest, paths, tmp_path, 1, sub="down", sync_fail_hook=always)

    seen = []

    def once(attempt):
        seen.append(attempt)
        if len(seen) == 1:
            raise OSError("blip")

    res = run(SPEC, tiny_cfg, manifest, paths, tmp_path, 1, sub="blip", sync_fail_hook=once)
    assert res["last_epoch"] == 1 and len(seen) >= 2


def make_ctx(manifest, paths, cfg, tmp, **kw):
    return RunContext(cfg=cfg, manifest=manifest, paths=paths, work_root=str(tmp / "work"), out_root=str(tmp / "drive"),
                      device=DEVICE, phase="smoke", max_epochs=2, split_sha256="s", code_commit="test",
                      worker_log=str(tmp / "logs_w.csv"), log=lambda *_: None, **kw)


def test_execute_run_cap_limited_then_skipped_without_test_leakage(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham
    spec = RunSpec("mobilenet_v3_small__softmax__base_flips__seed42", "mobilenet_v3_small", "softmax", "base_flips", 42)
    d = tmp_path / "drive" / spec.run_id

    st = execute_run(spec, make_ctx(manifest, paths, tiny_cfg, tmp_path))
    assert st["state"] == "cap_limited" and not (d / "predictions_id.npz").exists()
    assert not (d / "test_metrics.json").exists()

    # Verify summary metrics match the epoch selected by validation Macro-F1.
    summary_path = d / "summary.json"
    assert summary_path.exists(), "summary.json was not generated"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    with (d / "epochs.csv").open(
        "r", newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    best_epoch = int(summary["best_epoch"])

    best_row = next(
        (row for row in rows if int(row["epoch"]) == best_epoch),
        None,
    )
    assert best_row is not None, (
        f"No epochs.csv row found for best epoch {best_epoch}"
    )

    assert float(summary["best_val_accuracy"]) == pytest.approx(
        float(best_row["val_accuracy"])
    )
    assert float(summary["best_val_ece"]) == pytest.approx(
        float(best_row["val_ece"])
    )

    st = execute_run(spec, make_ctx(manifest, paths, tiny_cfg, tmp_path))  # same cap: no rerun
    assert st["state"] == "cap_limited"
    # The deprecated finalize-cap flag must not turn a cap-limited run into a completed/test-evaluated run.
    st = execute_run(spec, make_ctx(manifest, paths, tiny_cfg, tmp_path, finalize_cap=True))
    assert st["state"] == "cap_limited" and RunStatus(str(d)).read()["state"] == "cap_limited"
    for name in ["summary.json", "status.json", "env.json", "epochs.csv", BEST, LATEST]:
        assert (d / name).exists(), name
    assert not (d / "predictions_id.npz").exists() and not (d / "test_metrics.json").exists()
    lines = (tmp_path / "logs_w.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == WORKER_LOG_FIELDS and len(lines) == 2  # header + one real execution


def test_a_failing_run_is_recorded_and_does_not_raise(synthetic_ham, tiny_cfg, tmp_path):
    manifest, paths = synthetic_ham
    broken = dict(paths)
    for image_id in manifest.loc[manifest["split"] == "train", "image_id"]:
        broken[image_id] = str(tmp_path / "does_not_exist.jpg")      # every training image unreadable
    spec = RunSpec("mobilenet_v3_small__softmax__base_flips__seed42", "mobilenet_v3_small", "softmax", "base_flips", 42)
    st = execute_run(spec, make_ctx(manifest, broken, tiny_cfg, tmp_path))
    assert st["state"] == "failed" and "cannot read" in st["error"]
    assert (tmp_path / "logs_w.csv").read_text().strip().splitlines()[1].split(",")[5] == "failed"
