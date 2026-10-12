"""Manifest-driven runner. No held-out test inference occurs during training/smoke/pilot.
Each worker filters the same immutable 144-row global manifest and writes its own log CSV.
"""
import datetime
import os
import traceback
from dataclasses import dataclass, field

from edl.config import config_hash
from edl.training.checkpoint import check_identity, sync_file
from edl.training.engine import RunSpec, train_run
from edl.training.manifest import RunStatus, append_worker_log
from edl.utils.env import collect_env, training_source_fingerprint
from edl.utils.io import write_json_atomic
from edl.constants import CLASS_NAMES

@dataclass
class RunContext:
    cfg: dict
    manifest: object
    paths: dict
    work_root: str
    out_root: str
    device: object
    phase: str
    max_epochs: int
    split_sha256: str
    code_commit: str
    debug_subset: bool = False
    finalize_cap: bool = False
    allow_commit_change: bool = False
    worker_log: str = None
    sync_fail_hook: object = None
    log: object = field(default=print)


def make_identity(spec: RunSpec, ctx: RunContext) -> dict:
    return {"run_id": spec.run_id, "protocol_version": ctx.cfg["protocol_version"],
            "config_hash": config_hash(ctx.cfg), "split_sha256": ctx.split_sha256,
            "code_commit": ctx.code_commit, "phase": ctx.phase, "debug_subset": bool(ctx.debug_subset),
            "source_fingerprint": training_source_fingerprint(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))}


def _persist_small(work_dir: str, drive_dir: str, names) -> None:
    for n in names:
        p = os.path.join(work_dir, n)
        if os.path.exists(p): sync_file(p, drive_dir)


def execute_run(spec: RunSpec, ctx: RunContext) -> dict:
    work_dir = os.path.join(ctx.work_root, spec.run_id)
    drive_dir = os.path.join(ctx.out_root, spec.run_id)
    os.makedirs(work_dir, exist_ok=True); os.makedirs(drive_dir, exist_ok=True)
    status = RunStatus(drive_dir)
    st = status.read()
    identity = make_identity(spec, ctx)
    # Identity must be checked BEFORE completed/cap-limited shortcuts; stale runs may never be silently skipped.
    if st.get("identity"):
        check_identity(st["identity"], identity, allow_commit_change=ctx.allow_commit_change)
    elif st.get("state") in {"completed", "cap_limited"}:
        raise RuntimeError(f"{spec.run_id}: existing status has no identity; do not reuse it")
    if st.get("state") == "completed":
        ctx.log(f"[{spec.run_id}] already completed with matching identity - skipping")
        return st
    if st.get("state") == "cap_limited" and ctx.max_epochs <= int(st.get("last_epoch", 0)):
        ctx.log(f"[{spec.run_id}] cap-limited at epoch {st['last_epoch']}; raise the frozen global cap")
        return st

    env = collect_env(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    write_json_atomic(env, os.path.join(work_dir, "env.json"))
    status.update(state="running", attempts=st.get("attempts", 0) + 1, identity=identity,
                  gpu_model=env.get("gpu_model"), last_attempt_commit=ctx.code_commit)
    try:
        result = train_run(spec, ctx.cfg, ctx.manifest, ctx.paths, work_dir, drive_dir, identity, ctx.device,
                           ctx.max_epochs, ctx.allow_commit_change, ctx.sync_fail_hook, ctx.log)
        
        # Find the validation metrics from the exact epoch selected by Macro-F1.
        history = result.get("history", [])
        best_epoch = int(result["best_epoch"])

        best_row = next(
            (
                row
                for row in history
                if int(row.get("epoch", -1)) == best_epoch
            ),
            None,
        )

        # Fail explicitly if the selected epoch's metrics are unavailable.
        if best_row is None:
            raise RuntimeError(
                f"{spec.run_id}: no history row for best epoch {best_epoch}"
            )

        required_metrics = (
            "val_accuracy",
            "val_macro_f1",
            "val_ece",
            "val_macro_cwece",
            "val_brier",
            "val_aurc_native",
            "val_aurc_msp",
            "val_aurc_entropy",
        )
        missing_metrics = [
            key for key in required_metrics
            if key not in best_row
        ]
        missing_metrics.extend(
            f"val_recall_{name}"
            for name in CLASS_NAMES
            if f"val_recall_{name}" not in best_row
        )
        missing_metrics.extend(
            f"val_support_{name}"
            for name in CLASS_NAMES
            if f"val_support_{name}" not in best_row
        )
        if missing_metrics:
            raise RuntimeError(
                f"{spec.run_id}: missing best-epoch metrics: "
                f"{missing_metrics}"
            )

        summary = {
            k: v
            for k, v in result.items()
            if k not in ("history", "best_path", "latest_path")
        }
        summary.update({
            "best_val_accuracy": float(best_row["val_accuracy"]),
            "best_val_macro_f1": float(best_row["val_macro_f1"]),
            "best_val_ece": float(best_row["val_ece"]),
            "best_val_macro_cwece": float(best_row["val_macro_cwece"]),
            "best_val_brier": float(best_row["val_brier"]),
            "best_val_aurc": {
                "native": float(best_row["val_aurc_native"]),
                "msp": float(best_row["val_aurc_msp"]),
                "entropy": float(best_row["val_aurc_entropy"]),
            },
            "best_val_per_class_recall": {
                name: (
                    float(best_row[f"val_recall_{name}"])
                    if best_row[f"val_recall_{name}"] not in (None, "")
                    else None
                )
                for name in CLASS_NAMES
            },
            "best_val_support": {
                name: int(best_row[f"val_support_{name}"])
                for name in CLASS_NAMES
            },
            "identity": identity,
            "env": env,
            "test_evaluation_deferred": True,
        })

        write_json_atomic(summary, os.path.join(work_dir, "summary.json"))
        _persist_small(work_dir, drive_dir, ["summary.json", "env.json", "epochs.csv"])
        state = "completed" if result["stop_reason"] == "early_stop" else "cap_limited"
        ceiling = ctx.cfg["training"].get("absolute_epoch_ceiling")
        at_ceiling = bool(ctx.phase == "primary" and state == "cap_limited" and ceiling is not None
                          and int(ctx.max_epochs) == int(ceiling) and int(result["last_epoch"]) >= int(ceiling))
        st = status.update(
            state=state,
            last_epoch=result["last_epoch"],
            best_epoch=result["best_epoch"],
            best_val_macro_f1=summary["best_val_macro_f1"],
            best_val_accuracy=summary["best_val_accuracy"],
            best_val_ece=summary["best_val_ece"],
            best_val_macro_cwece=summary["best_val_macro_cwece"],
            best_val_brier=summary["best_val_brier"],
            best_val_aurc_native=summary["best_val_aurc"]["native"],
            best_val_aurc_msp=summary["best_val_aurc"]["msp"],
            best_val_aurc_entropy=summary["best_val_aurc"]["entropy"],
            best_val_per_class_recall=summary["best_val_per_class_recall"],
            best_val_support=summary["best_val_support"],
            stop_reason=result["stop_reason"],
            cap_limited_at_resource_ceiling=at_ceiling,
            test_exported=False,
            error=None,
        )
    except KeyboardInterrupt:
        status.update(state="running", note="interrupted; resume by running again")
        raise
    except Exception as e:  # noqa: BLE001 - record failure and continue; never swap seeds
        st = status.update(state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        ctx.log(f"[{spec.run_id}] FAILED: {e}")
        if ctx.worker_log:
            append_worker_log(ctx.worker_log, {**spec.__dict__, "state": "failed", "gpu_model": env.get("gpu_model"),
                                               "code_commit": ctx.code_commit,
                                               "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        return st

    if ctx.worker_log:
        append_worker_log(
            ctx.worker_log,
            {
                **spec.__dict__,
                "state": st["state"],
                "stop_reason": st.get("stop_reason"),
                "last_epoch": st.get("last_epoch"),
                "best_epoch": st.get("best_epoch"),
                "best_val_macro_f1": st.get("best_val_macro_f1"),
                "best_val_accuracy": st.get("best_val_accuracy"),
                "best_val_ece": st.get("best_val_ece"),
                "best_val_macro_cwece": st.get("best_val_macro_cwece"),
                "best_val_brier": st.get("best_val_brier"),
                "best_val_aurc_native": st.get("best_val_aurc_native"),
                "best_val_aurc_msp": st.get("best_val_aurc_msp"),
                "best_val_aurc_entropy": st.get("best_val_aurc_entropy"),
                "seconds_this_session": result["seconds_this_session"],
                "gpu_model": env.get("gpu_model"),
                "code_commit": ctx.code_commit,
                "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    return st
