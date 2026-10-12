"""Benchmark selected seed-42/base-flips checkpoints on one fixed machine and join quality metrics."""
import argparse
import json
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402
from edl.cli import add_common_args, check_gpu, check_primary_environment, expected_run_identity, load_protocol  # noqa: E402
from edl.config import require_frozen  # noqa: E402
from edl.constants import BACKBONES, FRAMEWORKS, NUM_CLASSES  # noqa: E402
from edl.evaluation.efficiency import benchmark_model, checkpoint_size_bytes, count_macs, count_parameters  # noqa: E402
from edl.frameworks import build_framework  # noqa: E402
from edl.training.checkpoint import BEST, RunCheckpointer, check_identity, load_checkpoint  # noqa: E402
from edl.training.engine import RunSpec, load_best_model  # noqa: E402
from edl.training.manifest import RunStatus  # noqa: E402
from edl.utils.env import collect_env, git_commit  # noqa: E402
from edl.utils.io import read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--require-gpu", default="T4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg, split_df, _, split_sha, commit = load_protocol(args)
    require_frozen(cfg)
    try: check_primary_environment()
    except Exception as exc: raise SystemExit(str(exc))
    check_gpu(args.require_gpu)
    e = cfg["efficiency"]
    torch.set_num_threads(e["cpu_threads"])
    env = collect_env()
    rows = []
    for backbone in BACKBONES:
        for framework_name in FRAMEWORKS:
            run_id = f"{backbone}__{framework_name}__base_flips__seed42"
            drive_dir = os.path.join(args.drive_root, "primary", "runs", run_id)
            ck_path = os.path.join(drive_dir, BEST)
            status = RunStatus(drive_dir).read()
            if status.get("state") != "completed" and not status.get("cap_limited_at_resource_ceiling", False):
                raise RuntimeError(f"{run_id}: run is not eligible for final efficiency benchmark (state={status.get('state')})")
            expected = expected_run_identity(run_id, cfg, split_sha, commit, "primary", False)
            check_identity(status.get("identity", {}), expected, allow_commit_change=True)
            if not os.path.exists(ck_path): raise FileNotFoundError(f"missing selected checkpoint: {ck_path}")
            RunCheckpointer(os.path.join(args.work_root, "primary", "runs", run_id), drive_dir, expected).verify_drive_checkpoint(BEST)
            spec = RunSpec(run_id, backbone, framework_name, "base_flips", 42)
            # Load the selected trained checkpoint. Random initialization is never benchmarked.
            model, ck = load_best_model(spec, cfg, ck_path, torch.device("cpu"), expected)
            metrics_path = os.path.join(drive_dir, "test_metrics.json")
            ood_path = os.path.join(drive_dir, "ood_metrics.json")
            if not os.path.exists(metrics_path) or not os.path.exists(ood_path):
                raise RuntimeError(f"{run_id}: test_metrics.json and ood_metrics.json must exist before efficiency benchmarking")
            metrics = read_json(metrics_path)
            ood = read_json(ood_path)
            check_identity(metrics.get("identity", {}), expected, allow_commit_change=True)
            check_identity(ood.get("identity", {}), expected, allow_commit_change=True)
            pad_native = ood.get("scores", {}).get("pad_ufes20", {}).get("native", {})
            if "auroc" not in pad_native: raise RuntimeError(f"{run_id}: missing PAD-UFES-20 native AUROC")
            fw = build_framework(framework_name, cfg)
            row = {"run_id": run_id, "backbone": backbone, "framework": framework_name,
                   "reference_seed": 42, "reference_augmentation": "base_flips", "best_epoch": int(ck["epoch"]),
                   **count_parameters(model), "macs": count_macs(model, tuple(e["input_shape"])),
                   "checkpoint_bytes": checkpoint_size_bytes(ck_path),
                   "test_macro_f1": metrics["test"]["macro_f1"], "test_ece": metrics["test"]["ece"],
                   "pad_ufes20_native_auroc": pad_native["auroc"],
                   "gpu_model": env.get("gpu_model"), "torch": env.get("torch"),
                   "torchvision": env.get("torchvision"), "python": env.get("python"), "sklearn": env.get("sklearn")}
            for dev in ["cuda", "cpu"]:
                if dev == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("primary efficiency comparison requires the configured CUDA GPU")
                result = benchmark_model(model, fw, dev, tuple(e["input_shape"]),
                                         e["warmup_iterations"], e["timed_iterations"])
                for kind, stats in result.items():
                    row.update({f"{dev}_{kind}_{k}": v for k, v in stats.items()})
            rows.append(row)
            print(backbone, framework_name, "selected-checkpoint benchmark complete")
    out = args.out or os.path.join(args.drive_root, "results", "efficiency.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)

if __name__ == "__main__": main()
