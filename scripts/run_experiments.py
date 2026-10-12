"""Run (or resume) training runs from the immutable global 144-run manifest.

Every worker loads the SAME manifest and filters it locally, e.g.
    Worker 1: python scripts/run_experiments.py --phase primary --backbone efficientnet_b0 --worker-tag efficientnet
    Worker 2: python scripts/run_experiments.py --phase primary --backbone mobilenet_v3_small --worker-tag mobilenet
    Worker 3: python scripts/run_experiments.py --phase primary --backbone shufflenet_v2 --worker-tag shufflenet
Each run owns <drive-root>/<phase>/runs/<run_id>/; each worker owns its own log CSV.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from edl.cli import add_common_args, check_gpu, check_primary_environment, load_protocol  # noqa: E402
from edl.config import require_frozen  # noqa: E402
from edl.data.ham10000 import debug_subset, find_image_paths, verify_images  # noqa: E402
from edl.training.engine import RunSpec  # noqa: E402
from edl.training.manifest import filter_runs  # noqa: E402
from edl.training.runner import RunContext, execute_run  # noqa: E402

# Short covering set for smoke tests: every backbone, framework and augmentation appears at least once.
SMOKE_COVERING = [("efficientnet_b0", "softmax", "base_flips"), ("mobilenet_v3_small", "edl", "mixup"),
                  ("shufflenet_v2", "r_edl", "cutmix"), ("efficientnet_b0", "f_edl", "randaugment"),
                  ("mobilenet_v3_small", "f_edl", "cutmix"), ("shufflenet_v2", "softmax", "mixup")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--phase", required=True, choices=["smoke", "pilot", "primary"])
    ap.add_argument("--backbone"), ap.add_argument("--framework"), ap.add_argument("--augmentation")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-ids", help="comma-separated run IDs")
    ap.add_argument("--shard", help="i/n: take every n-th run starting at i (optional load balancing)")
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--max-epochs", type=int, help="smoke/pilot diagnostic cap only; primary cap comes from the committed config")
    ap.add_argument("--debug-per-class", type=int, default=10, help="smoke only: images per class and partition")
    ap.add_argument("--worker-tag", help="name of this worker; its log is worker_logs/logs_<tag>_<phase>.csv")
    ap.add_argument("--require-gpu", default="", help="abort unless the GPU name contains this text, e.g. T4")
    ap.add_argument("--finalize-cap", action="store_true", help="deprecated; cap-limited runs stay cap_limited and test evaluation is always deferred")
    ap.add_argument("--allow-commit-change", action="store_true", help="allow resuming across code commits (not for primary)")
    args = ap.parse_args()

    cfg, split_df, run_df, split_sha, commit = load_protocol(args)
    check_gpu(args.require_gpu)
    if args.phase == "primary":
        require_frozen(cfg)
        if args.allow_commit_change:
            sys.exit("--allow-commit-change is not permitted for the primary study")
        if args.max_epochs is not None:
            sys.exit("primary max_epochs must come only from the committed configs/primary.yaml; update the same config on all workers")
        if args.finalize_cap:
            sys.exit("--finalize-cap is not supported; cap-limited runs remain cap_limited and test evaluation is deferred")
        try:
            check_primary_environment()
        except Exception as exc:
            sys.exit(str(exc))
    elif args.finalize_cap:
        sys.exit("--finalize-cap is not needed in smoke/pilot; runs that hit the cap remain cap_limited")
    max_epochs = args.max_epochs if args.max_epochs is not None else {"smoke": 3, "pilot": cfg["pilot"]["initial_ceiling"],
                                     "primary": cfg["training"]["max_epochs"]}[args.phase]
    if args.phase == "primary" and max_epochs is None:
        sys.exit("training.max_epochs is not set")
    ceiling = cfg["training"]["absolute_epoch_ceiling"]
    if args.phase == "primary" and ceiling is not None and max_epochs > ceiling:
        sys.exit(f"max epochs {max_epochs} exceeds the absolute ceiling {ceiling}")

    shard = tuple(int(v) for v in args.shard.split("/")) if args.shard else None
    runs = filter_runs(run_df, args.backbone, args.framework, args.augmentation, args.seed,
                       args.run_ids.split(",") if args.run_ids else None, shard,
                       cfg["pilot"]["configs"] if args.phase == "pilot" else None, cfg["pilot"]["seed"])
    if args.phase == "smoke" and not (args.backbone or args.framework or args.augmentation or args.run_ids):
        keep = set(SMOKE_COVERING)
        runs = runs[[(r.backbone, r.framework, r.augmentation) in keep and r.seed == 42 for r in runs.itertuples()]]
    if args.max_runs:
        runs = runs.iloc[:args.max_runs]
    if runs.empty:
        sys.exit("no runs selected")

    paths = find_image_paths(args.data_root)
    verify_images(split_df, paths)
    manifest = debug_subset(split_df, args.debug_per_class) if args.phase == "smoke" else split_df
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.worker_tag
    ctx = RunContext(cfg=cfg, manifest=manifest, paths=paths,
                     work_root=os.path.join(args.work_root, args.phase, "runs"),
                     out_root=os.path.join(args.drive_root, args.phase, "runs"),
                     device=device, phase=args.phase, max_epochs=max_epochs, split_sha256=split_sha, code_commit=commit,
                     debug_subset=args.phase == "smoke", finalize_cap=args.finalize_cap,
                     allow_commit_change=args.allow_commit_change,
                     worker_log=os.path.join(args.drive_root, "worker_logs", f"logs_{tag}_{args.phase}.csv") if tag else None)
    print(f"phase={args.phase} | runs={len(runs)} | max_epochs={max_epochs} | device={device} | commit={commit[:8]}")
    states = []
    for r in runs.itertuples():
        spec = RunSpec(
            r.run_id,
            r.backbone,
            r.framework,
            r.augmentation,
            int(r.seed),
        )
        st = execute_run(spec, ctx)
        states.append((
            r.run_id,
            st.get("state"),
            st.get("stop_reason"),
            st.get("best_epoch"),
            st.get("best_val_accuracy"),
            st.get("best_val_macro_f1"),
            st.get("best_val_ece"),
            st.get("best_val_macro_cwece"),
            st.get("best_val_brier"),
            st.get("best_val_aurc_native"),
            st.get("best_val_aurc_msp"),
            st.get("best_val_aurc_entropy"),
        ))
    print(
        "\nrun_id | state | stop_reason | best_epoch | "
        "best_val_accuracy | best_val_macro_f1 | best_val_ece | "
        "best_val_macro_cwece | best_val_brier | "
        "best_val_aurc_native | best_val_aurc_msp | "
        "best_val_aurc_entropy"
    )
    for row in states:
        print(" | ".join(str(v) for v in row))


if __name__ == "__main__":
    main()
