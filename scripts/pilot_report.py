"""Validate all five pilot runs, then decide whether to extend the cap or calculate E_max."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from edl.cli import add_common_args, load_protocol  # noqa: E402
from edl.training.pilot import compute_epoch_cap, next_cap, pilot_needs_extension  # noqa: E402
from edl.training.manifest import filter_runs  # noqa: E402
from edl.utils.io import read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ceiling", type=int, default=None, help="current diagnostic ceiling (default: initial ceiling)")
    args = ap.parse_args()
    cfg, _, run_df, _, _ = load_protocol(args)
    p = cfg["pilot"]
    ceiling = p["initial_ceiling"] if args.ceiling is None else args.ceiling
    if ceiling < 1: raise SystemExit("diagnostic ceiling must be positive")
    if p["resource_ceiling"] is not None and ceiling > p["resource_ceiling"]:
        raise SystemExit("diagnostic ceiling exceeds the prespecified resource ceiling")
    runs = filter_runs(run_df, pilot_configs=p["configs"], pilot_seed=p["seed"])
    if len(runs) != 5: raise SystemExit(f"protocol error: expected five pilot runs, manifest selected {len(runs)}")
    best, need_extension, incomplete = [], False, False
    for r in runs.itertuples():
        sp = os.path.join(args.drive_root, "pilot", "runs", r.run_id, "status.json")
        if not os.path.exists(sp):
            print(f"{r.run_id}: missing status.json")
            incomplete = True
            continue
        st = read_json(sp)
        state, stop = st.get("state"), st.get("stop_reason")
        be, le = st.get("best_epoch"), st.get("last_epoch")
        valid = state in ("completed", "cap_limited") and stop in ("early_stop", "cap")                 and isinstance(be, int) and be >= 1 and isinstance(le, int) and le >= be
        if not valid:
            print(f"{r.run_id}: invalid/incomplete status: state={state}, stop={stop}, best_epoch={be}, last_epoch={le}")
            incomplete = True
            continue
        if stop == "early_stop" and state != "completed":
            print(f"{r.run_id}: contradiction: early_stop but state={state}")
            incomplete = True
            continue
        if stop == "cap" and (state != "cap_limited" or le < ceiling):
            print(f"{r.run_id}: cap-limited state does not match ceiling {ceiling} (last_epoch={le})")
            incomplete = True
            continue
        ext = pilot_needs_extension(stop, be, ceiling)
        print(f"{r.run_id}: state={state} stop={stop} best_epoch={be} last_epoch={le} extension_needed={ext}")
        best.append(be)
        need_extension = need_extension or ext
    if incomplete or len(best) != 5:
        print("\nPilot report is incomplete/invalid. Do not calculate E_max.")
        raise SystemExit(1)
    if need_extension:
        new = next_cap(ceiling, p["extension_step"], p["resource_ceiling"])
        if new == ceiling:
            print(f"\nResource ceiling {ceiling} reached while a pilot remains cap-limited. Investigate convergence; do not freeze the cap.")
            raise SystemExit(2)
        print(f"\nExtension rule triggered: raise the diagnostic ceiling to {new}; resume only cap-limited pilots.")
    else:
        cap = compute_epoch_cap(best, p["patience"], p["safety_margin"])
        if p["resource_ceiling"] is not None and cap > p["resource_ceiling"]:
            print(f"\nCalculated E_max={cap} exceeds the resource ceiling {p['resource_ceiling']}; resolve before freeze.")
            raise SystemExit(2)
        print(f"\nNo extension needed. Global cap E_max = max({best}) + {p['patience']} + {p['safety_margin']} = {cap}")

if __name__ == "__main__": main()
