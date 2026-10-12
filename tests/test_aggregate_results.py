import pandas as pd
import pytest

from scripts.aggregate_results import aggregate_corruption_curves


def test_corruption_aggregation_excludes_clean_reference_and_averages_20_conditions():
    rows = [{"run_id": "run-a", "condition": "clean:0", "ece_delta": 0.0, "macro_f1_delta": 0.0}]
    for i in range(1, 21):
        rows.append({"run_id": "run-a", "condition": f"corruption:{i}",
                     "ece_delta": float(i), "macro_f1_delta": -float(i) / 100})
    out = aggregate_corruption_curves(pd.DataFrame(rows)).iloc[0]
    assert out["corruption_mean_ece_delta"] == pytest.approx(10.5)
    assert out["corruption_mean_macro_f1_delta"] == pytest.approx(-0.105)


def test_corruption_aggregation_rejects_incomplete_condition_groups():
    rows = [{"run_id": "run-a", "condition": "clean:0", "ece_delta": 0.0, "macro_f1_delta": 0.0}]
    for i in range(1, 20):
        rows.append({"run_id": "run-a", "condition": f"corruption:{i}",
                     "ece_delta": float(i), "macro_f1_delta": -float(i) / 100})
    with pytest.raises(ValueError, match="exactly 20"):
        aggregate_corruption_curves(pd.DataFrame(rows))


def test_corruption_aggregation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_corruption_curves(pd.DataFrame([{"run_id": "run-a"}]))
