from scripts.evaluate_runs import _ood_score_map


def test_ood_score_map_includes_fedl_secondary_score_when_both_bundles_have_it():
    id_arrays = {"test_native_uncertainty": [], "test_msp_uncertainty": [],
                 "test_entropy": [], "test_total_uncertainty": []}
    ood_arrays = {"native_uncertainty": [], "msp_uncertainty": [], "entropy": [],
                  "total_uncertainty": []}
    scores = _ood_score_map(id_arrays, ood_arrays)
    assert scores["total_uncertainty"] == ("test_total_uncertainty", "total_uncertainty")
    assert {"native", "msp", "entropy"}.issubset(scores)


def test_ood_score_map_does_not_require_fedl_secondary_score_for_other_frameworks():
    id_arrays = {"test_native_uncertainty": [], "test_msp_uncertainty": [], "test_entropy": []}
    ood_arrays = {"native_uncertainty": [], "msp_uncertainty": [], "entropy": []}
    scores = _ood_score_map(id_arrays, ood_arrays)
    assert set(scores) == {"native", "msp", "entropy"}
