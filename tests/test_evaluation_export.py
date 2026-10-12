import numpy as np
import pytest

from edl.evaluation.export import verify_id_bundle
from edl.evaluation.predict import save_bundle
from edl.constants import NUM_CLASSES


def _bundle_arrays(n_val=3, n_test=4):
    arrays = {}
    for part, n in (("val", n_val), ("test", n_test)):
        arrays[f"{part}_probs"] = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float32)
        arrays[f"{part}_labels"] = np.arange(n, dtype=np.int64) % NUM_CLASSES
        arrays[f"{part}_image_ids"] = np.array([f"{part}-{i}" for i in range(n)])
        arrays[f"{part}_native_uncertainty"] = np.full(n, 0.5, dtype=np.float32)
        arrays[f"{part}_msp_uncertainty"] = np.full(n, 1.0 - 1.0 / NUM_CLASSES, dtype=np.float32)
        arrays[f"{part}_entropy"] = np.full(n, 1.0, dtype=np.float32)
    return arrays


def test_verify_id_bundle_checks_probability_shapes_ids_and_identity(tmp_path):
    path = tmp_path / "predictions_id.npz"
    arrays = _bundle_arrays()
    identity = {"run_id": "run-a", "config_hash": "abc"}
    save_bundle(str(path), arrays, {"identity": identity})
    verify_id_bundle(str(path), 3, 4,
                     expected_val_ids=["val-0", "val-1", "val-2"],
                     expected_test_ids=["test-0", "test-1", "test-2", "test-3"],
                     expected_identity=identity)


def test_verify_id_bundle_rejects_id_order_or_identity_mismatch(tmp_path):
    path = tmp_path / "predictions_id.npz"
    identity = {"run_id": "run-a", "config_hash": "abc"}
    save_bundle(str(path), _bundle_arrays(), {"identity": identity})
    with pytest.raises(ValueError, match="frozen-manifest order"):
        verify_id_bundle(str(path), 3, 4, expected_val_ids=["val-1", "val-0", "val-2"])
    with pytest.raises(ValueError, match="identity"):
        verify_id_bundle(str(path), 3, 4, expected_identity={"run_id": "different"})
