import numpy as np
import pytest

from edl.augment import cutmix_box_and_weight


def painted_area(h, w, box):
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return int(mask.sum())


def test_weight_comes_from_the_actual_clipped_area():
    rng = np.random.default_rng(0)
    for _ in range(500):
        h, w = int(rng.integers(20, 240)), int(rng.integers(20, 240))
        lam, cx, cy = float(rng.random()), int(rng.integers(0, w)), int(rng.integers(0, h))
        box, weight = cutmix_box_and_weight(h, w, lam, cx, cy)
        assert 0 <= box[0] <= box[2] <= w and 0 <= box[1] <= box[3] <= h
        assert weight == pytest.approx(1.0 - painted_area(h, w, box) / (h * w))
        assert 0.0 <= weight <= 1.0


def test_limiting_cases():
    _, weight = cutmix_box_and_weight(224, 224, 1.0, 100, 100)     # lam = 1: nothing is pasted
    assert weight == 1.0
    box, weight = cutmix_box_and_weight(224, 224, 0.0, 112, 112)   # lam = 0: (almost) the whole image
    assert weight < 0.02 and box[0] == 0 and box[2] == 224


def test_edge_clipping_reduces_the_pasted_area():
    interior_box, interior_w = cutmix_box_and_weight(224, 224, 0.5, 112, 112)
    corner_box, corner_w = cutmix_box_and_weight(224, 224, 0.5, 0, 0)
    assert painted_area(224, 224, corner_box) < painted_area(224, 224, interior_box)
    assert corner_w > interior_w            # more of the original image survives
