import pytest

from edl.losses.schedule import annealing_coefficient
from edl.training.pilot import compute_epoch_cap, next_cap, pilot_needs_extension


def test_annealing_is_one_based():
    assert annealing_coefficient(1, 10) == pytest.approx(0.1)
    assert annealing_coefficient(5, 10) == pytest.approx(0.5)
    assert annealing_coefficient(10, 10) == 1.0
    assert annealing_coefficient(11, 10) == 1.0 and annealing_coefficient(80, 10) == 1.0
    with pytest.raises(ValueError):
        annealing_coefficient(0, 10)


def test_epoch_cap_formula_matches_the_protocol_example():
    assert compute_epoch_cap([40, 62, 55, 48, 51], patience=8, margin=15) == 85   # 62 + 23


def test_pilot_extension_rule():
    assert pilot_needs_extension("cap", 50, 100)                # reached the ceiling
    assert pilot_needs_extension("early_stop", 95, 100)         # best epoch in the last ten epochs
    assert not pilot_needs_extension("early_stop", 60, 100)
    assert not pilot_needs_extension("early_stop", 90, 100)     # exactly ten epochs before: not within the last ten


def test_next_cap_respects_the_ceiling():
    assert next_cap(100, 20, None) == 120
    assert next_cap(100, 20, 110) == 110
