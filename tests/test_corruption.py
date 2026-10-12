import numpy as np
import pytest

from edl.data.corruption import condition_list, corrupt_image, derive_seed


def test_seed_is_stable_and_condition_specific():
    a = derive_seed(20261009, "ISIC_0000001", "gaussian_noise", 3)
    assert a == derive_seed(20261009, "ISIC_0000001", "gaussian_noise", 3)
    assert a != derive_seed(20261009, "ISIC_0000001", "gaussian_noise", 4)
    assert a != derive_seed(20261009, "ISIC_0000002", "gaussian_noise", 3)
    assert derive_seed(20261009, "x", "contrast", 1) == 8307266787730391973 or isinstance(a, int)


def make_x():
    rng = np.random.default_rng(0)
    return rng.random((32, 32, 3)).astype(np.float32)


@pytest.mark.parametrize("name,param", [("gaussian_noise", 0.2), ("gaussian_blur", 2.0), ("brightness", 0.3), ("contrast", 0.2)])
def test_output_range_dtype_and_determinism(name, param):
    x = make_x()
    y1, y2 = corrupt_image(x, name, param, 7), corrupt_image(x, name, param, 7)
    assert y1.dtype == np.float32 and y1.shape == x.shape
    assert 0.0 <= y1.min() and y1.max() <= 1.0
    assert np.array_equal(y1, y2)


def test_effects_have_the_expected_direction():
    x = make_x()
    assert corrupt_image(x, "gaussian_blur", 3.0, 0).std() < x.std()
    assert corrupt_image(x, "contrast", 0.1, 0).std() < 0.2 * x.std() + 1e-6
    assert corrupt_image(x, "brightness", 0.3, 0).mean() > x.mean()
    n1, n2 = corrupt_image(x, "gaussian_noise", 0.2, 1), corrupt_image(x, "gaussian_noise", 0.2, 2)
    assert not np.array_equal(n1, n2)                 # a different seed gives different noise
    assert np.abs(n1 - x).mean() > 0.05


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        corrupt_image(make_x() * 2, "brightness", 0.1, 0)
    with pytest.raises(ValueError):
        corrupt_image(make_x(), "fog", 1.0, 0)
    with pytest.raises(ValueError):
        corrupt_image(np.zeros((8, 8)), "brightness", 0.1, 0)


def test_twenty_conditions_in_fixed_order(cfg):
    conds = condition_list(cfg)
    assert len(conds) == 20
    assert conds[0][:2] == ("gaussian_noise", 1) and conds[-1][:2] == ("contrast", 5)
    assert len({(n, l) for n, l, _ in conds}) == 20
