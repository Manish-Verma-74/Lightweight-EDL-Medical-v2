import copy

import pytest

from edl.config import ConfigError, config_hash, placeholders, require_frozen, validate_config


def test_primary_config_is_valid(cfg):
    validate_config(cfg)
    assert cfg["design"]["total_runs"] == 144


def test_unresolved_fields_are_reported_and_block_the_primary_study(cfg):
    left = placeholders(cfg)
    assert "data.split.sha256" in left and "design.run_manifest_sha256" not in left
    assert "ood.pad_ufes_20.sha256" in left and "corruption.sha256" in left
    with pytest.raises(ConfigError):
        require_frozen(cfg)


def test_changed_design_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["design"]["seeds"] = [1, 2, 3]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_unexpected_section_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["surprise"] = 1
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_spectral_norm_must_stay_off(cfg):
    bad = copy.deepcopy(cfg)
    bad["frameworks"]["f_edl"]["spectral_norm"] = True
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_only_current_epoch_cap_is_excluded_from_identity_hash(cfg):
    other = copy.deepcopy(cfg)
    other["training"]["max_epochs"] = 85
    assert config_hash(cfg) == config_hash(other)
    # Changing the prespecified resource ceiling is a protocol change, not an operational cap extension.
    other["training"]["absolute_epoch_ceiling"] = 140
    assert config_hash(cfg) != config_hash(other)
    other = copy.deepcopy(cfg)
    other["training"]["learning_rate"] = 0.001
    assert config_hash(cfg) != config_hash(other)
