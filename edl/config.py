"""Load and validate configs/primary.yaml. Unknown or missing settings are errors."""
import copy
import os

from edl.constants import AUGMENTATIONS, BACKBONES, CLASS_NAMES, FRAMEWORKS, SEEDS
from edl.utils.hashing import canonical_json_hash

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from e


class ConfigError(ValueError):
    pass


TOP_LEVEL_KEYS = {
    "protocol_version", "status", "study", "design", "data", "models", "frameworks", "augmentation",
    "training", "data_loader", "pilot", "checkpoint", "evaluation", "ood", "corruption", "efficiency", "statistics",
}
PLACEHOLDER_PREFIX = "TO_FILL"

# Strict nested schema: typos/additional knobs must be made explicit in the reviewed protocol.
NESTED_KEYS = {
    "study": {"title", "type"},
    "design": {"backbones", "frameworks", "augmentations", "seeds", "total_runs", "run_manifest", "run_manifest_sha256", "run_id_pattern"},
    "data": {"dataset", "classes", "kaggle_dataset", "split", "preprocessing"},
    "data.split": {"manifest", "sha256", "generator", "roles", "regression_expectations"},
    "data.split.generator": {"class", "n_splits", "shuffle", "random_state", "group_column", "label_column", "sklearn_version"},
    "data.split.roles": {"train_folds", "validation_fold", "test_fold"},
    "data.split.regression_expectations": {"images", "lesions", "counts", "reference_sha256"},
    "data.split.regression_expectations.counts": {"train", "val", "test"},
    "data.preprocessing": {"input_size", "resize", "interpolation", "normalize", "color"},
    "models": {"num_classes", "pretrained", "weights"},
    "models.weights": {"efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"},
    "frameworks": {"softmax", "edl", "r_edl", "f_edl"},
    "frameworks.softmax": {"loss"},
    "frameworks.edl": {"evidence", "kl_annealing_step", "epoch_indexing", "soft_target_kl"},
    "frameworks.r_edl": {"lambda", "evidence", "kl_annealing_step", "epoch_indexing", "soft_target_kl", "objective_status"},
    "frameworks.f_edl": {"spectral_norm", "objective_status"},
    "augmentation": {"base_flips", "mixup", "cutmix", "randaugment"},
    "augmentation.base_flips": {"horizontal_flip_p", "vertical_flip_p"},
    "augmentation.mixup": {"alpha"}, "augmentation.cutmix": {"alpha"},
    "augmentation.randaugment": {"num_ops", "magnitude"},
    "training": {"optimizer", "learning_rate", "weight_decay", "batch_size", "scheduler", "precision", "sampler", "selection_metric", "tie_break", "patience", "max_epochs", "epoch_extension_step", "absolute_epoch_ceiling"},
    "data_loader": {"num_workers", "pin_memory", "persistent_workers", "non_blocking_transfer", "data_location"},
    "checkpoint": {"roles", "sync_every_n_epochs", "max_sync_retries", "retry_backoff_seconds"},
    "evaluation": {"ece_bins", "cw_ece_bins", "entropy_log_base", "ood_positive_class", "fpr_recall_target", "melanoma"},
    "evaluation.melanoma": {"specificity_targets", "coverage_targets", "thresholds_from"},
    "ood": {"id_reference", "pad_ufes_20", "cifar100"},
    "ood.pad_ufes_20": {"source", "manifest", "sha256", "description", "cluster_column"},
    "ood.cifar100": {"split", "manifest", "sha256", "resize", "interpolation", "limitation"},
    "corruption": {"pipeline", "global_seed", "manifest", "sha256", "types", "subset"},
    "corruption.types": {"gaussian_noise", "gaussian_blur", "brightness", "contrast"},
    "corruption.types.gaussian_noise": {"param", "severities"}, "corruption.types.gaussian_blur": {"param", "severities"},
    "corruption.types.brightness": {"param", "severities"}, "corruption.types.contrast": {"param", "severities"},
    "corruption.subset": {"backbones", "frameworks", "augmentations"},
    "efficiency": {"gpu", "precision", "input_shape", "warmup_iterations", "timed_iterations", "cpu_threads", "reference_checkpoint", "mac_counter"},
    "statistics": {"bootstrap_replicates", "curve_bootstrap_replicates", "bootstrap_rng_seed", "p1_interval_level", "default_interval_level", "resampling_units", "hypotheses"},
    "statistics.resampling_units": {"ham10000", "pad_ufes_20", "cifar100"},
    "statistics.hypotheses": {"P1_cutmix_vs_base_ece", "P2_fedl_native_vs_softmax_msp_pad_ufes20_auroc", "P4_cutmix_vs_base_corruption_ece_degradation"},
    "statistics.hypotheses.P1_cutmix_vs_base_ece": {"softmax", "f_edl", "edl", "r_edl", "note"},
    "pilot": {"seed", "patience", "initial_ceiling", "extension_step", "safety_margin", "resource_ceiling", "configs"},
}


def _validate_nested_keys(value, path=""):
    if isinstance(value, dict):
        if path in NESTED_KEYS:
            expected = NESTED_KEYS[path]
            extra, missing = set(value) - expected, expected - set(value)
            _need(not extra, f"unexpected keys in {path}: {sorted(extra)}")
            _need(not missing, f"missing keys in {path}: {sorted(missing)}")
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            _validate_nested_keys(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            if path == "pilot.configs" and isinstance(child, dict):
                expected = {"backbone", "framework", "augmentation"}
                _need(set(child) == expected, f"pilot.configs[{idx}] keys must be {sorted(expected)}")
            else:
                _validate_nested_keys(child, path)


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg: dict) -> None:
    _need(isinstance(cfg, dict), "config must be a mapping")
    missing = TOP_LEVEL_KEYS - set(cfg)
    extra = set(cfg) - TOP_LEVEL_KEYS
    _need(not missing, f"missing config sections: {sorted(missing)}")
    _need(not extra, f"unexpected config sections: {sorted(extra)}")
    _validate_nested_keys(cfg)

    d = cfg["design"]
    _need(list(d["backbones"]) == BACKBONES, "design.backbones differs from the frozen constants")
    _need(list(d["frameworks"]) == FRAMEWORKS, "design.frameworks differs from the frozen constants")
    _need(list(d["augmentations"]) == AUGMENTATIONS, "design.augmentations differs from the frozen constants")
    _need(list(d["seeds"]) == SEEDS, "design.seeds differs from the frozen constants")
    n = len(d["backbones"]) * len(d["frameworks"]) * len(d["augmentations"]) * len(d["seeds"])
    _need(d["total_runs"] == n == 144, f"design.total_runs must equal {n} (= 144)")

    data = cfg["data"]
    _need(list(data["classes"]) == CLASS_NAMES, "data.classes must follow the frozen class order")
    split = data["split"]
    roles = split["roles"]
    folds = list(roles["train_folds"]) + [roles["validation_fold"], roles["test_fold"]]
    _need(sorted(folds) == list(range(split["generator"]["n_splits"])),
          "split roles must use every fold exactly once")

    t = cfg["training"]
    _need(t["optimizer"] == "adamw" and t["scheduler"] == "none", "optimizer/scheduler differ from the protocol")
    _need(t["learning_rate"] > 0 and t["weight_decay"] >= 0 and t["batch_size"] > 0, "invalid training numbers")
    _need(t["patience"] >= 1, "patience must be >= 1")
    _need(t["max_epochs"] is None or (isinstance(t["max_epochs"], int) and t["max_epochs"] >= 1),
          "max_epochs must be null or a positive integer")
    _need(t["selection_metric"] == "val_macro_f1", "checkpoint selection must be validation macro-F1")

    f = cfg["frameworks"]
    _need(set(f) == set(FRAMEWORKS), "frameworks section must define exactly the four frameworks")
    _need(f["r_edl"]["lambda"] > 0, "r_edl.lambda must be positive")
    _need(f["edl"]["epoch_indexing"] == "one_based" and f["r_edl"]["epoch_indexing"] == "one_based",
          "KL epoch indexing must be one_based")
    _need(f["f_edl"]["spectral_norm"] is False, "F-EDL spectral normalization must stay disabled")

    a = cfg["augmentation"]
    _need(set(a) == set(AUGMENTATIONS), "augmentation section must define exactly the four conditions")
    c = cfg["corruption"]
    _need(set(c["types"]) == {"gaussian_noise", "gaussian_blur", "brightness", "contrast"}, "corruption types differ")
    for name, spec in c["types"].items():
        _need(len(spec["severities"]) == 5, f"corruption {name} needs five severities")
    s = cfg["statistics"]
    _need(s["bootstrap_replicates"] >= 1000, "too few bootstrap replicates")


def placeholders(cfg: dict, prefix: str = PLACEHOLDER_PREFIX, _path: str = "") -> list:
    """Return the config paths whose value is still a TO_FILL placeholder (or None where required)."""
    out = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out += placeholders(v, prefix, f"{_path}.{k}" if _path else k)
    elif isinstance(cfg, str) and cfg.startswith(prefix):
        out.append(_path)
    return out


def config_hash(cfg: dict) -> str:
    """Identity of settings affecting the study; only the operational current epoch cap is excluded."""
    c = copy.deepcopy(cfg)
    c.pop("status", None)
    c.pop("study", None)
    # Only the current cap may change under the preregistered extension rule. The resource ceiling,
    # pilot configurations, objectives and all other protocol decisions remain part of the identity.
    c["training"].pop("max_epochs", None)
    return canonical_json_hash(c)


def require_frozen(cfg: dict) -> None:
    """Raise if the primary experiment is started with unresolved fields."""
    left = placeholders(cfg)
    if left:
        raise ConfigError(f"unresolved placeholders: {left}")
    if cfg["training"]["max_epochs"] is None:
        raise ConfigError("training.max_epochs is not set (determine it from the convergence pilot)")
    if cfg["training"]["absolute_epoch_ceiling"] is None:
        raise ConfigError("training.absolute_epoch_ceiling is not set")
    if cfg.get("status") != "FROZEN":
        raise ConfigError("status must be exactly 'FROZEN' before primary runs")
    ceiling = cfg["training"]["absolute_epoch_ceiling"]
    if cfg["training"]["max_epochs"] > ceiling:
        raise ConfigError("training.max_epochs cannot exceed training.absolute_epoch_ceiling")
    if cfg["pilot"].get("resource_ceiling") != ceiling:
        raise ConfigError("pilot.resource_ceiling must equal training.absolute_epoch_ceiling")
    unresolved = []
    def scan(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items(): scan(v, f"{path}.{k}" if path else k)
        elif isinstance(x, list):
            for i, v in enumerate(x): scan(v, f"{path}[{i}]")
        elif isinstance(x, str):
            low = x.lower()
            if "verify" in low or "proposed" in low or "to_fill" in low or "todo" in low:
                unresolved.append((path, x))
    scan(cfg)
    if unresolved:
        raise ConfigError(f"unresolved protocol decisions remain: {unresolved}")


def run_id(backbone: str, framework: str, augmentation: str, seed: int) -> str:
    return f"{backbone}__{framework}__{augmentation}__seed{seed}"


def resolve(path: str, root: str) -> str:
    """Resolve a repo-relative path against the repository root."""
    return path if os.path.isabs(path) else os.path.join(root, path)
