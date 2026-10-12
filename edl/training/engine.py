"""Training engine for one run: train/validate, select by validation macro-F1, early stop, resume.

Epochs are ONE-BASED everywhere (the first epoch is 1), which fixes the EDL KL annealing convention.
The test partition is never touched here; test predictions are exported later by the runner,
only after the run is final.
"""
import csv
import os
import time
import copy
from dataclasses import dataclass

import numpy as np

from edl.constants import NUM_CLASSES
from edl.data.ham10000 import (build_eval_transform, build_train_transform, class_weights_sampler,
                               make_dataset, make_loader)
from edl.evaluation.metrics import ece, macro_f1
from edl.evaluation.predict import predict_arrays
from edl.frameworks import build_framework
from edl.models.model import LightweightModel
from edl.training.checkpoint import BEST, LATEST, RunCheckpointer, load_checkpoint
from edl.utils.seed import get_rng_states, seed_everything, set_rng_states


class TrainingError(RuntimeError):
    """Raised for non-finite losses or invalid training states. The run is recorded as failed."""


@dataclass
class RunSpec:
    run_id: str
    backbone: str
    framework: str
    augmentation: str
    seed: int


def _targets_for_batch(labels, augmentation, aug_cfg, images, aug_rng):
    """Return (images, targets). Mixup/CutMix produce soft targets [B, K]; others keep integer labels."""
    import torch
    import torch.nn.functional as F
    from edl.augment import apply_cutmix, apply_mixup
    if augmentation == "mixup":
        onehot = F.one_hot(labels, NUM_CLASSES).to(torch.float32)
        return apply_mixup(images, onehot, aug_cfg["mixup"]["alpha"], aug_rng)
    if augmentation == "cutmix":
        onehot = F.one_hot(labels, NUM_CLASSES).to(torch.float32)
        return apply_cutmix(images, onehot, aug_cfg["cutmix"]["alpha"], aug_rng)
    return images, labels


def train_one_epoch(model, framework, loader, optimizer, device, epoch, augmentation, aug_cfg, aug_rng,
                    non_blocking=True):
    """Train one epoch and return scalar loss plus stability/objective diagnostics."""
    import torch
    import torch.nn.functional as F
    from edl.losses.evidential import edl_alpha, redl_alpha
    model.train()
    totals, n_batches = {}, 0
    max_grad_norm, max_evidence, max_concentration, clamp_rates = 0.0, 0.0, 0.0, []
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        x, targets = _targets_for_batch(y, augmentation, aug_cfg, x, aug_rng)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss, parts = framework.loss_with_parts(out, targets, epoch)
        if not torch.isfinite(loss):
            raise TrainingError(f"non-finite loss at epoch {epoch}, step {step}")
        loss.backward()
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise TrainingError(f"non-finite gradient at epoch {epoch}, step {step}")
        if grads:
            grad_norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float(), 2) for g in grads]), 2).cpu())
            max_grad_norm = max(max_grad_norm, grad_norm)
        if framework.name == "edl":
            evidence = F.relu(out["logits"]); alpha = edl_alpha(out["logits"])
            max_evidence = max(max_evidence, float(evidence.max().detach().cpu()))
            max_concentration = max(max_concentration, float(alpha.sum(1).max().detach().cpu()))
        elif framework.name == "r_edl":
            evidence = F.relu(out["logits"]); alpha = redl_alpha(out["logits"], framework.lam)
            max_evidence = max(max_evidence, float(evidence.max().detach().cpu()))
            max_concentration = max(max_concentration, float(alpha.sum(1).max().detach().cpu()))
        elif framework.name == "f_edl":
            alpha, tau = out["alpha"], out["tau"]
            max_evidence = max(max_evidence, float(alpha.max().detach().cpu()))
            max_concentration = max(max_concentration, float((alpha.sum(1, keepdim=True) + tau).max().detach().cpu()))
            rate = getattr(model.head, "last_clamp_activation_rate", 0.0)
            clamp_rates.append(rate if torch.is_tensor(rate) else torch.tensor(float(rate), device=device))
        optimizer.step()
        for key, value in parts.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + float(value)
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())
        n_batches += 1
    den = max(n_batches, 1)
    diag = {f"train_{k}": v / den for k, v in totals.items()}
    diag.update({"max_grad_norm": max_grad_norm, "max_evidence_or_alpha": max_evidence,
                 "max_concentration": max_concentration,
                 "fedl_log_alpha_clamp_activation_rate": float(torch.stack(clamp_rates).mean().cpu()) if clamp_rates else 0.0,
                 "nonfinite_detected": False})
    return totals.get("loss", 0.0) / den, diag


def evaluate_split(model, framework, loader, device, non_blocking=True) -> dict:
    arrays = predict_arrays(model, framework, loader, device, non_blocking)
    pred = arrays["probs"].argmax(1)
    arrays["macro_f1"] = macro_f1(arrays["labels"], pred, NUM_CLASSES)
    arrays["accuracy"] = float((pred == arrays["labels"]).mean())
    arrays["ece"] = ece(arrays["probs"], arrays["labels"])
    return arrays


def _write_history(path, history):
    if not history:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def train_run(spec: RunSpec, cfg: dict, manifest, paths: dict, work_dir: str, drive_dir: str, identity: dict,
              device, max_epochs: int, allow_commit_change: bool = False, sync_fail_hook=None, log=print) -> dict:
    """Train (or resume) one run until early stopping or the epoch cap. Returns a result dict."""
    import torch

    seed_everything(spec.seed)
    t_cfg, l_cfg, aug_cfg = cfg["training"], cfg["data_loader"], cfg["augmentation"]
    non_blocking = bool(l_cfg["non_blocking_transfer"])

    model = LightweightModel(spec.backbone, spec.framework, NUM_CLASSES, pretrained=cfg["models"]["pretrained"]).to(device)
    framework = build_framework(spec.framework, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"])

    train_ds = make_dataset(manifest, "train", paths, build_train_transform(spec.augmentation, aug_cfg))
    val_ds = make_dataset(manifest, "val", paths, build_eval_transform())
    sampler, sampler_gen = class_weights_sampler(train_ds.labels, spec.seed)
    train_loader, loader_gen = make_loader(train_ds, t_cfg["batch_size"], l_cfg, sampler=sampler, seed=spec.seed)
    val_loader, _ = make_loader(val_ds, t_cfg["batch_size"], l_cfg, seed=spec.seed)
    aug_rng = np.random.default_rng([spec.seed, 7])

    ckpt = RunCheckpointer(work_dir, drive_dir, identity, cfg["checkpoint"]["sync_every_n_epochs"],
                           cfg["checkpoint"]["max_sync_retries"], cfg["checkpoint"]["retry_backoff_seconds"],
                           fail_hook=sync_fail_hook)

    start_epoch, best_f1, best_epoch, no_improve, history, stop_reason = 0, -1.0, 0, 0, [], None
    best_model_state, best_extra = None, {}
    state = ckpt.load_resume(allow_commit_change)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_f1, best_epoch = state["epoch"], state["best_f1"], state["best_epoch"]
        no_improve, history, stop_reason = state["no_improve"], state["history"], state["stop_reason"]
        best_model_state = state.get("best_model_state")
        best_extra = state.get("best_extra", {})
        if best_model_state is None:
            # Compatibility for early development checkpoints: load only an exact matching local BEST.
            best_file = os.path.join(work_dir, BEST)
            if not os.path.exists(best_file):
                raise TrainingError("resume checkpoint lacks embedded best_model_state and matching best_checkpoint.pth")
            legacy_best = load_checkpoint(best_file)
            if int(legacy_best.get("epoch", -1)) != int(best_epoch):
                raise TrainingError("resume best checkpoint epoch does not match latest.best_epoch")
            best_model_state = legacy_best["model"]
            best_extra = legacy_best.get("extra", {})
        set_rng_states(state["rng"])
        aug_rng.bit_generator.state = state["aug_rng"]
        sampler_gen.set_state(state["sampler_gen"])
        loader_gen.set_state(state["loader_gen"])
        ckpt.note_best_epoch(best_epoch)
        log(f"[{spec.run_id}] resumed from epoch {start_epoch} ({state['_source']} copy)")
        if stop_reason == "cap" and max_epochs > start_epoch:
            stop_reason = None  # the cap was raised: continue
    t0 = time.time()

    def cpu_state_dict(state_dict):
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def snapshot():
        return {"model": cpu_state_dict(model.state_dict()), "optimizer": optimizer.state_dict(), "best_f1": best_f1,
                "best_epoch": best_epoch, "best_model_state": best_model_state, "best_extra": best_extra,
                "no_improve": no_improve, "history": history, "stop_reason": stop_reason,
                "rng": get_rng_states(), "aug_rng": aug_rng.bit_generator.state,
                "sampler_gen": sampler_gen.get_state(), "loader_gen": loader_gen.get_state()}

    epoch = start_epoch
    while stop_reason is None and epoch < max_epochs:
        epoch += 1
        e0 = time.time()
        train_loss, train_diag = train_one_epoch(model, framework, train_loader, optimizer, device, epoch,
                                                 spec.augmentation, aug_cfg, aug_rng, non_blocking)
        val = evaluate_split(model, framework, val_loader, device, non_blocking)
        improved = val["macro_f1"] > best_f1          # strict: the earliest epoch wins ties
        if improved:
            best_f1, best_epoch, no_improve = val["macro_f1"], epoch, 0
            best_model_state = cpu_state_dict(model.state_dict())
            best_extra = {"val_accuracy": val["accuracy"]}
            ckpt.save_best(best_model_state, epoch, best_f1, best_extra)
        else:
            no_improve += 1
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_macro_f1": round(val["macro_f1"], 6),
                        "val_accuracy": round(val["accuracy"], 6), "val_ece": round(val["ece"], 6),
                        **{k: round(float(v), 8) if isinstance(v, (float, int)) else v for k, v in train_diag.items()},
                        "best_epoch": best_epoch, "no_improve": no_improve, "seconds": round(time.time() - e0, 1)})
        if no_improve >= t_cfg["patience"]:
            stop_reason = "early_stop"
        elif epoch >= max_epochs:
            stop_reason = "cap"
        ckpt.save_latest(snapshot(), epoch, force=stop_reason is not None)
        _write_history(os.path.join(work_dir, "epochs.csv"), history)
        log(f"[{spec.run_id}] epoch {epoch}: loss {train_loss:.4f} | val F1 {val['macro_f1']:.4f} | "
            f"best {best_f1:.4f}@{best_epoch} | {history[-1]['seconds']}s")

    ckpt.finalize(epoch)
    _write_history(os.path.join(work_dir, "epochs.csv"), history)
    return {"run_id": spec.run_id, "last_epoch": epoch, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            "stop_reason": stop_reason, "max_epochs": max_epochs, "patience": t_cfg["patience"],
            "seconds_this_session": round(time.time() - t0, 1), "history": history,
            "best_path": os.path.join(work_dir, BEST), "latest_path": os.path.join(work_dir, LATEST)}


def load_best_model(spec: RunSpec, cfg: dict, best_path: str, device, expect_identity: dict = None):
    """Rebuild the model and load best_checkpoint.pth (weights only)."""
    from edl.training.checkpoint import check_identity
    ck = load_checkpoint(best_path)
    if expect_identity is not None:
        check_identity(ck["identity"], expect_identity, allow_commit_change=True)
    model = LightweightModel(spec.backbone, spec.framework, NUM_CLASSES, pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
