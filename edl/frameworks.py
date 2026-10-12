"""Framework strategy objects: loss + predictive probabilities + native uncertainty.

Every framework's predict() returns the same keys:
    probs               [B, K]  predictive probabilities q
    native_uncertainty  [B]     framework's own OOD score (larger = more OOD)
    msp_uncertainty     [B]     1 - max_k q_k
    entropy             [B]     -sum_k q_k log q_k (natural log)
plus framework-specific raw outputs for export (logits / alpha / p / tau / total_uncertainty).
"""
import torch

from edl.constants import FRAMEWORKS
from edl.losses.evidential import (edl_alpha, edl_loss, redl_alpha, redl_loss, softmax_cross_entropy)
from edl.losses.fedl import compute_moments, fedl_loss


def _finish(probs, native, extras):
    msp = 1.0 - probs.max(dim=1).values
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    out = {"probs": probs, "native_uncertainty": native, "msp_uncertainty": msp, "entropy": ent}
    out.update(extras)
    return out


class Framework:
    name = ""

    def loss(self, out: dict, target: torch.Tensor, epoch: int) -> torch.Tensor:
        raise NotImplementedError

    def loss_with_parts(self, out: dict, target: torch.Tensor, epoch: int):
        loss = self.loss(out, target, epoch)
        return loss, {"loss": float(loss.detach().cpu())}

    def predict(self, out: dict) -> dict:
        raise NotImplementedError


class SoftmaxFramework(Framework):
    name = "softmax"

    def loss(self, out, target, epoch):
        return softmax_cross_entropy(out["logits"], target)

    def loss_with_parts(self, out, target, epoch):
        loss = self.loss(out, target, epoch)
        return loss, {"cross_entropy": float(loss.detach().cpu())}

    def predict(self, out):
        probs = torch.softmax(out["logits"], dim=1)
        native = 1.0 - probs.max(dim=1).values
        return _finish(probs, native, {"logits": out["logits"]})


class EDLFramework(Framework):
    name = "edl"

    def __init__(self, annealing_step: int = 10):
        self.annealing_step = annealing_step

    def loss(self, out, target, epoch):
        return edl_loss(out["logits"], target, epoch, self.annealing_step)

    def loss_with_parts(self, out, target, epoch):
        return edl_loss(out["logits"], target, epoch, self.annealing_step, return_parts=True)

    def predict(self, out):
        alpha = edl_alpha(out["logits"])
        s = alpha.sum(dim=1, keepdim=True)
        k = alpha.shape[1]
        return _finish(alpha / s, (k / s).squeeze(1), {"logits": out["logits"], "alpha": alpha})


class REDLFramework(Framework):
    name = "r_edl"

    def __init__(self, lam: float = 0.1, annealing_step: int = 10):
        self.lam = lam
        self.annealing_step = annealing_step

    def loss(self, out, target, epoch):
        return redl_loss(out["logits"], target, epoch, self.annealing_step, self.lam)

    def loss_with_parts(self, out, target, epoch):
        return redl_loss(out["logits"], target, epoch, self.annealing_step, self.lam, return_parts=True)

    def predict(self, out):
        alpha = redl_alpha(out["logits"], self.lam)
        s = alpha.sum(dim=1, keepdim=True)
        k = alpha.shape[1]
        native = (self.lam * k / s).squeeze(1)  # corrected uncertainty mass u = lambda * K / S
        return _finish(alpha / s, native, {"logits": out["logits"], "alpha": alpha})


class FEDLFramework(Framework):
    name = "f_edl"

    def loss(self, out, target, epoch):
        return fedl_loss(out["alpha"], out["p"], out["tau"], target)

    def loss_with_parts(self, out, target, epoch):
        return fedl_loss(out["alpha"], out["p"], out["tau"], target, return_parts=True)

    def predict(self, out):
        alpha, p, tau = out["alpha"], out["p"], out["tau"]
        mu, var = compute_moments(alpha, p, tau)
        epistemic = var.sum(dim=1)
        total = 1.0 - mu.pow(2).sum(dim=1)
        return _finish(mu, epistemic, {"alpha": alpha, "p": p, "tau": tau, "total_uncertainty": total})


def build_framework(name: str, cfg: dict) -> Framework:
    if name not in FRAMEWORKS:
        raise ValueError(f"unknown framework {name}; expected one of {FRAMEWORKS}")
    f = cfg["frameworks"]
    if name == "softmax":
        return SoftmaxFramework()
    if name == "edl":
        return EDLFramework(f["edl"]["kl_annealing_step"])
    if name == "r_edl":
        return REDLFramework(f["r_edl"]["lambda"], f["r_edl"]["kl_annealing_step"])
    return FEDLFramework()
