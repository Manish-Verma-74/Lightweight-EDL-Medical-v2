# Standard EDL objective finalized for protocol-v1.
# See protocol/protocol_v1.md for the objective and KL annealing policy.
"""Softmax (soft-target), standard EDL and R-EDL losses.

Targets may be integer labels [B] or soft targets [B, K] (Mixup / CutMix). Soft targets are
never converted back to hard labels.

EDL   : alpha = relu(output) + 1 ; loss = squared error + expected variance + annealed KL
R-EDL : alpha = relu(output) + lambda ; loss = squared error on alpha/S + annealed KL
        (controlled adaptation: verify against the official R-EDL objective before freeze)
KL    : KL(Dir(alpha_tilde) || Dir(1)) with alpha_tilde = alpha * (1 - t) + t
        (for EDL this equals the original evidence * (1 - t) + 1).

"""
import torch
import torch.nn.functional as F

from edl.losses.schedule import annealing_coefficient


def to_target_matrix(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Integer labels [B] or soft targets [B, K] -> float targets [B, K] (validated)."""
    if target.dim() == 1:
        return F.one_hot(target.long(), num_classes=num_classes).to(torch.float32)
    if target.dim() == 2:
        if target.shape[1] != num_classes:
            raise ValueError(f"expected targets of shape [B, {num_classes}], got {tuple(target.shape)}")
        t = target.to(torch.float32)
        if bool((t < 0).any()) or bool((t > 1).any()):
            raise ValueError("soft targets must lie in [0, 1]")
        ones = torch.ones(t.shape[0], dtype=t.dtype, device=t.device)
        if not torch.allclose(t.sum(dim=1), ones, atol=1e-5):
            raise ValueError("each soft-target row must sum to 1")
        return t
    raise ValueError(f"targets must be [B] or [B, K], got {tuple(target.shape)}")


def softmax_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that accepts hard or soft targets."""
    t = to_target_matrix(target, logits.shape[1])
    return -(t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def kl_to_uniform_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ) per sample, shape [B, 1]."""
    beta = torch.ones_like(alpha)
    s_alpha = alpha.sum(dim=1, keepdim=True)
    s_beta = beta.sum(dim=1, keepdim=True)
    ln_b = torch.lgamma(s_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    ln_b_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(s_beta)
    dg0 = torch.digamma(s_alpha)
    dg1 = torch.digamma(alpha)
    return ((alpha - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True) + ln_b + ln_b_uni


def edl_alpha(output: torch.Tensor) -> torch.Tensor:
    return F.relu(output) + 1.0


def redl_alpha(output: torch.Tensor, lam: float) -> torch.Tensor:
    return F.relu(output) + lam


def edl_loss(output: torch.Tensor, target: torch.Tensor, epoch: int, annealing_step: int,
             return_parts: bool = False):
    """Standard EDL (Sensoy et al., 2018) MSE loss. epoch is one-based."""
    k = output.shape[1]
    alpha = edl_alpha(output)
    s = alpha.sum(dim=1, keepdim=True)
    m = alpha / s
    t = to_target_matrix(target, k)
    a_term = ((t - m) ** 2).sum(dim=1, keepdim=True)
    b_term = (alpha * (s - alpha) / (s * s * (s + 1.0))).sum(dim=1, keepdim=True)
    coef = annealing_coefficient(epoch, annealing_step)
    alpha_tilde = alpha * (1.0 - t) + t
    kl = kl_to_uniform_dirichlet(alpha_tilde)
    total = (a_term + b_term + coef * kl).mean()
    if return_parts:
        return total, {"mse": a_term.mean().item(), "var": b_term.mean().item(), "kl": kl.mean().item(), "coef": coef}
    return total


def redl_loss(output: torch.Tensor, target: torch.Tensor, epoch: int, annealing_step: int, lam: float,
              return_parts: bool = False):
    """R-EDL loss with prior weight lambda (Chen et al., ICLR 2024; see objective_status in the config)."""
    k = output.shape[1]
    alpha = redl_alpha(output, lam)
    s = alpha.sum(dim=1, keepdim=True)
    p = alpha / s
    t = to_target_matrix(target, k)
    sq = ((t - p) ** 2).sum(dim=1, keepdim=True)
    coef = annealing_coefficient(epoch, annealing_step)
    alpha_tilde = alpha * (1.0 - t) + t
    kl = kl_to_uniform_dirichlet(alpha_tilde)
    total = (sq + coef * kl).mean()
    if return_parts:
        return total, {"mse": sq.mean().item(), "kl": kl.mean().item(), "coef": coef}
    return total
