"""F-EDL head and loss (Flexible Evidential Deep Learning), WITHOUT spectral normalization.

Head outputs alpha = exp(g_alpha(z)) (clamped), p = softmax(g_p(z)), tau = softplus(g_tau(z)).
Predictive mean : mu_k = (alpha_k + tau * p_k) / (sum_j alpha_j + tau)
Loss            : E||y - pi||^2 + ||y - p||^2, with E||y - pi||^2 = ||y - mu||^2 + sum_k var_k
Native OOD score: sum_k var_k (epistemic). Secondary: 1 - sum_k mu_k^2.
Verify the variance expression against the F-EDL paper before freeze (objective_status in the config).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from edl.losses.evidential import to_target_matrix


class FEDLHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.g_alpha = nn.Linear(in_features, num_classes)
        self.g_p = nn.Linear(in_features, num_classes)
        self.g_tau = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor):
        raw_log_alpha = self.g_alpha(features)
        self.last_clamp_activation_rate = ((raw_log_alpha < -20.0) | (raw_log_alpha > 20.0)).float().mean().detach()
        log_alpha = torch.clamp(raw_log_alpha, min=-20.0, max=20.0)
        alpha = torch.exp(log_alpha)
        p = F.softmax(self.g_p(features), dim=1)
        tau = F.softplus(self.g_tau(features))
        return alpha, p, tau


def compute_moments(alpha: torch.Tensor, p: torch.Tensor, tau: torch.Tensor):
    """Mean and class-wise variance of the Flexible Dirichlet."""
    alpha0 = alpha.sum(dim=1, keepdim=True)
    denom = alpha0 + tau
    mu = (alpha + tau * p) / denom
    var = mu * (1.0 - mu) / (denom + 1.0) + tau.pow(2) * p * (1.0 - p) / (denom * (denom + 1.0))
    return mu, var


def fedl_loss(alpha, p, tau, target, return_parts: bool = False):
    k = alpha.shape[1]
    y = to_target_matrix(target, k)
    mu, var = compute_moments(alpha, p, tau)
    cls = ((y - mu).pow(2) + var).sum(dim=1)
    alloc = (y - p).pow(2).sum(dim=1)
    total = (cls + alloc).mean()
    if return_parts:
        return total, {"classification": cls.mean().item(), "allocation": alloc.mean().item()}
    return total
