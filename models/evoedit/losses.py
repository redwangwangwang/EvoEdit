"""Losses for executable, invertible, and sparse clinical editing."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .constants import INVERSE_OPERATION_INDEX, EditOperation


def operation_supervision_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def inverse_algebra_loss(forward_probs: torch.Tensor, reverse_probs: torch.Tensor) -> torch.Tensor:
    inverse_index = torch.as_tensor(
        INVERSE_OPERATION_INDEX,
        device=forward_probs.device,
        dtype=torch.long,
    )
    desired_reverse = forward_probs.index_select(-1, inverse_index)
    midpoint = 0.5 * (desired_reverse + reverse_probs)
    eps = torch.finfo(forward_probs.dtype).eps
    first = desired_reverse * (desired_reverse.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    second = reverse_probs * (reverse_probs.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    return 0.5 * (first.sum(dim=-1).mean() + second.sum(dim=-1).mean())


def cycle_content_loss(forward_slots: torch.Tensor, reverse_slots: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(forward_slots, reverse_slots, dim=-1)).mean()


def sparsity_loss(operation_probs: torch.Tensor) -> torch.Tensor:
    return (1.0 - operation_probs[..., int(EditOperation.KEEP)]).mean()


def usage_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    """Prevent latent anatomy/severity codebooks from collapsing to one code."""

    mean_usage = probabilities.mean(dim=tuple(range(probabilities.ndim - 1)))
    mean_usage = mean_usage.clamp_min(torch.finfo(probabilities.dtype).eps)
    uniform_log_probability = -math.log(probabilities.shape[-1])
    return (mean_usage * (mean_usage.log() - uniform_log_probability)).sum()


def confidence_supervision_loss(confidence: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    certain = targets.ne(int(EditOperation.UNCERTAIN)).to(confidence.dtype).unsqueeze(-1)
    return F.binary_cross_entropy(confidence, certain)


def verifier_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def masked_intervention_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_samples: torch.Tensor,
) -> torch.Tensor:
    per_slot = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction="none",
    )
    if not valid_samples.any():
        return logits.sum() * 0.0
    return per_slot[valid_samples].mean()
