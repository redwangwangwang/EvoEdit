"""Losses for balanced, invertible, and executable clinical editing."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .constants import INVERSE_OPERATION_INDEX, EditOperation

DEFAULT_OPERATION_CLASS_WEIGHTS = (1.0, 3.0, 3.0, 4.0, 4.0, 2.0)


def _class_weights(
    class_weights: Sequence[float] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = DEFAULT_OPERATION_CLASS_WEIGHTS if class_weights is None else class_weights
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def operation_supervision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Sequence[float] | torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-balanced focal cross entropy for sparse edit operations."""

    logits_float = logits.float().reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1).long()
    log_probabilities = F.log_softmax(logits_float, dim=-1)
    target_log_probability = log_probabilities.gather(1, targets_flat[:, None]).squeeze(1)
    target_probability = target_log_probability.exp()
    alpha = _class_weights(class_weights, device=logits.device)[targets_flat]
    focal = (1.0 - target_probability).pow(float(gamma))
    return (alpha * focal * -target_log_probability).mean()


def inverse_algebra_loss(
    forward_probs: torch.Tensor,
    reverse_probs: torch.Tensor,
    targets: torch.Tensor | None = None,
    keep_weight: float = 0.05,
) -> torch.Tensor:
    """Active-slot Jensen-Shannon consistency for the inverse edit algebra."""

    forward = forward_probs.float()
    reverse = reverse_probs.float()
    inverse_index = torch.as_tensor(
        INVERSE_OPERATION_INDEX,
        device=forward.device,
        dtype=torch.long,
    )
    desired_reverse = forward.index_select(-1, inverse_index)
    midpoint = 0.5 * (desired_reverse + reverse)
    eps = torch.finfo(torch.float32).eps
    first = desired_reverse * (
        desired_reverse.clamp_min(eps).log() - midpoint.clamp_min(eps).log()
    )
    second = reverse * (reverse.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    per_slot = 0.5 * (first.sum(dim=-1) + second.sum(dim=-1))
    per_slot = per_slot.clamp_min(0.0)
    if targets is None:
        return per_slot.mean()
    weights = torch.where(
        targets.eq(int(EditOperation.KEEP)),
        torch.full_like(per_slot, float(keep_weight)),
        torch.ones_like(per_slot),
    )
    return _weighted_mean(per_slot, weights)


def cycle_content_loss(
    forward_slots: torch.Tensor,
    reverse_slots: torch.Tensor,
    targets: torch.Tensor | None = None,
    keep_weight: float = 0.05,
) -> torch.Tensor:
    """Match direction-invariant change content without the shared finding query.

    Opposite temporal directions may flip a residual's sign, so absolute cosine
    similarity is used. Computation is forced to FP32 to avoid bf16 zeros.
    """

    forward = F.normalize(forward_slots.float(), dim=-1)
    reverse = F.normalize(reverse_slots.float(), dim=-1)
    similarity = (forward * reverse).sum(dim=-1).abs().clamp(max=1.0)
    per_slot = 1.0 - similarity
    if targets is None:
        return per_slot.mean()
    weights = torch.where(
        targets.eq(int(EditOperation.KEEP)),
        torch.full_like(per_slot, float(keep_weight)),
        torch.ones_like(per_slot),
    )
    return _weighted_mean(per_slot, weights)


def sparsity_loss(
    operation_probs: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calibrate edit density instead of rewarding the all-KEEP solution."""

    predicted_change = 1.0 - operation_probs.float()[..., int(EditOperation.KEEP)]
    if targets is None:
        return predicted_change.mean()
    target_change = targets.ne(int(EditOperation.KEEP)).float()
    return F.smooth_l1_loss(
        predicted_change.mean(dim=-1),
        target_change.mean(dim=-1),
    )


def usage_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    """Encourage sharp per-slot codes and balanced batch-level code usage."""

    probabilities = probabilities.float().clamp_min(torch.finfo(torch.float32).eps)
    num_codes = probabilities.shape[-1]
    log_codes = math.log(num_codes)
    conditional_entropy = -(
        probabilities * probabilities.log()
    ).sum(dim=-1).mean() / log_codes
    mean_usage = probabilities.mean(dim=tuple(range(probabilities.ndim - 1)))
    marginal_kl = (
        mean_usage * (mean_usage.clamp_min(1e-8).log() + log_codes)
    ).sum()
    return conditional_entropy + marginal_kl


def confidence_supervision_loss(
    confidence: torch.Tensor,
    operation_logits: torch.Tensor,
    targets: torch.Tensor,
    non_keep_weight: float = 2.0,
) -> torch.Tensor:
    """Calibrate confidence to actual operation correctness, not uncertainty tags."""

    correctness = operation_logits.detach().argmax(dim=-1).eq(targets).float().unsqueeze(-1)
    per_slot = F.binary_cross_entropy(
        confidence.float().clamp(1e-6, 1.0 - 1e-6),
        correctness,
        reduction="none",
    ).squeeze(-1)
    weights = torch.where(
        targets.eq(int(EditOperation.KEEP)),
        torch.ones_like(per_slot),
        torch.full_like(per_slot, float(non_keep_weight)),
    )
    return _weighted_mean(per_slot, weights)


def preserve_gate_loss(
    preserve_gate: torch.Tensor,
    targets: torch.Tensor,
    non_keep_weight: float = 3.0,
) -> torch.Tensor:
    """Supervise preservation explicitly and upweight clinically active slots."""

    gate_targets = targets.eq(int(EditOperation.KEEP)).float()
    gate = preserve_gate.squeeze(-1).float().clamp(1e-6, 1.0 - 1e-6)
    per_slot = F.binary_cross_entropy(gate, gate_targets, reduction="none")
    weights = torch.where(
        gate_targets.bool(),
        torch.ones_like(per_slot),
        torch.full_like(per_slot, float(non_keep_weight)),
    )
    return _weighted_mean(per_slot, weights)


def verifier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    return operation_supervision_loss(
        logits,
        targets,
        class_weights=class_weights,
        gamma=0.0,
    )


def masked_intervention_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_slots: torch.Tensor,
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify only intervened and clinically active slots."""

    if not valid_slots.any():
        return logits.sum() * 0.0
    per_slot = F.cross_entropy(
        logits.float().transpose(1, 2),
        targets,
        reduction="none",
    )
    alpha = _class_weights(class_weights, device=logits.device)[targets.long()]
    mask = valid_slots.to(per_slot.dtype)
    return (per_slot * alpha * mask).sum() / (alpha * mask).sum().clamp_min(1e-8)


def operation_diagnostics(
    operation_probs: torch.Tensor,
    targets: torch.Tensor,
    preserve_gate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return collapse-sensitive operation metrics for Lightning logging."""

    probabilities = operation_probs.float()
    predictions = probabilities.argmax(dim=-1)
    targets = targets.long()
    keep = int(EditOperation.KEEP)
    eps = torch.tensor(1e-8, device=probabilities.device)

    accuracy = predictions.eq(targets).float().mean()
    f1_values = []
    for class_index in range(probabilities.shape[-1]):
        predicted_class = predictions.eq(class_index)
        target_class = targets.eq(class_index)
        true_positive = (predicted_class & target_class).float().sum()
        precision = true_positive / (predicted_class.float().sum() + eps)
        recall = true_positive / (target_class.float().sum() + eps)
        f1_values.append(2.0 * precision * recall / (precision + recall + eps))
    macro_f1 = torch.stack(f1_values).mean()

    predicted_change = predictions.ne(keep)
    target_change = targets.ne(keep)
    true_change = (predicted_change & target_change).float().sum()
    non_keep_precision = true_change / (predicted_change.float().sum() + eps)
    non_keep_recall = true_change / (target_change.float().sum() + eps)
    non_keep_f1 = 2.0 * non_keep_precision * non_keep_recall / (
        non_keep_precision + non_keep_recall + eps
    )

    entropy = -(
        probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()
    return {
        "operation_accuracy": accuracy,
        "operation_macro_f1": macro_f1,
        "non_keep_precision": non_keep_precision,
        "non_keep_recall": non_keep_recall,
        "non_keep_f1": non_keep_f1,
        "target_non_keep_rate": target_change.float().mean(),
        "pred_non_keep_rate": predicted_change.float().mean(),
        "mean_keep_probability": probabilities[..., keep].mean(),
        "operation_entropy": entropy,
        "mean_preserve_gate": preserve_gate.float().mean(),
    }
