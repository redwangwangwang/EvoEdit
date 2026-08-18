"""Online edit-target construction from the reports already used by TIM."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from .constants import (
    FINDING_ALIASES,
    FINDINGS,
    IMPROVE_WORDS,
    INVERSE_OPERATION_INDEX,
    WORSEN_WORDS,
    EditOperation,
)

_SENTENCE_SPLIT = re.compile(r"[.!?;]+")


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _direction_masks(
    reports: Sequence[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return worsen, improve, and ambiguous masks of shape ``[B, F]``."""

    batch_size = len(reports)
    num_findings = len(FINDINGS)
    worsen = torch.zeros(batch_size, num_findings, dtype=torch.bool, device=device)
    improve = torch.zeros_like(worsen)

    for batch_index, report in enumerate(reports):
        sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(report or "") if s.strip()]
        for finding_index, finding in enumerate(FINDINGS):
            aliases = FINDING_ALIASES[finding]
            for sentence in sentences:
                if not _contains_any(sentence, aliases):
                    continue
                worsen[batch_index, finding_index] |= _contains_any(sentence, WORSEN_WORDS)
                improve[batch_index, finding_index] |= _contains_any(sentence, IMPROVE_WORDS)

    ambiguous = worsen & improve
    return worsen & ~ambiguous, improve & ~ambiguous, ambiguous


def build_operation_targets(
    previous_labels: torch.Tensor,
    current_labels: torch.Tensor,
    current_reports: Sequence[str],
) -> torch.Tensor:
    """Create weak operation targets without adding or rewriting annotations.

    CheXbert labels use ``0=blank``, ``1=positive``, ``2=negative`` and
    ``3=uncertain``. The first 13 findings are used; ``no finding`` is excluded.
    """

    if previous_labels.ndim != 2 or current_labels.ndim != 2:
        raise ValueError("CheXbert labels must have shape [batch, findings].")
    if previous_labels.shape[0] != current_labels.shape[0]:
        raise ValueError("Previous and current label batches must match.")
    if previous_labels.shape[1] < len(FINDINGS) or current_labels.shape[1] < len(FINDINGS):
        raise ValueError(f"Expected at least {len(FINDINGS)} CheXbert findings.")
    if len(current_reports) != previous_labels.shape[0]:
        raise ValueError("The report batch must match the CheXbert batch.")

    previous = previous_labels[:, : len(FINDINGS)].long()
    current = current_labels[:, : len(FINDINGS)].long()
    device = previous.device

    targets = torch.full_like(previous, int(EditOperation.KEEP))
    previous_positive = previous.eq(1)
    current_positive = current.eq(1)
    previous_uncertain = previous.eq(3)
    current_uncertain = current.eq(3)

    targets[(~previous_positive) & current_positive] = int(EditOperation.APPEAR)
    targets[previous_positive & (~current_positive) & (~current_uncertain)] = int(EditOperation.RESOLVE)

    persistent = previous_positive & current_positive
    worsen, improve, ambiguous = _direction_masks(current_reports, device=device)
    targets[persistent & worsen] = int(EditOperation.WORSEN)
    targets[persistent & improve] = int(EditOperation.IMPROVE)
    targets[persistent & ambiguous] = int(EditOperation.UNCERTAIN)

    uncertain_transition = previous_uncertain | current_uncertain
    unresolved_uncertainty = uncertain_transition & targets.eq(int(EditOperation.KEEP))
    targets[unresolved_uncertainty] = int(EditOperation.UNCERTAIN)
    return targets


def invert_operation_targets(targets: torch.Tensor) -> torch.Tensor:
    """Apply the operation-level inverse algebra to integer targets."""

    inverse = torch.as_tensor(INVERSE_OPERATION_INDEX, device=targets.device, dtype=torch.long)
    return inverse[targets.long()]


def build_targets_with_chexbert(
    chexbert: torch.nn.Module,
    previous_reports: Sequence[str],
    current_reports: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the frozen TIM CheXbert model and derive forward/reverse edits."""

    with torch.no_grad():
        previous_labels = chexbert(list(previous_reports))
        current_labels = chexbert(list(current_reports))
        forward = build_operation_targets(previous_labels, current_labels, current_reports)
        reverse = invert_operation_targets(forward)
    return forward, reverse, previous_labels, current_labels
