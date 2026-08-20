"""Online edit-target construction from the reports already used by TIM."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

from .constants import (
    APPEAR_WORDS,
    FINDING_ALIASES,
    FINDINGS,
    IMPROVE_WORDS,
    INVERSE_OPERATION_INDEX,
    RESOLVE_WORDS,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return appear, resolve, worsen, improve, and ambiguous masks ``[B, F]``."""

    batch_size = len(reports)
    num_findings = len(FINDINGS)
    appear = torch.zeros(batch_size, num_findings, dtype=torch.bool, device=device)
    resolve = torch.zeros_like(appear)
    worsen = torch.zeros_like(appear)
    improve = torch.zeros_like(appear)

    for batch_index, report in enumerate(reports):
        sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(report or "") if s.strip()]
        for finding_index, finding in enumerate(FINDINGS):
            aliases = FINDING_ALIASES[finding]
            for sentence in sentences:
                if not _contains_any(sentence, aliases):
                    continue
                appear[batch_index, finding_index] |= _contains_any(sentence, APPEAR_WORDS)
                resolve[batch_index, finding_index] |= _contains_any(sentence, RESOLVE_WORDS)
                worsen[batch_index, finding_index] |= _contains_any(sentence, WORSEN_WORDS)
                improve[batch_index, finding_index] |= _contains_any(sentence, IMPROVE_WORDS)

    ambiguous = (appear & resolve) | (worsen & improve)
    return (
        appear & ~ambiguous,
        resolve & ~ambiguous,
        worsen & ~ambiguous,
        improve & ~ambiguous,
        ambiguous,
    )


def build_operation_targets(
    previous_labels: torch.Tensor,
    current_labels: torch.Tensor,
    current_reports: Sequence[str],
) -> torch.Tensor:
    """Create conservative weak operation targets without rewriting annotations.

    CheXbert uses ``0=not mentioned``, ``1=positive``, ``2=negative`` and
    ``3=uncertain``. A missing mention is *not* treated as resolution: a
    positive finding only resolves when the current report is explicitly
    negative or contains a finding-local resolution cue.
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
    previous_negative = previous.eq(2)
    current_negative = current.eq(2)
    previous_uncertain = previous.eq(3)
    current_uncertain = current.eq(3)

    appear, resolve, worsen, improve, ambiguous = _direction_masks(
        current_reports,
        device=device,
    )

    certain_transition = ~(previous_uncertain | current_uncertain)
    appearance = current_positive & (~previous_positive) & certain_transition
    appearance |= appear & current_positive & (~previous_positive) & certain_transition
    resolution = previous_positive & current_negative & certain_transition
    resolution |= resolve & previous_positive & (~current_positive) & certain_transition

    targets[appearance] = int(EditOperation.APPEAR)
    targets[resolution] = int(EditOperation.RESOLVE)

    persistent = previous_positive & current_positive & certain_transition
    targets[persistent & worsen] = int(EditOperation.WORSEN)
    targets[persistent & improve] = int(EditOperation.IMPROVE)
    targets[persistent & ambiguous] = int(EditOperation.UNCERTAIN)

    # Conflicting explicit cues and label states are safer as UNCERTAIN than as
    # a hard temporal direction.
    cue_conflict = (
        (appear & previous_positive)
        | (resolve & current_positive)
        | (worsen & improve)
        | (previous_negative & current_negative & (appear | resolve))
    )
    targets[cue_conflict] = int(EditOperation.UNCERTAIN)

    uncertain_transition = previous_uncertain | current_uncertain
    targets[uncertain_transition] = int(EditOperation.UNCERTAIN)
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
