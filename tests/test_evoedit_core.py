from __future__ import annotations

import torch

from models.evoedit.constants import EditOperation
from models.evoedit.copy import PointerCopyHead
from models.evoedit.losses import inverse_algebra_loss
from models.evoedit.program import (
    CopyAndEditExecutor,
    FactorizedEditProgram,
    SoftTemporalCorrespondence,
)
from models.evoedit.targets import build_operation_targets, invert_operation_targets


def test_operation_targets_and_inverse() -> None:
    previous = torch.zeros(2, 13, dtype=torch.long)
    current = torch.zeros(2, 13, dtype=torch.long)
    current[0, 9] = 1
    previous[1, 8] = 1
    previous[0, 7] = 1
    current[0, 7] = 1
    reports = [
        "Increased bibasilar atelectasis. New pleural effusion.",
        "No pneumothorax.",
    ]
    targets = build_operation_targets(previous, current, reports)
    assert targets[0, 9].item() == int(EditOperation.APPEAR)
    assert targets[1, 8].item() == int(EditOperation.RESOLVE)
    assert targets[0, 7].item() == int(EditOperation.WORSEN)
    inverse = invert_operation_targets(targets)
    assert inverse[0, 9].item() == int(EditOperation.RESOLVE)
    assert inverse[1, 8].item() == int(EditOperation.APPEAR)
    assert inverse[0, 7].item() == int(EditOperation.IMPROVE)


def test_program_and_executor_shapes() -> None:
    torch.manual_seed(0)
    batch, previous_tokens, current_tokens, hidden = 2, 7, 9, 32
    previous = torch.randn(batch, previous_tokens, hidden)
    current = torch.randn(batch, current_tokens, hidden)
    output = SoftTemporalCorrespondence(hidden, dropout=0.0)(previous, current)
    assert output.change_tokens.shape == (batch, current_tokens, hidden)
    assert output.attention.shape == (batch, current_tokens, previous_tokens)
    program = FactorizedEditProgram(
        hidden_size=hidden,
        num_findings=13,
        num_operations=6,
        num_anatomy_codes=8,
        num_severity_codes=3,
        num_heads=4,
        dropout=0.0,
    )(output.change_tokens)
    assert program.program_slots.shape == (batch, 13, hidden)
    assert program.operation_probs.shape == (batch, 13, 6)
    prior_states = torch.randn(batch, 11, hidden)
    prior_mask = torch.ones(batch, 11, dtype=torch.bool)
    execution = CopyAndEditExecutor(hidden, num_heads=4, dropout=0.0)(
        program.program_slots,
        program.operation_probs,
        program.confidence,
        prior_states,
        prior_mask,
    )
    assert execution.executed_slots.shape == (batch, 13, hidden)
    assert torch.all((execution.preserve_gate >= 0) & (execution.preserve_gate <= 1))


def test_inverse_distribution_loss() -> None:
    forward = torch.zeros(1, 1, 6)
    reverse = torch.zeros(1, 1, 6)
    forward[..., int(EditOperation.APPEAR)] = 1
    reverse[..., int(EditOperation.RESOLVE)] = 1
    assert inverse_algebra_loss(forward, reverse).item() < 1e-7


def test_pointer_copy_loss_is_finite() -> None:
    torch.manual_seed(0)
    head = PointerCopyHead(hidden_size=16)
    decoder = torch.randn(1, 3, 16)
    prior = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    attention = head(decoder, prior, mask)
    prior_ids = torch.tensor([[4, 8, 9, 2]])
    targets = torch.tensor([[8, 5, 2]])
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    loss = head.loss(attention, prior_ids, targets, target_mask)
    assert torch.isfinite(loss)
