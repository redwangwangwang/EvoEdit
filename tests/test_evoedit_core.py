from __future__ import annotations

import torch

from models.evoedit.constants import EditOperation
from models.evoedit.copy import PointerCopyHead
from models.evoedit.losses import (
    inverse_algebra_loss,
    operation_supervision_loss,
    preserve_gate_loss,
    sparsity_loss,
    usage_balance_loss,
)
from models.evoedit.program import (
    CopyAndEditExecutor,
    FactorizedEditProgram,
    SoftTemporalCorrespondence,
)
from models.evoedit.targets import build_operation_targets, invert_operation_targets
from tools.audit_annotation import audit


def test_operation_targets_and_inverse() -> None:
    previous = torch.zeros(2, 13, dtype=torch.long)
    current = torch.zeros(2, 13, dtype=torch.long)
    current[0, 9] = 1
    previous[1, 8] = 1
    current[1, 8] = 2
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


def test_unmentioned_positive_finding_is_not_forced_to_resolve() -> None:
    previous = torch.zeros(1, 13, dtype=torch.long)
    current = torch.zeros(1, 13, dtype=torch.long)
    previous[0, 9] = 1
    targets = build_operation_targets(previous, current, ["Heart size is normal."])
    assert targets[0, 9].item() == int(EditOperation.KEEP)


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
    assert program.change_slots.shape == (batch, 13, hidden)
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


def test_gate_does_not_backprop_directly_into_keep_probability() -> None:
    torch.manual_seed(0)
    hidden = 16
    executor = CopyAndEditExecutor(hidden, num_heads=4, dropout=0.0)
    program_slots = torch.randn(1, 2, hidden, requires_grad=True)
    operation_logits = torch.randn(1, 2, 6, requires_grad=True)
    operation_probs = operation_logits.softmax(dim=-1)
    confidence = torch.rand(1, 2, 1, requires_grad=True)
    prior_states = torch.randn(1, 4, hidden)
    prior_mask = torch.ones(1, 4, dtype=torch.bool)
    output = executor(
        program_slots,
        operation_probs,
        confidence,
        prior_states,
        prior_mask,
    )
    output.preserve_gate.sum().backward()
    assert operation_logits.grad is None
    assert confidence.grad is None


def test_inverse_distribution_loss() -> None:
    forward = torch.zeros(1, 1, 6)
    reverse = torch.zeros(1, 1, 6)
    forward[..., int(EditOperation.APPEAR)] = 1
    reverse[..., int(EditOperation.RESOLVE)] = 1
    targets = torch.tensor([[int(EditOperation.APPEAR)]])
    assert inverse_algebra_loss(forward, reverse, targets).item() < 1e-7


def test_balanced_operation_and_gate_losses_penalize_active_collapse() -> None:
    keep_logits = torch.tensor([[[8.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    keep_target = torch.tensor([[int(EditOperation.KEEP)]])
    appear_target = torch.tensor([[int(EditOperation.APPEAR)]])
    assert operation_supervision_loss(keep_logits, appear_target) > operation_supervision_loss(
        keep_logits,
        keep_target,
    )

    good_gate = torch.tensor([[[0.9], [0.1]]])
    collapsed_gate = torch.tensor([[[0.9], [0.9]]])
    targets = torch.tensor([[int(EditOperation.KEEP), int(EditOperation.APPEAR)]])
    assert preserve_gate_loss(good_gate, targets) < preserve_gate_loss(collapsed_gate, targets)


def test_edit_rate_and_factor_balance_objectives() -> None:
    targets = torch.tensor(
        [[int(EditOperation.KEEP), int(EditOperation.APPEAR)]],
    )
    calibrated = torch.tensor(
        [[[0.9, 0.1, 0, 0, 0, 0], [0.1, 0.9, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    all_keep = torch.tensor(
        [[[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )
    assert sparsity_loss(calibrated, targets) < sparsity_loss(all_keep, targets)

    balanced = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uniform = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert usage_balance_loss(balanced) < usage_balance_loss(collapsed)
    assert usage_balance_loss(balanced) < usage_balance_loss(uniform)


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


def test_annotation_audit_detects_prompt_leakage() -> None:
    payload = {
        "train": [
            {"id": "clean", "report": "No focal consolidation or pleural effusion."},
            {
                "id": "dirty",
                "report": "User: You are a radiologist. Clinical edit program: keep.",
            },
        ]
    }
    summary = audit(payload)
    report = summary["splits"]["train"]["fields"]["report"]
    assert report["prompt_contaminated_records"] == 1
    assert summary["totals"]["duplicate_id_occurrences"] == 0
