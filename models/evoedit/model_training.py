"""Training mixin for EvoEdit."""

from __future__ import annotations

from typing import Any

import torch

from .losses import (
    confidence_supervision_loss,
    cycle_content_loss,
    inverse_algebra_loss,
    operation_supervision_loss,
    sparsity_loss,
    usage_balance_loss,
    verifier_loss,
)
from .targets import build_targets_with_chexbert


class EvoEditTrainingMixin:
    def forward(self, samples: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.llama_tokenizer.padding_side = "right"
        previous_reports = samples["prev_text"]
        current_reports = samples["curr_text"]
        previous_visual, current_visual = self._encode_pair(samples)

        forward, forward_exec, previous_ids, previous_tokens, previous_mask = self._build_direction(
            previous_visual, current_visual, previous_reports
        )
        reverse, reverse_exec, current_ids, current_tokens, current_mask = self._build_direction(
            current_visual, previous_visual, current_reports
        )
        current_prompt, current_prompt_mask = self.prompt_wrap(
            current_visual, forward_exec.executed_slots, previous_reports, "curr"
        )
        previous_prompt, previous_prompt_mask = self.prompt_wrap(
            previous_visual, reverse_exec.executed_slots, current_reports, "prior"
        )
        current_lm, current_copy = self._run_language_model(
            current_prompt,
            current_prompt_mask,
            current_reports,
            previous_ids,
            previous_tokens,
            previous_mask,
        )
        previous_lm, previous_copy = self._run_language_model(
            previous_prompt,
            previous_prompt_mask,
            previous_reports,
            current_ids,
            current_tokens,
            current_mask,
        )

        forward_targets, reverse_targets, _, _ = build_targets_with_chexbert(
            self.chexbert_metrics.chexbert, previous_reports, current_reports
        )
        forward_targets = forward_targets.to(previous_visual.device)
        reverse_targets = reverse_targets.to(previous_visual.device)

        operation = 0.5 * (
            operation_supervision_loss(forward.operation_logits, forward_targets)
            + operation_supervision_loss(reverse.operation_logits, reverse_targets)
        )
        inverse = inverse_algebra_loss(forward.operation_probs, reverse.operation_probs)
        cycle = cycle_content_loss(forward.content_slots, reverse.content_slots)
        verification = 0.5 * (
            verifier_loss(self.program_verifier(forward_exec.executed_slots), forward_targets)
            + verifier_loss(self.program_verifier(reverse_exec.executed_slots), reverse_targets)
        )
        confidence = 0.5 * (
            confidence_supervision_loss(forward.confidence, forward_targets)
            + confidence_supervision_loss(reverse.confidence, reverse_targets)
        )
        intervention = 0.5 * (
            self._intervention_loss(forward, previous_tokens, previous_mask, forward_targets)
            + self._intervention_loss(reverse, current_tokens, current_mask, reverse_targets)
        )
        sparse = 0.5 * (
            sparsity_loss(forward.operation_probs) + sparsity_loss(reverse.operation_probs)
        )
        factor_balance = 0.25 * (
            usage_balance_loss(forward.anatomy_probs)
            + usage_balance_loss(reverse.anatomy_probs)
            + usage_balance_loss(forward.severity_probs)
            + usage_balance_loss(reverse.severity_probs)
        )
        copy = 0.5 * (current_copy + previous_copy)
        report = current_lm + self.hparams.prior_report_weight * previous_lm
        pathology = report.new_zeros(())
        if self.hparams.pathology_loss_weight > 0:
            pathology = 0.5 * (
                self.pathology_loss(previous_visual, previous_reports)
                + self.pathology_loss(current_visual, current_reports)
            )

        total = (
            report
            + self.hparams.operation_loss_weight * operation
            + self.hparams.inverse_loss_weight * inverse
            + self.hparams.cycle_loss_weight * cycle
            + self.hparams.verifier_loss_weight * verification
            + self.hparams.confidence_loss_weight * confidence
            + self.hparams.intervention_loss_weight * intervention
            + self.hparams.sparsity_loss_weight * sparse
            + self.hparams.factor_balance_weight * factor_balance
            + self.hparams.copy_loss_weight * copy
            + self.hparams.pathology_loss_weight * pathology
        )
        return {
            "loss": total,
            "report_loss": report.detach(),
            "operation_loss": operation.detach(),
            "inverse_loss": inverse.detach(),
            "cycle_loss": cycle.detach(),
            "verifier_loss": verification.detach(),
            "intervention_loss": intervention.detach(),
            "copy_loss": copy.detach(),
            "sparsity_loss": sparse.detach(),
            "factor_balance_loss": factor_balance.detach(),
            "pathology_loss": pathology.detach(),
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        result = self(batch)
        self.log_dict(result, prog_bar=True, sync_dist=True)
        return result
