"""Core TIM integration for EvoEdit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from models.model_stage1 import LongitudinalR2GenGPT as TIMStage1

from .constants import FINDINGS, OPERATION_NAMES, EditOperation
from .copy import PointerCopyHead
from .losses import masked_intervention_loss
from .program import (
    CopyAndEditExecutor,
    EditProgramOutput,
    ExecutorOutput,
    FactorizedEditProgram,
    SoftTemporalCorrespondence,
)


class EvoEditCore(TIMStage1):
    """TIM Stage I augmented with executable clinical edit programs."""

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        hidden_size = self.llama_model.config.hidden_size
        self.temporal_correspondence = SoftTemporalCorrespondence(
            hidden_size, dropout=args.evoedit_dropout
        )
        self.edit_program = FactorizedEditProgram(
            hidden_size=hidden_size,
            num_findings=len(FINDINGS),
            num_operations=len(OPERATION_NAMES),
            num_anatomy_codes=args.num_anatomy_codes,
            num_severity_codes=args.num_severity_codes,
            num_heads=args.evoedit_heads,
            dropout=args.evoedit_dropout,
            temperature=args.operation_temperature,
        )
        self.edit_executor = CopyAndEditExecutor(
            hidden_size,
            num_heads=args.evoedit_heads,
            dropout=args.evoedit_dropout,
        )
        self.program_verifier = nn.Linear(hidden_size, len(OPERATION_NAMES))
        self.pointer_copy = PointerCopyHead(hidden_size)
        self.prompt = (
            "You are a radiologist. Update the prior report using only changes "
            "supported by the current chest X-ray and executable clinical edit program. "
            "Preserve clinically stable facts and avoid unnecessary rewriting."
        )
        self.prompt_prior = (
            "You are a radiologist. Reconstruct the prior report from the later report, "
            "the prior image, and the inverse clinical edit program."
        )
        self.val_edit_outputs: list[dict[str, Any]] = []
        self.test_edit_outputs: list[dict[str, Any]] = []

    def _encode_pair(self, samples: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self.layer_norm(self.encode_img(samples["prev_image"]))
        current = self.layer_norm(self.encode_img(samples["curr_image"]))
        return previous, current

    def _encode_report_context(
        self,
        reports: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.llama_tokenizer(
            list(reports),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(device)
        return (
            tokens.input_ids,
            self.embed_tokens(tokens.input_ids),
            tokens.attention_mask.bool(),
        )

    def _build_direction(
        self,
        source_visual: torch.Tensor,
        target_visual: torch.Tensor,
        source_reports: Sequence[str],
    ) -> tuple[EditProgramOutput, ExecutorOutput, torch.Tensor, torch.Tensor, torch.Tensor]:
        change = self.temporal_correspondence(source_visual, target_visual).change_tokens
        program = self.edit_program(change)
        token_ids, token_embeddings, token_mask = self._encode_report_context(
            source_reports, source_visual.device
        )
        execution = self.edit_executor(
            program.program_slots,
            program.operation_probs,
            program.confidence,
            token_embeddings,
            token_mask,
        )
        return program, execution, token_ids, token_embeddings, token_mask

    def prompt_wrap(
        self,
        image_embed: torch.Tensor,
        prog_embed: torch.Tensor,
        context: Sequence[str],
        timepoint: str = "curr",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace TIM placeholders with image tokens and edit-program slots."""

        device = image_embed.device
        instruction = self.prompt if timepoint == "curr" else self.prompt_prior
        relation = "Prior report" if timepoint == "curr" else "Later report"
        context_tokens = self.llama_tokenizer(
            list(context),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        )
        clipped = self.llama_tokenizer.batch_decode(
            context_tokens.input_ids,
            add_special_tokens=False,
            skip_special_tokens=True,
        )
        prompts = [
            f"User: Image: <Image>. Clinical edit program: <Progression>. "
            f"{relation}: {report}. {instruction}\nAssistant:"
            for report in clipped
        ]
        tokens = self.llama_tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.hparams.prompt_max_length,
            add_special_tokens=False,
        )
        input_ids = tokens.input_ids.to(device)
        prompt_embeddings = self.embed_tokens(input_ids)
        prompt_attention = tokens.attention_mask.to(device)
        embedded_rows, attention_rows = [], []

        for batch_index in range(image_embed.shape[0]):
            image_pos = torch.nonzero(
                input_ids[batch_index].eq(self.image_token_id), as_tuple=False
            ).flatten()
            edit_pos = torch.nonzero(
                input_ids[batch_index].eq(self.progression_token_id), as_tuple=False
            ).flatten()
            if image_pos.numel() != 1 or edit_pos.numel() != 1:
                raise RuntimeError(
                    "Prompt lost <Image>/<Progression>; increase --prompt_max_length."
                )
            replacements = sorted(
                [
                    (int(image_pos.item()), image_embed[batch_index]),
                    (int(edit_pos.item()), prog_embed[batch_index]),
                ],
                key=lambda item: item[0],
            )
            embed_parts, mask_parts, cursor = [], [], 0
            for position, replacement in replacements:
                embed_parts.append(prompt_embeddings[batch_index, cursor:position])
                mask_parts.append(prompt_attention[batch_index, cursor:position])
                embed_parts.append(replacement)
                mask_parts.append(
                    torch.ones(
                        replacement.shape[0],
                        dtype=prompt_attention.dtype,
                        device=device,
                    )
                )
                cursor = position + 1
            embed_parts.append(prompt_embeddings[batch_index, cursor:])
            mask_parts.append(prompt_attention[batch_index, cursor:])
            embedded_rows.append(torch.cat(embed_parts, dim=0))
            attention_rows.append(torch.cat(mask_parts, dim=0))
        return torch.stack(embedded_rows), torch.stack(attention_rows)

    def _run_language_model(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_attention: torch.Tensor,
        target_reports: Sequence[str],
        copy_token_ids: torch.Tensor,
        copy_token_embeddings: torch.Tensor,
        copy_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = prompt_embeddings.device
        reports = [report + self.end_sym for report in target_reports]
        target_tokens, target_embeddings, targets = self.training_input_generate(
            reports, prompt_attention.shape[1], device
        )
        batch_size = prompt_embeddings.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=target_tokens.input_ids.dtype,
            device=device,
        )
        inputs = torch.cat(
            [self.embed_tokens(bos), prompt_embeddings, target_embeddings], dim=1
        )
        attention = torch.cat(
            [
                torch.ones(
                    (batch_size, 1),
                    dtype=prompt_attention.dtype,
                    device=device,
                ),
                prompt_attention,
                target_tokens.attention_mask,
            ],
            dim=1,
        )
        outputs = self.llama_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            labels=targets,
            output_hidden_states=True,
            return_dict=True,
        )
        shifted_targets = targets[:, 1:]
        copy_attention = self.pointer_copy(
            outputs.hidden_states[-1][:, :-1],
            copy_token_embeddings,
            copy_attention_mask,
        )
        copy_loss = self.pointer_copy.loss(
            copy_attention,
            copy_token_ids,
            shifted_targets,
            shifted_targets.ne(-100),
        )
        return outputs.loss, copy_loss

    def _intervention_loss(
        self,
        program: EditProgramOutput,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        non_keep = targets.ne(int(EditOperation.KEEP))
        valid = non_keep.any(dim=1)
        if not valid.any():
            return program.program_slots.sum() * 0.0
        scores = torch.rand_like(targets, dtype=program.operation_probs.dtype)
        selected = scores.masked_fill(~non_keep, -1.0).argmax(dim=1)
        batch = torch.arange(targets.shape[0], device=targets.device)
        probabilities = program.operation_probs.clone()
        probabilities[batch[valid], selected[valid]] = 0.0
        probabilities[batch[valid], selected[valid], int(EditOperation.KEEP)] = 1.0
        slots = self.edit_program.compose(
            program.content_slots,
            probabilities,
            program.anatomy_probs,
            program.severity_probs,
        )
        execution = self.edit_executor(
            slots, probabilities, program.confidence, token_embeddings, token_mask
        )
        intervention_targets = targets.clone()
        intervention_targets[batch[valid], selected[valid]] = int(EditOperation.KEEP)
        return masked_intervention_loss(
            self.program_verifier(execution.executed_slots),
            intervention_targets,
            valid,
        )
