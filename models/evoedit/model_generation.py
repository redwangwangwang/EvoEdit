"""Generation and audit-trail mixin for EvoEdit."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import torch

from .constants import (
    ANATOMY_CODE_NAMES,
    FINDINGS,
    OPERATION_NAMES,
    SEVERITY_CODE_NAMES,
)


class EvoEditGenerationMixin:
    @staticmethod
    def _safe_code_name(names: Sequence[str], index: int, prefix: str) -> str:
        return names[index] if index < len(names) else f"{prefix}_{index}"

    def _serialize_program(self, program, execution) -> list[list[dict[str, Any]]]:
        operation_ids = program.operation_probs.argmax(dim=-1).detach().cpu()
        anatomy_ids = program.anatomy_probs.argmax(dim=-1).detach().cpu()
        severity_ids = program.severity_probs.argmax(dim=-1).detach().cpu()
        confidence = program.confidence.squeeze(-1).detach().float().cpu()
        preserve = execution.preserve_gate.squeeze(-1).detach().float().cpu()
        result = []
        for batch_index in range(operation_ids.shape[0]):
            sample = []
            for finding_index, finding in enumerate(FINDINGS):
                operation = int(operation_ids[batch_index, finding_index])
                anatomy = int(anatomy_ids[batch_index, finding_index])
                severity = int(severity_ids[batch_index, finding_index])
                sample.append(
                    {
                        "finding": finding,
                        "operation": OPERATION_NAMES[operation],
                        "anatomy_code": self._safe_code_name(
                            ANATOMY_CODE_NAMES, anatomy, "anatomy"
                        ),
                        "severity_code": self._safe_code_name(
                            SEVERITY_CODE_NAMES, severity, "severity"
                        ),
                        "confidence": round(float(confidence[batch_index, finding_index]), 6),
                        "preserve_gate": round(float(preserve[batch_index, finding_index]), 6),
                    }
                )
            result.append(sample)
        return result

    def _generate_current(self, samples: dict[str, Any], split: str):
        self.llama_tokenizer.padding_side = "right"
        references_tokens = self.llama_tokenizer(
            samples["curr_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        )
        previous_visual, current_visual = self._encode_pair(samples)
        program, execution, _, _, _ = self._build_direction(
            previous_visual, current_visual, samples["prev_text"]
        )
        prompt, prompt_mask = self.prompt_wrap(
            current_visual, execution.executed_slots, samples["prev_text"], "curr"
        )
        batch_size = current_visual.shape[0]
        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=references_tokens.input_ids.dtype,
            device=current_visual.device,
        )
        inputs = torch.cat([self.embed_tokens(bos), prompt], dim=1)
        attention = torch.cat(
            [
                torch.ones(
                    (batch_size, 1),
                    dtype=prompt_mask.dtype,
                    device=current_visual.device,
                ),
                prompt_mask,
            ],
            dim=1,
        )
        temperature = self.hparams.temperature if self.hparams.temperature > 0 else None
        outputs = self.llama_model.generate(
            inputs_embeds=inputs,
            attention_mask=attention,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            num_beam_groups=self.hparams.num_beam_groups,
            do_sample=self.hparams.do_sample,
            no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            diversity_penalty=self.hparams.diversity_penalty,
            temperature=temperature,
        )
        hypotheses = [self.decode(output) for output in outputs]
        references = [self.decode(output) for output in references_tokens.input_ids]
        programs = self._serialize_program(program, execution)
        records = [
            {"id": sample_id, "program": sample_program}
            for sample_id, sample_program in zip(samples["id"], programs)
        ]
        output_record = {"hypo": hypotheses, "ref": references, "id": samples["id"]}
        if split == "val":
            self.val_step_outputs.append(output_record)
            self.val_edit_outputs.extend(records)
        else:
            self.test_step_outputs.append(output_record)
            self.test_edit_outputs.extend(records)
        return hypotheses, references

    def validation_step(self, samples: dict[str, Any], batch_idx: int):
        return self._generate_current(samples, "val")

    def test_step(self, samples: dict[str, Any], batch_idx: int):
        return self._generate_current(samples, "test")

    def _write_programs(self, filename: str, records: list[dict[str, Any]]) -> None:
        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        with open(os.path.join(result_folder, filename), "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.local_rank == 0:
            self._write_programs(
                f"edit_programs_{self.trainer.current_epoch}_{self.trainer.global_step}.json",
                self.val_edit_outputs,
            )
        self.val_edit_outputs.clear()
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._write_programs(
            f"test_edit_programs_localrank{self.local_rank}.json",
            self.test_edit_outputs,
        )
        self.test_edit_outputs.clear()
        super().on_test_epoch_end()
