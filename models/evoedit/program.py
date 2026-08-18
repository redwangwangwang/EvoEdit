"""Neural components for factorized and executable clinical edit programs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EditOperation


@dataclass
class CorrespondenceOutput:
    change_tokens: torch.Tensor
    aligned_previous: torch.Tensor
    attention: torch.Tensor


@dataclass
class EditProgramOutput:
    content_slots: torch.Tensor
    program_slots: torch.Tensor
    operation_logits: torch.Tensor
    operation_probs: torch.Tensor
    anatomy_logits: torch.Tensor
    anatomy_probs: torch.Tensor
    severity_logits: torch.Tensor
    severity_probs: torch.Tensor
    confidence: torch.Tensor
    slot_attention: torch.Tensor


@dataclass
class ExecutorOutput:
    executed_slots: torch.Tensor
    prior_facts: torch.Tensor
    preserve_gate: torch.Tensor
    prior_attention: torch.Tensor


class SoftTemporalCorrespondence(nn.Module):
    """Softly align historical visual tokens to current visual tokens."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.previous_norm = nn.LayerNorm(hidden_size)
        self.current_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, previous_tokens: torch.Tensor, current_tokens: torch.Tensor) -> CorrespondenceOutput:
        if previous_tokens.ndim != 3 or current_tokens.ndim != 3:
            raise ValueError("Visual tokens must have shape [batch, tokens, hidden].")
        if previous_tokens.shape[0] != current_tokens.shape[0]:
            raise ValueError("The two timepoints must use the same batch size.")
        if previous_tokens.shape[-1] != self.hidden_size or current_tokens.shape[-1] != self.hidden_size:
            raise ValueError("Visual token width does not match hidden_size.")

        previous = self.previous_norm(previous_tokens)
        current = self.current_norm(current_tokens)
        queries = self.query(current)
        keys = self.key(previous)
        values = self.value(previous)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        attention = scores.softmax(dim=-1)
        aligned_previous = torch.matmul(attention, values)
        change_tokens = self.fusion(
            torch.cat(
                [
                    current,
                    aligned_previous,
                    current - aligned_previous,
                    current * aligned_previous,
                ],
                dim=-1,
            )
        )
        return CorrespondenceOutput(change_tokens, aligned_previous, attention)


class FactorizedEditProgram(nn.Module):
    """Convert temporal change tokens into finding-wise edit slots."""

    def __init__(
        self,
        hidden_size: int,
        num_findings: int,
        num_operations: int,
        num_anatomy_codes: int,
        num_severity_codes: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_findings = num_findings
        self.num_operations = num_operations
        self.temperature = temperature

        self.finding_queries = nn.Parameter(torch.empty(num_findings, hidden_size))
        nn.init.normal_(self.finding_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.content_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        self.operation_head = nn.Linear(hidden_size, num_operations)
        self.anatomy_head = nn.Linear(hidden_size, num_anatomy_codes)
        self.severity_head = nn.Linear(hidden_size, num_severity_codes)
        self.confidence_head = nn.Linear(hidden_size, 1)
        self.operation_codebook = nn.Embedding(num_operations, hidden_size)
        self.anatomy_codebook = nn.Embedding(num_anatomy_codes, hidden_size)
        self.severity_codebook = nn.Embedding(num_severity_codes, hidden_size)
        self.composition_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _expected_embedding(probabilities: torch.Tensor, codebook: nn.Embedding) -> torch.Tensor:
        return probabilities @ codebook.weight

    def compose(
        self,
        content_slots: torch.Tensor,
        operation_probs: torch.Tensor,
        anatomy_probs: torch.Tensor,
        severity_probs: torch.Tensor,
    ) -> torch.Tensor:
        composed = (
            content_slots
            + self._expected_embedding(operation_probs, self.operation_codebook)
            + self._expected_embedding(anatomy_probs, self.anatomy_codebook)
            + self._expected_embedding(severity_probs, self.severity_codebook)
        )
        return self.composition_norm(composed)

    def forward(self, change_tokens: torch.Tensor) -> EditProgramOutput:
        batch_size = change_tokens.shape[0]
        queries = self.finding_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention = self.cross_attention(
            query=self.query_norm(queries),
            key=change_tokens,
            value=change_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        content_slots = self.output_norm(queries + attended)
        content_slots = self.output_norm(content_slots + self.content_ffn(content_slots))

        operation_logits = self.operation_head(content_slots)
        anatomy_logits = self.anatomy_head(content_slots)
        severity_logits = self.severity_head(content_slots)
        operation_probs = F.softmax(operation_logits / self.temperature, dim=-1)
        anatomy_probs = F.softmax(anatomy_logits, dim=-1)
        severity_probs = F.softmax(severity_logits, dim=-1)
        confidence = torch.sigmoid(self.confidence_head(content_slots))
        program_slots = self.compose(content_slots, operation_probs, anatomy_probs, severity_probs)
        return EditProgramOutput(
            content_slots=content_slots,
            program_slots=program_slots,
            operation_logits=operation_logits,
            operation_probs=operation_probs,
            anatomy_logits=anatomy_logits,
            anatomy_probs=anatomy_probs,
            severity_logits=severity_logits,
            severity_probs=severity_probs,
            confidence=confidence,
            slot_attention=attention,
        )


class CopyAndEditExecutor(nn.Module):
    """Preserve stable prior facts and inject only supported edit content."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.prior_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        program_slots: torch.Tensor,
        operation_probs: torch.Tensor,
        confidence: torch.Tensor,
        prior_token_embeddings: torch.Tensor,
        prior_attention_mask: torch.Tensor,
    ) -> ExecutorOutput:
        if prior_attention_mask.dtype != torch.bool:
            prior_attention_mask = prior_attention_mask.bool()
        prior_facts, attention = self.prior_attention(
            query=program_slots,
            key=prior_token_embeddings,
            value=prior_token_embeddings,
            key_padding_mask=~prior_attention_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        keep_probability = operation_probs[..., int(EditOperation.KEEP)].unsqueeze(-1)
        learned_gate = torch.sigmoid(
            self.gate(torch.cat([program_slots, prior_facts, keep_probability, confidence], dim=-1))
        )
        preserve_gate = 0.5 * learned_gate + 0.5 * keep_probability
        executed = preserve_gate * prior_facts + (1.0 - preserve_gate) * program_slots
        executed = self.output_norm(executed + program_slots)
        return ExecutorOutput(executed, prior_facts, preserve_gate, attention)
