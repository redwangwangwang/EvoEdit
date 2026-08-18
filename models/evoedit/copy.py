"""Pointer-style copy supervision for stable report content."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PointerCopyHead(nn.Module):
    """Attend from decoder states to prior-report tokens.

    The head is used as an auxiliary training objective, so standard Hugging
    Face beam search remains available without a custom generation loop.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_states: torch.Tensor,
        prior_states: torch.Tensor,
        prior_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.query(decoder_states)
        keys = self.key(prior_states)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(~prior_attention_mask[:, None, :].bool(), torch.finfo(scores.dtype).min)
        return scores.softmax(dim=-1)

    @staticmethod
    def loss(
        copy_attention: torch.Tensor,
        prior_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_targets = target_token_ids.clamp_min(0)
        matches = prior_token_ids[:, None, :].eq(safe_targets[:, :, None])
        copyable = matches.any(dim=-1) & target_mask.bool()
        if not copyable.any():
            return copy_attention.sum() * 0.0
        probability = (copy_attention * matches.to(copy_attention.dtype)).sum(dim=-1)
        return -probability.clamp_min(1e-8).log()[copyable].mean()
