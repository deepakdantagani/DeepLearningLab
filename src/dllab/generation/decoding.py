"""Logits processors for text generation decoding strategies."""

from __future__ import annotations

import torch
from torch import Tensor

from dllab.generation.filters import top_k_top_p_filtering

from .base import LogitsProcessor


class RepetitionPenalty(LogitsProcessor):
    """Apply repetition penalty to reduce likelihood of repeated tokens."""

    def __init__(self, penalty: float = 1.1) -> None:
        if penalty < 1.0:
            raise ValueError("penalty must be >= 1.0")
        self.penalty = penalty

    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        for token_id in torch.unique(generated_ids):
            logits[..., token_id] /= self.penalty
        return logits


class TopKTopP(LogitsProcessor):
    """Apply top-k and top-p filtering to logits for controlled sampling."""

    def __init__(self, k: int = 0, p: float = 0.0):
        self.k = k
        self.p = p

    def __call__(self, logits: Tensor, _: Tensor) -> Tensor:
        return top_k_top_p_filtering(logits, self.k, self.p)
