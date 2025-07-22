"""Logits processors for text generation decoding strategies."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel

from dllab.generation.filters import top_k_top_p_filtering

from .base import LogitsProcessor, SamplingStrategy


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
        """Initialize top-k and top-p filtering parameters."""
        self.k = k
        self.p = p

    def __call__(self, logits: Tensor, _: Tensor) -> Tensor:
        return top_k_top_p_filtering(logits, self.k, self.p)


class Decoder:
    """Main decoder class for text generation with configurable sampling strategies."""

    def __init__(
        self,
        model: PreTrainedModel,
        strategy: SamplingStrategy,
        logits_processors: Optional[List[LogitsProcessor]] = None,
    ) -> None:
        """Initialize the decoder with model, sampling strategy, and optional logits processors."""
        self.model = model
        self.strategy = strategy
        self.processors = logits_processors or []

    def generate(
        self, input_ids: Tensor, max_new_tokens: int = 32, eos_id: Optional[int] = None
    ) -> Tensor:
        """Generate text tokens using the configured model and sampling strategy."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.model(generated).logits[:, -1, :]

            for process in self.processors:
                logits = process(logits, generated)

            next_token = self.strategy.sample(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated
