"""Logits processors for text generation strategies."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel

from dllab.generation.filters import top_k_top_p_filtering

from .base import LogitsProcessor, SamplingStrategy


class TemperatureLogitsProcessor(LogitsProcessor):
    """Apply temperature scaling to logits for controlled randomness."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def __call__(self, logits: Tensor, _: Tensor) -> Tensor:
        return logits / self.temperature


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


class Generator:
    """Main generator class for text generation with configurable sampling strategies."""

    def __init__(
        self,
        model: PreTrainedModel,
        strategy: SamplingStrategy,
        logits_processors: Optional[List[LogitsProcessor]] = None,
    ) -> None:
        """Initialize the generator with model, sampling strategy, and optional logits processors."""
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

            # Apply logits processors first (temperature, top-k, top-p, etc.)
            for process in self.processors:
                logits = process(logits, generated)

            # Then apply sampling strategy
            next_token = self.strategy.sample(logits)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated
