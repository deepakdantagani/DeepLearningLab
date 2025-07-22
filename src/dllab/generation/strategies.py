"""Sampling strategies for text generation."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import SamplingStrategy


class GreedyStrategy(SamplingStrategy):
    """
    Greedy strategy: select the token with the highest probability.
    Note: argmax returns the index of the maximum value along the specified dimension.
    """

    def sample(self, logits: Tensor) -> Tensor:  # [B, S, V]
        return torch.argmax(
            logits, dim=-1, keepdim=True
        )  # [B, S, 1]  # cspell:ignore keepdim


class TemperatureSampling(SamplingStrategy):
    """
    Temperature sampling strategy: divide the logits by the temperature and then add get the prods
    Note: torch.multinomial returns the indices of the sampled tokens.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def sample(self, logits: Tensor) -> Tensor:  # [B, V]
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # cspell:ignore multinomial


class MultinomialSampling(SamplingStrategy):
    """Multinomial sampling strategy: sample from the probability distribution."""

    def sample(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # cspell:ignore multinomial
