"""
Abstract base interfaces for generation components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class LogitsProcessor(ABC):
    """Abstract base class for logits processing components."""

    @abstractmethod
    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        """Return *modified* logits."""


class SamplingStrategy(ABC):
    """Abstract base class for token sampling strategies."""

    @abstractmethod
    def sample(self, logits: Tensor) -> Tensor:
        """Return the next token id (shape: [B, 1])."""
