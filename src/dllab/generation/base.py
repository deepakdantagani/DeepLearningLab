"""
Abstract base interfaces for decoding components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        """Return *modified* logits."""
        pass


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, logits: Tensor) -> Tensor:
        """Return the next token id (shape: [B, 1])."""
        pass
