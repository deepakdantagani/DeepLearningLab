"""
Abstract base interfaces for decoding components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class LogitsProcessor(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        pass


class SamplingStrategy(ABC):
    @abstractmethod
    def __call__(self, logits: Tensor) -> Tensor:
        pass
