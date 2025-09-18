from __future__ import annotations

import torch
from torch import Tensor


class RotaryCache:
    """Cache for rotary embeddings."""

    def __init__(self, dim: int, base: int = 10_000) -> None:
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.cos_cached: Tensor | None = None
        self.sin_cached: Tensor | None = None

    def build(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build the cache for rotary embeddings."""
        k = torch.arange(0, self.dim // 2, device=device, dtype=dtype)  # dimension: [dim // 2], Example: [0, 1, ..., self.dim // 2 - 1]
        inv_scale = 1.0 / (self.base ** (2 * k / self.dim))  # dimensions: [dim // 2], Example: [1.0, 1.0 / base^(2/dim), 1.0 / base^(4/dim), ..., 1.0 / base^((self.dim // 2 - 1) * 2 / self.dim)]
        idx = torch.arange(seq_len, device=device, dtype=dtype)  # dimensions: [seq_len], Example: [0, 1, ..., seq_len - 1]
        angles = idx * inv_scale  # dimensions: [seq_len], Example: [0.0, 1.0 / base^(2/dim), 2.0 / base^(4/dim), ..., (seq_len - 1) * 1.0 / base^((self.dim // 2 - 1) * 2 / self.dim)]

        self.cos_cached = torch.cos(angles)  # dimensions: [seq_len, dim // 2]
        self.sin_cached = torch.sin(angles)  # dimensions: [seq_len, dim // 2]
    
    def get(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Get the cached cos and sin values."""
        if self.cos_cached is None or 
