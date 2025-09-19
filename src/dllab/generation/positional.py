"""Cache for rotary embeddings."""

from __future__ import annotations

from typing import Tuple

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
        m = torch.arange(seq_len, device=device, dtype=dtype)  # dimensions: [seq_len], Example: [0, 1, ..., seq_len - 1]

        angles = m * inv_scale  # dimensions: [seq_len], Example: [0.0, 1.0 / base^(2/dim), 2.0 / base^(4/dim), ..., (seq_len - 1) * 1.0 / base^((self.dim // 2 - 1) * 2 / self.dim)]

        self.cos_cached = torch.cos(angles)  # dimensions: [seq_len, dim // 2]
        self.sin_cached = torch.sin(angles)  # dimensions: [seq_len, dim // 2]

    def get(self, seq_len: int, device=None, dtype=None) -> Tuple[Tensor, Tensor]:
        """Get the cached cos and sin values."""
        if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:  # seq_len > self.cos_cached.shape[0] : seq_len is greater than the cached cos and sin values then build the new cos and sin
            self.build(seq_len, device=device or torch.device("cpu"), dtype=dtype or torch.float32)

        if self.cos_cached is None or self.sin_cached is None:
            raise RuntimeError("RotaryCache not built properly: cos_cached or sin_cached is None.")

        # only return the cached cos and sin values for the first seq_len tokens. if seq_len is less than the cached values then return the cached cos and sin values for the first seq_len tokens.
        # if seq_len is greater than it is handled above in the get function.
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),  # dimensions: [seq_len, dim // 2]
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),  # dimensions: [seq_len, dim // 2]
        )


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to the input tensor.

    [cos(θ)  -sin(θ)] [x_even]   [x_even*cos - x_odd*sin]
    [sin(θ)   cos(θ)] [x_odd ] = [x_even*sin + x_odd*cos]
    """
    x_even, x_odd = x[..., ::2], x[..., 1::2]  # each: [B, S, H, dim/2]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, dim/2]

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    x[..., ::2] = rot_even  # [1, s, 1, dim/2]
    x[..., 1::2] = rot_odd  # [1, s, 1, dim/2]

    return x  # [B, S, H, dim]


# Notes: shape flow for apply_rotary
# ----------------------------------
# Inputs:
#   x   : [B, S, H, D]            (D must be even)
#   cos : [S, D/2]
#   sin : [S, D/2]
# Steps:
# 1) Split even/odd channels along last dim
#    x_even = x[..., ::2] -> [B, S, H, D/2]
#    x_odd  = x[..., 1::2] -> [B, S, H, D/2]
# 2) Prepare cos/sin for broadcasting over batch and heads
#    cos = cos.unsqueeze(0).unsqueeze(2) -> [1, S, 1, D/2]
#    sin = sin.unsqueeze(0).unsqueeze(2) -> [1, S, 1, D/2]
#    (These broadcast to [B, S, H, D/2] when multiplied with x_even/x_odd)
# 3) Apply 2D rotation per (even, odd) pair at each position/frequency
#    rot_even = x_even * cos - x_odd * sin   -> [B, S, H, D/2]
#    rot_odd  = x_even * sin + x_odd * cos   -> [B, S, H, D/2]
# 4) Interleave back
#    x[..., ::2] = rot_even; x[..., 1::2] = rot_odd -> x: [B, S, H, D]
