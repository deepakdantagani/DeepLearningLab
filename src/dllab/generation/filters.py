"""Filtering utilities for top-k and top-p sampling in sequence generation."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

FILTER_VALUE: Final[float] = -float("inf")


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = FILTER_VALUE,
) -> Tensor:
    """
    Apply top-k and top-p (nucleus) filtering to logits.

    TOP-K FILTERING EXAMPLE:

    Input logits: [batch_size=2, vocab_size=4]
    [
        [1.2, 2.1, 0.5, 1.8],  # Batch 0
        [0.8, 2.5, 1.1, 0.3]   # Batch 1
    ]

    Step 1: Find top-k values (top_k = 2)
    k_values = [
        [2.1, 1.8],  # Top 2 values for Batch 0
        [2.5, 1.1]   # Top 2 values for Batch 1
    ]

    Step 2: Create condition mask
    condition = logits < k_values[..., -1, None]
    # k_values[..., -1] = [1.8, 1.1] (minimum of top-k)
    # condition = [
    #     [True, False, True, False],  # Batch 0: values < 1.8
    #     [True, False, False, True]   # Batch 1: values < 1.1
    # ]

    Step 3: Apply filter
    Final logits = [
        [-inf, 2.1, -inf, 1.8],  # Batch 0: keep top 2
        [-inf, 2.5, 1.1, -inf]   # Batch 1: keep top 2
    ]

    TOP-P FILTERING EXAMPLE:

    Input logits: [batch_size=2, vocab_size=4]
    [
        [1.2, 2.1, 0.5, 1.8],  # Batch 0
        [0.8, 2.5, 1.1, 0.3]   # Batch 1
    ]

    Step 1: Sort in descending order
    sorted_logits = [
        [2.1, 1.8, 1.2, 0.5],  # Batch 0
        [2.5, 1.1, 0.8, 0.3]   # Batch 1
    ]
    sorted_idx = [
        [1, 3, 0, 2],  # Original positions
        [1, 2, 0, 3]   # Original positions
    ]

    Step 2: Calculate cumulative probabilities
    cum_probs = [
        [0.35, 0.70, 0.85, 1.0],  # Batch 0
        [0.45, 0.75, 0.85, 1.0]   # Batch 1
    ]

    Step 3: Create mask (top_p = 0.8)
    mask = [
        [False, False, True, True],   # Positions > 0.8
        [False, False, True, True]    # Positions > 0.8
    ]

    Step 4: Shift mask to include cutoff token
    mask = [
        [False, False, False, True],  # Shifted right
        [False, False, False, True]   # Shifted right
    ]

    Step 5: Apply -inf filter
    sorted_logits = [
        [2.1, 1.8, 1.2, -inf],  # Last position filtered
        [2.5, 1.1, 0.8, -inf]   # Last position filtered
    ]

    Step 6: Scatter back to original positions
    Final logits = [
        [1.2, 2.1, 0.5, -inf],  # Position 3 filtered
        [0.8, 2.5, 1.1, -inf]   # Position 3 filtered
    ]

    The -inf values ensure zero probability after softmax.
    """
    if top_k > 0:
        k_values, _ = torch.topk(logits, top_k)  # [b , top_k]
        condition = logits < k_values[..., -1, None]  # [b, 1]
        logits = torch.where(condition, filter_value, logits)  # [b, v]

    if 0.0 < top_p < 1:
        sorted_logits, sorted_inx = torch.sort(
            logits, dim=-1, descending=True
        )  # [b, V]
        probs = torch.softmax(logits, dim=-1)  # [b, V]
        cum_probs = probs.cumsum(dim=-1)

        # remove token with cumulative prob above threshold
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        sorted_logits[mask] = filter_value
        logits.scatter_(dim=-1, index=sorted_inx, src=sorted_logits)

    return logits
