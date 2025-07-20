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
    if top_k > 0:
        k_values, _ = torch.topk(logits, top_k)  # [b * top_k]
        condition = logits < k_values[..., -1, None]  # [b, 1]
        logits = torch.where(condition, filter_value, logits)  # [b*v]

    return logits
