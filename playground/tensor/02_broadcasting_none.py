"""Demonstration of broadcasting with None dimension in PyTorch tensors."""

import torch

BATCH_SIZE = 1024
VOCAB_SIZE = 50000
logits = torch.randn(
    BATCH_SIZE, VOCAB_SIZE
)  # shape [1024, 50000]  # cspell:ignore randn

top_k = 10
kth_vals, _ = torch.topk(logits, top_k, dim=-1)
print("kth_vals shape:", kth_vals.shape)  # [1024, 10]
print("kth_vals[..., -1] shape:", kth_vals[..., -1].shape)  # [1024]
kth_vals_last_col = kth_vals[..., -1, None]
print("kth_vals[..., -1, None] shape:", kth_vals_last_col.shape)  # [1024, 1]

# Broadcasting comparison
mask = logits < kth_vals[..., -1, None]
print("mask shape:", mask.shape)  # [1024, 50000]

# Show a small sample for verification
print("Sample kth_vals[..., -1, None]:", kth_vals[:2, -1, None])
print("Sample mask:", mask[:2, :12])
