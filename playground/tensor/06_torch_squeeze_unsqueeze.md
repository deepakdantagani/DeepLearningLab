# torch.squeeze and torch.unsqueeze

Quick reference and examples for adding/removing size-1 dimensions.

## Summary
- `unsqueeze(dim)`: add a dimension of size 1 at index `dim`.
- `squeeze(dim=None)`: remove dimensions of size 1; if `dim` is given, remove that dim only if it is size 1.

## Why use them?
- Prepare shapes for broadcasting (e.g., match `[batch, seq, heads, d]`).
- Add batch/channel dims; clean up singleton dims after ops.

## Examples
```python
import torch

# 1) Unsqueeze examples
x = torch.tensor([1, 2, 3])      # [3]
print(x.unsqueeze(0).shape)      # [1, 3]
print(x.unsqueeze(1).shape)      # [3, 1]
print(x.unsqueeze(-1).shape)     # [3, 1]

# 2) Squeeze examples
y = torch.randn(1, 3, 1, 5)      # [1, 3, 1, 5]
print(y.squeeze().shape)         # [3, 5]
print(y.squeeze(0).shape)        # [3, 1, 5]
print(y.squeeze(2).shape)        # [1, 3, 5]

# 3) Broadcasting with RoPE-style shapes
seq, d = 10, 8
cos = torch.randn(seq, d)                 # [S, d]
cos_b = cos.unsqueeze(0).unsqueeze(2)     # [1, S, 1, d]
x = torch.randn(4, seq, 2, d)             # [B, S, H, d]
z = x * cos_b                             # broadcasted → [4, S, 2, d]
```

## Notes
- Negative dims work: `unsqueeze(-1)` appends, `unsqueeze(-2)` inserts before last.
- `squeeze(dim)` is a no-op if that dim is not size 1.

