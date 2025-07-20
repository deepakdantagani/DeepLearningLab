# Understanding `kth_vals` and Indexing Tricks

## What is `kth_vals`?
- `kth_vals` is the tensor of top-k values per batch, returned by `torch.topk` (sorted in descending order).
- Commonly used in top-k filtering for language models and other batch operations.

## Key Indexing Patterns
- `kth_vals[..., -1]`: Selects the smallest value among the top-k for each batch (last in descending order).
- `kth_vals[..., -1, None]`: Adds a new axis, turning the result into a column vector for broadcasting (shape `[batch_size, 1]`).

## Real-world Usage
- Used to determine the threshold for masking logits: all logits less than this value are excluded from sampling.
- Broadcasting with `kth_vals[..., -1, None]` ensures shape compatibility for elementwise comparison with the logits tensor.

## Example Code
See `01_kth_vals_last_none.py` for a runnable example. Here is a summary:

```python
import torch

logits = torch.tensor([
    [2.0, 1.0, 0.5, -1.0],
    [3.0, 2.5, 0.0, -2.0]
])  # shape [2, 4]

top_k = 2
kth_vals, _ = torch.topk(logits, top_k, dim=-1)
print('kth_vals:', kth_vals)  # shape [2, 2]
print('kth_vals[..., -1]:', kth_vals[..., -1])  # shape [2]
print('kth_vals[..., -1, None]:', kth_vals[..., -1, None])  # shape [2, 1]

# Broadcasting comparison
mask = logits < kth_vals[..., -1, None]
print('mask:', mask)
```

**Output:**
```
kth_vals: tensor([[2.0, 1.0],
                 [3.0, 2.5]])
kth_vals[..., -1]: tensor([1.0, 2.5])
kth_vals[..., -1, None]: tensor([[1.0],
                                [2.5]])
mask: tensor([[False, False, True, True],
              [False, False, True, True]])
```

## Math Dimensions
- **logits:** `[batch_size, vocab_size]` (e.g., `[2, 4]`)
- **kth_vals:** `[batch_size, k]` (e.g., `[2, 2]`)
- **kth_vals[..., -1]:** `[batch_size]` (e.g., `[2]`)
- **kth_vals[..., -1, None]:** `[batch_size, 1]` (e.g., `[2, 1]`)
- **mask:** `[batch_size, vocab_size]` (broadcasted)

## Visual/Diagram (ASCII Art)
```
kth_vals (batch, k)
        |
        | kth_vals[..., -1]
        v
kth_vals_last (batch)
        |
        | .unsqueeze(-1) or None
        v
kth_vals_last_col (batch, 1)
        |
        | compare: logits < kth_vals_last_col
        v
mask (batch, vocab)
```

## Summary Table
| Expression                | Meaning                                      | Shape         |
|---------------------------|----------------------------------------------|--------------|
| `kth_vals[..., 0]`        | Largest value in the top_k for each batch    | `[batch_size]`|
| `kth_vals[..., -1]`       | Smallest value in the top_k for each batch   | `[batch_size]`|
| `kth_vals[..., -1, None]` | Smallest value as column for broadcasting    | `[batch_size, 1]`|

---

**See also:** `01_kth_vals_last_none.py` for a runnable code example.
