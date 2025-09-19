# PyTorch Notes

## 1. What is `Tensor`?
- `Tensor` is a data structure from the PyTorch library used for multi-dimensional arrays and numerical computations.
- In code: `from torch import Tensor`
- Used for type annotations in function signatures, e.g., `def foo(x: Tensor) -> Tensor:`
- Represents the primary way to store and manipulate data in PyTorch.

---

## 2. What is `torch.topk`?

**Definition:**
- `torch.topk` returns the `k` largest elements of the input tensor along a given dimension.
- It returns both the values and their indices.

**Real-world usage:**
- Commonly used in language models to select the top-k most likely tokens from logits for each item in a batch (e.g., in top-k sampling for text generation).
- Useful for focusing computation or sampling on the most probable candidates.

**Sample code:**
```python
import torch

logits = torch.randn(1024, 50000)  # Example: batch size 1024, vocab size 50,000
k = 10
values, indices = torch.topk(logits, k, dim=-1)
print(values.shape)   # torch.Size([1024, 10])
print(indices.shape)  # torch.Size([1024, 10])
```

**Math dimensions:**
- **Input:** `[batch_size, vocab_size]` (e.g., `[1024, 50000]`)
- **Output:**
  - `values`: `[batch_size, k]` (top-k values for each batch item)
  - `indices`: `[batch_size, k]` (indices of top-k values in the original tensor)

---

## 3. What is `torch.where`?

**Definition:**
- `torch.where` returns a tensor of elements selected from either of two tensors, depending on a condition.
- For each element, if the condition is true, the result is taken from the first tensor; otherwise, from the second tensor.

**Real-world usage:**
- Used for masking, filtering, or conditional replacement in tensors.
- Common in top-k/top-p sampling to mask out unwanted logits (e.g., set to `-inf`).
- Can be used for element-wise conditional logic in neural network operations.

**Sample code:**
```python
import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10, 20, 30, 40])
condition = a > 2
result = torch.where(condition, a, b)
print(result)  # tensor([10, 20,  3,  4])
```

**Math dimensions:**
- **Input:**
  - `condition`: any shape (e.g., `[N]`, `[batch, vocab]`)
  - `x`, `y`: same shape as `condition` (or broadcastable)
- **Output:**
  - Same shape as `condition`, with elements from `x` where `condition` is true, else from `y`.

---

## 4. What are `torch.squeeze` and `torch.unsqueeze`?

**Definitions:**
- `torch.unsqueeze(t, dim)` adds a new dimension of size 1 at index `dim`.
- `torch.squeeze(t, dim=None)` removes dimensions of size 1. If `dim` is given, removes that dimension only if its size is 1.

**Why they matter:**
- Shape alignment for broadcasting (e.g., preparing `[seq, d]` to multiply with `[batch, seq, heads, d]`).
- Adding batch or channel dimensions.
- Cleaning up extra singleton dimensions after ops.

**Syntax:**
```python
out = t.unsqueeze(dim)
out = torch.unsqueeze(t, dim)

out = t.squeeze()        # remove all size-1 dims
out = t.squeeze(dim)     # remove that dim if size-1
```

**Examples:**
```python
import torch

# 1) Unsqueeze: add dimensions
x = torch.tensor([1, 2, 3])      # [3]
x0 = x.unsqueeze(0)              # [1, 3]
x1 = x.unsqueeze(1)              # [3, 1]
x_last = x.unsqueeze(-1)         # [3, 1]

# 2) Squeeze: remove size-1 dimensions
y = torch.randn(1, 3, 1, 5)      # [1, 3, 1, 5]
y_all = y.squeeze()              # [3, 5]
y_keep = y.squeeze(0)            # [3, 1, 5]
y_try = y.squeeze(2)             # [1, 3, 5] (dim 2 removed)

# 3) Broadcasting prep (common pattern)
cos = torch.randn(10, 8)         # [seq=10, d=8]
cos_b = cos.unsqueeze(0).unsqueeze(2)  # [1, 10, 1, 8]
x = torch.randn(4, 10, 2, 8)     # [batch=4, seq=10, heads=2, d=8]
z = x * cos_b                    # broadcasted elementwise multiply → [4, 10, 2, 8]
```

**Tips:**
- Use negative indices: `unsqueeze(-1)` adds a trailing dimension; `unsqueeze(-2)` adds before last.
- `squeeze` will not remove non-1 dimensions; specify `dim` to be explicit.
