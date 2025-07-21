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
