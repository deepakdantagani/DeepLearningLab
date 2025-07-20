# Understanding `torch.sort`

## Definition

`torch.sort` sorts the elements of a tensor along a specified dimension.

**Signature:**

```python
sorted_tensor, indices = torch.sort(input, dim=-1, descending=False)
```

- **input**: The tensor to sort
- **dim**: The dimension along which to sort (default: last dimension, `-1`)
- **descending**: If `True`, sorts in descending order (default: `False`)

## What does it return?

A tuple:

1. **sorted_tensor**: The sorted values
2. **indices**: The indices of the elements in the original tensor that gave the sorted order

## Mathematical Explanation

Suppose you have a tensor \( A \) of shape \([N, M]\):

- If you call `torch.sort(A, dim=1)`, for each row \( i \), you get:
  - `sorted_tensor[i, :]` is the sorted version of `A[i, :]`
  - `indices[i, :]` tells you where each sorted value came from in the original row

**Formally:**
\[
\text{sorted\_tensor}[i, j] = A[i, \text{indices}[i, j]]
\]
for all \( i \) and \( j \).

## Example

```python
import torch

x = torch.tensor([[3, 1, 2], [9, 7, 8]])
sorted_x, indices = torch.sort(x, dim=1)
print("sorted_x:\n", sorted_x)
print("indices:\n", indices)
```

**Output:**

```python
sorted_x:
 tensor([[1, 2, 3],
         [7, 8, 9]])
indices:
 tensor([[1, 2, 0],
         [1, 2, 0]])
```

## Key Points

- **Stable sort**: If two elements are equal, their order is preserved.
- **Works on any dimension**: You can sort along rows, columns, or any axis.
- **Indices for reordering**: The `indices` output lets you recover the original order or apply the same sort to other tensors.

## Common Uses

- Sorting scores, logits, or predictions
- Getting the top-k or bottom-k elements (for top-k, use `torch.topk`)
- Ranking or ordering data in batches

## Summary Table

| Argument      | Meaning                                 |
|---------------|-----------------------------------------|
| `input`       | The tensor to sort                      |
| `dim`         | Dimension to sort along                 |
| `descending`  | Sort in descending order if True        |
| **Returns**   | `(sorted_tensor, indices)`              |
