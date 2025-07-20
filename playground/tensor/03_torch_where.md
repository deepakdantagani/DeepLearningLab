# Understanding `torch.where`

## Definition
- `torch.where` is a conditional selection function in PyTorch.
- It returns elements chosen from two tensors (or values) depending on a condition tensor.
- Signature: `torch.where(condition, x, y)`

## How it works
- For each element, if `condition` is `True`, the result is taken from `x`; otherwise, from `y`.
- All three arguments must be broadcastable to the same shape.

## Example Usage
```python
import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10, 20, 30, 40])
condition = a > 2
result = torch.where(condition, a, b)
print(result)  # tensor([10, 20, 3, 4])
```
- For `a > 2`, the condition is `[False, False, True, True]`.
- So, result is `[b[0], b[1], a[2], a[3]]` → `[10, 20, 3, 4]`.

## Broadcasting Example
```python
x = torch.arange(6).reshape(2, 3)
mask = x % 2 == 0
out = torch.where(mask, x, torch.tensor(-1))
print(out)
# tensor([[ 0, -1,  2],
#         [-1,  4, -1]])
```

## Common Patterns
- Replacing values based on a mask (e.g., set all negatives to zero):
  ```python
  x = torch.tensor([-2, -1, 0, 1, 2])
  y = torch.where(x < 0, torch.tensor(0), x)
  # y: tensor([0, 0, 0, 1, 2])
  ```
- Selecting between two tensors elementwise:
  ```python
  a = torch.tensor([1, 2, 3])
  b = torch.tensor([9, 8, 7])
  cond = torch.tensor([True, False, True])
  out = torch.where(cond, a, b)  # tensor([1, 8, 3])
  ```

## Summary Table
| Argument         | Meaning                                  |
|-----------------|-------------------------------------------|
| `condition`     | Boolean tensor (mask)                     |
| `x`             | Values where condition is True            |
| `y`             | Values where condition is False           |

## Notes
- If only `condition` is provided (no `x`, `y`), returns indices where condition is True (like `np.where`).
- Useful for masking, thresholding, and conditional logic in tensor operations.
