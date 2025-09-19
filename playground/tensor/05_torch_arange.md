# torch.arange — Playground

Quick hands-on snippets to understand `torch.arange`. Run the Python file to see outputs.

## What is `torch.arange`?

Returns a 1D tensor with values in `[start, end)` taking steps of `step`.

## Quick examples

```python
import torch

# End only
print(torch.arange(5))           # tensor([0, 1, 2, 3, 4])

# Start, end
print(torch.arange(2, 6))        # tensor([2, 3, 4, 5])

# Step (integer)
print(torch.arange(0, 10, 3))    # tensor([0, 3, 6, 9])

# Step (float)
print(torch.arange(0.0, 1.0, 0.2))  # 0.0 to <1.0 by 0.2

# Negative step
print(torch.arange(5, 0, -1))    # tensor([5, 4, 3, 2, 1])

# Device / dtype
print(torch.arange(0, 4, device="cpu", dtype=torch.int64))
```

## When to use `arange` vs `linspace`

- Use `arange` when you know the step size.
- Use `linspace` when you need an exact number of samples (especially with floats).

```python
print(torch.linspace(0.0, 1.0, steps=5))  # 5 values inclusive of end
```

## Run it

```bash
python playground/tensor/05_torch_arange.py
```


