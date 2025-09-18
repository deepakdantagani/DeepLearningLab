# `torch.arange`

Generate evenly spaced values in a half-open interval `[start, end)` with a given `step`.

## Signature

```python
torch.arange(start=0, end, step=1, *, dtype=None, layout=torch.strided,
             device=None, requires_grad=False) -> Tensor
```

- **start**: first value (default `0` if only `end` is provided)
- **end**: one-past-the-last value (exclusive)
- **step**: increment (can be float); negative allowed
- **dtype**: inferred from `start`, `end`, `step` unless specified
- **device**: target device (`"cpu"`, `"cuda"`, etc.)

## Behavior

- Produces a 1D tensor like Python `range`, but returns a Tensor.
- Interval is half-open: includes `start`, excludes `end`.
- Floating `step` is supported (beware accumulation error). For a fixed number of steps, prefer `torch.linspace`.
- Negative steps count downward: `torch.arange(5, 0, -1) -> [5, 4, 3, 2, 1]`.

## Examples

```python
import torch

# 1) End only
x = torch.arange(5)              # tensor([0, 1, 2, 3, 4])

# 2) Start, end
y = torch.arange(2, 6)           # tensor([2, 3, 4, 5])

# 3) Step (integer)
z = torch.arange(0, 10, 3)       # tensor([0, 3, 6, 9])

# 4) Step (float) + dtype
w = torch.arange(0.0, 1.0, 0.2)  # tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])

# 5) Negative step
r = torch.arange(5, 0, -2)       # tensor([5, 3, 1])

# 6) Device and dtype
c = torch.arange(0, 4, device="cpu", dtype=torch.int64)  # tensor([0,1,2,3])
```

## `arange` vs `linspace`

- **`arange(start, end, step)`**: control step size; number of elements is implied.
- **`linspace(start, end, steps)`**: control the number of samples; step size is implied. Use for precise sample counts (especially with floats).

```python
# 5 values from 0 to 1 inclusive
torch.linspace(0.0, 1.0, steps=5)  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

## Pitfalls

- Floating-point steps can accumulate error; you may miss `end` unexpectedly.
- With `step=0` a runtime error is raised.
- If `end <= start` and `step > 0` (or vice versa), result is empty.

## See also

- `torch.linspace` — evenly spaced numbers over an interval by count
- `torch.range` — deprecated; prefer `torch.arange`


