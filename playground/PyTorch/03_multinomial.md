# PyTorch Multinomial Sampling

## Overview

`torch.multinomial` is a fundamental PyTorch function for sampling from probability distributions. It's widely used in text generation, reinforcement learning, and other probabilistic sampling scenarios.

## Function Signature

```python
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
```

## Key Parameters

- **`input`**: Probability distribution tensor (must be non-negative and sum to 1)
- **`num_samples`**: Number of samples to draw
- **`replacement`**: Whether to sample with replacement (default: False)

## What It Returns

`torch.multinomial` returns **indices** (positions) from the input tensor, not the probability values themselves.

## Examples

### Basic Usage

```python
import torch

# Probability distribution for 5 tokens
probs = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])

# Sample 1 token
sample = torch.multinomial(probs, num_samples=1)
print(sample)  # tensor([2]) - index of selected token
```

### Temperature Sampling

```python
# Original logits from model
logits = torch.tensor([2.0, 1.0, 3.0, 0.5, 1.5])

# Apply temperature scaling
temperature = 1.0
scaled_logits = logits / temperature
probs = torch.softmax(scaled_logits, dim=-1)

# Sample next token
next_token_idx = torch.multinomial(probs, num_samples=1)
```

### Batch Processing

```python
# Batch of probability distributions
batch_probs = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],  # Sequence 1
    [0.4, 0.3, 0.2, 0.1],  # Sequence 2
])

# Sample 1 token per sequence
batch_samples = torch.multinomial(batch_probs, num_samples=1)
# Returns tensor([[3], [0]]) - shape [batch_size, num_samples]
```

## Key Concepts

### 1. Sampling vs Greedy Selection

- **Multinomial**: Random sampling based on probabilities
- **Argmax**: Always selects the highest probability token

```python
probs = torch.tensor([0.05, 0.1, 0.6, 0.15, 0.1])

# Greedy (deterministic)
greedy = torch.argmax(probs)  # Always returns 2

# Multinomial (stochastic)
sample = torch.multinomial(probs, num_samples=1)  # Can return any index
```

### 2. Temperature Scaling

Temperature controls the randomness of sampling:

- **Low temperature (< 1.0)**: More concentrated, deterministic
- **High temperature (> 1.0)**: More uniform, random

```python
def temperature_sample(logits, temperature):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 3. Replacement vs Non-replacement

- **With replacement**: Can sample the same token multiple times
- **Without replacement**: Each token can only be sampled once

```python
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

# With replacement
samples = torch.multinomial(probs, num_samples=3, replacement=True)
# Can return [3, 3, 2] - token 3 sampled twice

# Without replacement
samples = torch.multinomial(probs, num_samples=3, replacement=False)
# Returns [3, 2, 1] - each token sampled once
```

## Common Use Cases

### 1. Text Generation

```python
def generate_next_token(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 2. Reinforcement Learning

```python
def select_action(action_probs):
    return torch.multinomial(action_probs, num_samples=1)
```

### 3. Data Augmentation

```python
def sample_with_weights(data, weights):
    indices = torch.multinomial(weights, num_samples=len(data))
    return data[indices]
```

## Important Notes

1. **Input validation**: Probabilities must be non-negative and sum to 1
2. **Numerical stability**: Use `torch.softmax` to convert logits to probabilities
3. **Temperature bounds**: Avoid very small temperatures (< 1e-5) to prevent numerical issues
4. **Batch dimensions**: The function works with any number of batch dimensions

## Related Operations

- `torch.softmax`: Convert logits to probabilities
- `torch.argmax`: Greedy selection (deterministic)
- `torch.categorical`: Alternative sampling function
- `torch.distributions.Categorical`: More advanced categorical distribution

## Performance Considerations

- **Memory efficient**: Works in-place when possible
- **GPU friendly**: Optimized for CUDA tensors
- **Batch processing**: Efficient for multiple sequences
- **Gradient computation**: Supports autograd for training

## Troubleshooting

### Common Issues

1. **Invalid probabilities**: Ensure input sums to 1 and is non-negative
2. **Numerical instability**: Check for very small temperature values
3. **Shape mismatches**: Verify tensor dimensions match expected shapes
4. **Device mismatch**: Ensure tensors are on the same device (CPU/GPU)

### Debugging Tips

```python
# Check probability distribution
print(f"Sum: {probs.sum():.6f}")
print(f"Min: {probs.min():.6f}")
print(f"Max: {probs.max():.6f}")

# Verify sampling behavior
samples = torch.multinomial(probs, num_samples=1000)
unique, counts = torch.unique(samples, return_counts=True)
empirical_probs = counts.float() / 1000
print(f"Empirical vs theoretical: {empirical_probs} vs {probs}")
```
