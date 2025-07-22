# Repetition Penalty

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Step-by-Step Process](#step-by-step-process)
4. [When to Use Repetition Penalty](#when-to-use-repetition-penalty)
5. [Limitations](#limitations)
6. [Comparison with Other Strategies](#comparison-with-other-strategies)
7. [Implementation](#implementation)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

Repetition penalty is a **logits processing technique** that reduces the probability of tokens that have already been generated in the current sequence. It helps prevent repetitive text generation by scaling down logits for previously seen tokens.

**Key Characteristics:**

- **Prevents loops**: Avoids infinite repetition of the same tokens
- **Encourages diversity**: Makes novel tokens more likely to be selected
- **Compatible**: Works with all sampling strategies (greedy, temperature, top-k, top-p)
- **Configurable**: Adjustable penalty strength via hyperparameter

---

## Mathematical Foundation

### Final Formula

**Let:**

- $z_i$: original logit for token $i$
- $r > 1$: repetition penalty factor (commonly 1.1 to 2.0)
- $S$: set of tokens already generated in the sequence

**Then:**
$$
z_i' = \begin{cases}
\frac{z_i}{r} & \text{if } i \in S \\
z_i & \text{if } i \notin S
\end{cases}
$$

### Formula Breakdown

1. **Token Identification**: Check if token $i$ exists in previously generated sequence $S$
2. **Penalty Application**: If seen before, divide logit by penalty factor $r$
3. **No Change**: If not seen before, leave logit unchanged
4. **Effect**: Reduced logits lead to lower probabilities after softmax

---

## Step-by-Step Process

### Algorithm Flow

1. **Get current logits** for next token generation

   ```
   logits ∈ ℝ^{B × |V|}
   ```

2. **Extract unique tokens** from generated sequence

   ```
   unique_tokens = unique(generated_ids)
   ```

3. **Apply penalty** to previously seen tokens

   ```
   for token_id in unique_tokens:
       logits[:, token_id] /= penalty_factor
   ```

4. **Continue with normal sampling** (softmax + sample)

   ```
   probs = softmax(logits)
   next_token = sample(probs)
   ```

### PyTorch Implementation

```python
class RepetitionPenalty(LogitsProcessor):
    """Scale down logits for tokens already generated (Ctrl-style penalty)."""

    def __init__(self, penalty: float = 1.1) -> None:
        if penalty < 1.0:
            raise ValueError("penalty must be >= 1.0")
        self.penalty = penalty

    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        for token_id in torch.unique(generated_ids):
            logits[:, token_id] /= self.penalty
        return logits
```

### Usage Example

```python
# Initialize with penalty factor
repetition_penalty = RepetitionPenalty(penalty=1.2)

# During generation loop
logits = model(input_ids)  # [B, V]
logits = repetition_penalty(logits, generated_ids)  # Apply penalty
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

---

## When to Use Repetition Penalty

### ✅ Recommended Scenarios

- **Long text generation**: Prevents repetitive patterns in extended sequences
- **Creative writing**: Encourages diverse vocabulary usage
- **High-temperature sampling**: Balances diversity with coherence
- **All sampling strategies**: Compatible with greedy, temperature, top-k, top-p
- **Beam search**: Avoids beams that repeat subphrases

### ❌ When to Avoid

- **Short responses**: May not be necessary for brief outputs
- **Very low penalty values**: Values close to 1.0 have minimal effect
- **Very high penalty values**: Can make generation too random

---

## Limitations

### Current Limitations

1. **Token-level only**: Doesn't prevent phrase-level repetition
2. **No frequency consideration**: Treats single and multiple occurrences equally
3. **Global penalty**: Same penalty applied to all previously seen tokens
4. **No context awareness**: Doesn't consider semantic similarity

### Potential Issues

- **Over-penalization**: Very high penalty values can hurt coherence
- **Vocabulary bias**: May favor rare tokens over common ones
- **No adaptive penalty**: Fixed penalty regardless of repetition frequency

---

## Comparison with Other Strategies

| Strategy | Purpose | Effect | Compatibility |
|----------|---------|--------|---------------|
| **Repetition Penalty** | Prevent token repetition | Reduces logits of seen tokens | All strategies |
| **Temperature** | Control randomness | Scales all logits uniformly | All strategies |
| **Top-k** | Limit vocabulary | Masks low-probability tokens | All strategies |
| **Top-p** | Dynamic vocabulary | Masks tokens below cumulative threshold | All strategies |

### Combined Usage

```python
# Common combination: Temperature + Repetition Penalty
logits = model(input_ids)
logits = logits / temperature  # Temperature scaling
logits = repetition_penalty(logits, generated_ids)  # Repetition penalty
probs = torch.softmax(logits, dim=-1)
```

---

## Implementation

### Integration with Sampling Strategies

```python
def generate_with_repetition_penalty(
    model,
    input_ids,
    max_length=100,
    temperature=1.0,
    penalty=1.1
):
    """Generate text with repetition penalty applied."""

    repetition_penalty = RepetitionPenalty(penalty)
    generated_ids = input_ids.clone()

    for _ in range(max_length):
        # Get logits
        logits = model(generated_ids)[:, -1, :]  # [B, V]

        # Apply temperature
        logits = logits / temperature

        # Apply repetition penalty
        logits = repetition_penalty(logits, generated_ids)

        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Check for end token
        if next_token.item() == eos_token_id:
            break

    return generated_ids
```

### Advanced Implementation

```python
class AdaptiveRepetitionPenalty(LogitsProcessor):
    """Repetition penalty that adapts based on frequency."""

    def __init__(self, base_penalty: float = 1.1, frequency_factor: float = 0.1):
        self.base_penalty = base_penalty
        self.frequency_factor = frequency_factor

    def __call__(self, logits: Tensor, generated_ids: Tensor) -> Tensor:
        # Count frequency of each token
        for token_id in torch.unique(generated_ids):
            count = (generated_ids == token_id).sum()
            penalty = self.base_penalty + (count - 1) * self.frequency_factor
            logits[:, token_id] /= penalty
        return logits
```

---

## Tips and Best Practices

### Parameter Tuning

| Use Case | Recommended Penalty | Notes |
|----------|-------------------|-------|
| **Conservative** | 1.05 - 1.1 | Minimal effect, safe for most cases |
| **Balanced** | 1.1 - 1.2 | Good default for most applications |
| **Aggressive** | 1.2 - 1.5 | Strong effect, use for creative tasks |
| **Very Strong** | 1.5 - 2.0 | May hurt coherence, use carefully |

### Best Practices

1. **Start conservative**: Begin with penalty values around 1.1
2. **Test thoroughly**: Evaluate on your specific use case
3. **Combine wisely**: Works well with temperature and top-p sampling
4. **Monitor quality**: Ensure penalty doesn't hurt overall text quality
5. **Consider context**: Adjust based on generation length and task

### Common Pitfalls

- **Too high penalty**: Can make generation too random and incoherent
- **Too low penalty**: May not effectively prevent repetition
- **Inconsistent application**: Apply penalty at every generation step
- **Ignoring other factors**: Consider temperature and other sampling parameters

### Performance Considerations

- **Efficient implementation**: Use vectorized operations when possible
- **Memory usage**: Penalty application is memory-efficient
- **Computation cost**: Minimal overhead compared to model inference
- **Batch processing**: Works efficiently with batched generation

### Time and Space Complexity

#### Time Complexity

**Per Generation Step:**

- **Token extraction**: $O(S)$ where $S$ is sequence length
- **Unique token identification**: $O(S \log S)$ for sorting unique tokens
- **Penalty application**: $O(U \cdot B)$ where $U$ is number of unique tokens, $B$ is batch size
- **Overall per step**: $O(S \log S + U \cdot B)$

**Full Generation:**

- **Total complexity**: $O(L \cdot (S \log S + U \cdot B))$ where $L$ is generation length
- **Worst case**: $O(L \cdot S \log S)$ when most tokens are unique
- **Best case**: $O(L \cdot B)$ when few unique tokens

#### Space Complexity

**Memory Usage:**

- **Unique token storage**: $O(U)$ for storing unique token IDs
- **Logits modification**: $O(1)$ additional space (in-place modification)
- **Total additional space**: $O(U)$ per generation step

**Comparison with Model Inference:**

- **Model forward pass**: $O(S \cdot H^2)$ time, $O(S \cdot H)$ space
- **Repetition penalty**: $O(S \log S)$ time, $O(U)$ space
- **Penalty overhead**: Negligible compared to model computation

#### Optimization Strategies

**Efficient Implementation:**

```python
def optimized_repetition_penalty(logits: Tensor, generated_ids: Tensor, penalty: float) -> Tensor:
    """Optimized repetition penalty with better complexity."""
    # Use set for O(1) lookup instead of list
    unique_tokens = set(generated_ids.flatten().tolist())

    # Vectorized penalty application
    for token_id in unique_tokens:
        logits[:, token_id] /= penalty

    return logits
```

**Memory-Efficient Version:**

```python
def memory_efficient_penalty(logits: Tensor, generated_ids: Tensor, penalty: float) -> Tensor:
    """Memory-efficient repetition penalty."""
    # Process in chunks to reduce memory usage
    chunk_size = 1000
    unique_tokens = torch.unique(generated_ids)

    for i in range(0, len(unique_tokens), chunk_size):
        chunk = unique_tokens[i:i+chunk_size]
        logits[:, chunk] /= penalty

    return logits
```
