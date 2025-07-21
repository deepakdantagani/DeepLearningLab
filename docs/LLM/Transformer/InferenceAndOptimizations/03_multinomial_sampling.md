# Multinomial Sampling (Non-Greedy Decoding)

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Temperature Scaling](#temperature-scaling)
4. [Multinomial Sampling Process](#multinomial-sampling-process)
5. [Comparison with Other Strategies](#comparison-with-other-strategies)
6. [Benefits and Limitations](#benefits-and-limitations)
7. [When to Use](#when-to-use)
8. [Implementation](#implementation)
9. [Best Practices](#best-practices)

---

## Overview

**Non-greedy decoding** allows language models to generate more **diverse** and **creative** outputs by sampling from a distribution, rather than always choosing the most probable token (as greedy decoding does).

**Key Characteristics:**

- **Stochastic**: Produces different outputs for the same input
- **Diverse**: Explores multiple possible token sequences
- **Creative**: Suitable for open-ended and creative tasks
- **Human-like**: Introduces natural variation in responses

---

## Mathematical Foundation

### Temperature-Scaled Softmax

The core mathematical operation is temperature scaling applied to the softmax function:

$$
P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Where:

- $P(i)$: Probability of selecting token $i$
- $z_i$: Logit for token $i$
- $T$: Temperature parameter
- $\sum_j$: Summation over all vocabulary tokens

### Temperature Effects

| Temperature $T$ | Behavior | Distribution Shape |
|----------------|----------|-------------------|
| $T = 1$ | Normal softmax | Original distribution |
| $T < 1$ | Sharper distribution | More confident, deterministic |
| $T > 1$ | Flatter distribution | More random, diverse |

---

## Temperature Scaling

### Purpose

Control how "confident" the softmax output is by scaling the logits before applying softmax.

### Formula

$$
\text{scaled\_logits} = \frac{z}{T}
$$

### Visual Effect

**Low Temperature (T < 1):**

- Makes high-probability tokens even more likely
- Reduces diversity, increases determinism
- Good for factual tasks

**High Temperature (T > 1):**

- Flattens the probability distribution
- Increases diversity, reduces determinism
- Good for creative tasks

### Example

```python
# Original logits: [2.0, 1.0, 0.5]
# T = 0.5 (sharp): [0.88, 0.12, 0.00]
# T = 1.0 (normal): [0.66, 0.24, 0.10]
# T = 2.0 (flat): [0.47, 0.32, 0.21]
```

---

## Multinomial Sampling Process

### Algorithm

1. **Get logits** from the model: $z \in \mathbb{R}^{|V|}$
2. **Apply temperature scaling**: $z' = z / T$
3. **Apply softmax**: $P(i) = \text{softmax}(z')$
4. **Sample from distribution**: $\text{next\_token} \sim \text{Categorical}(P)$

### Key Difference from Greedy

| Method | Selection | Formula |
|--------|-----------|---------|
| **Greedy** | Deterministic | $\text{next\_token} = \arg\max_i P(i)$ |
| **Multinomial** | Stochastic | $\text{next\_token} \sim \text{Categorical}(P)$ |

### Characteristics

- **Token is sampled based on its probability**
- **Higher probability → more likely, but not guaranteed**
- **Lower probability tokens can still be selected**
- **Introduces randomness and diversity**

---

## Comparison with Other Strategies

| Category | Greedy Decoding | Temperature Scaling | Multinomial Sampling |
|----------|----------------|-------------------|---------------------|
| **Definition** | Always picks the token with the highest probability | Scales logits before softmax to control sharpness | Samples from a scaled softmax distribution |
| **Formula** | $\arg\max(\text{softmax}(z))$ | $\text{softmax}(z/T)$ | $x \sim \text{Categorical}(\text{softmax}(z/T))$ |
| **Determinism** | ✅ Yes | ✅ Yes (until sampling added) | ❌ No - random sampling |
| **Diversity** | ❌ None | ⚠️ Medium (if T > 1) | ✅ High |
| **Accuracy** | ✅ High (local) | ⚠️ Balanced | ❌ Low (can hallucinate) |
| **Creativity** | ❌ None | ⚠️ Some | ✅ Great for creative tasks |
| **Speed** | ✅ Fastest | ✅ Fast | ✅ Fast |
| **Use Case Fit** | Code, QA, factual | Chatbots, summarization | Brainstorming, writing, creative assistants |
| **Risk of Hallucination** | ❌ Very Low | ⚠️ Moderate | ✅ High |
| **Example Output** | "time → there → was → a → man" | "time → a → little → girl → named" (varies by T) | "dragon → slept → under → the → castle" (random) |

---

## Benefits and Limitations

### ✅ **Benefits of Non-Greedy Decoding**

| Benefit | Explanation |
|---------|-------------|
| **Diversity** | Each generation can be different — good for creative tasks |
| **Exploration** | Can escape local optima → better long-term sequences |
| **Human-like responses** | Introduces natural variation in dialogue and text |
| **Better for open-ended tasks** | Poems, stories, chatbots, analogies |

### ❌ **Limitations**

| Limitation | Explanation |
|------------|-------------|
| **Wild answers** | May pick low-probability words that break coherence or truth |
| **Hallucinations** | Can generate factually incorrect statements |
| **Non-reproducibility** | Same input may produce different output each time |
| **Risk of repetition** | Without repetition penalties, may loop or repeat phrases |

---

## When to Use

| Use Case | Recommended? | Why |
|----------|-------------|-----|
| **Code generation** | ❌ No | Needs accuracy and determinism |
| **Factual QA** | ❌ No | Should produce correct, known answer |
| **Chatbots** | ✅ Yes | Needs creativity and variation |
| **Storytelling** | ✅ Yes | Requires open-ended, engaging language |
| **Brainstorming** | ✅ Yes | Explore diverse ideas |
| **Creative writing** | ✅ Yes | Generate unique and varied content |
| **Dialogue systems** | ✅ Yes | Natural conversation variation |

---

## Implementation

### Basic Multinomial Sampling

```python
def multinomial_sample(logits, temperature=1.0):
    """
    Basic multinomial sampling with temperature scaling.

    Args:
        logits: Model output logits [batch_size, vocab_size]
        temperature: Temperature parameter (T > 1 for diversity)

    Returns:
        Sampled token indices [batch_size, 1]
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Apply softmax to get probabilities
    probs = torch.softmax(scaled_logits, dim=-1)

    # Sample from categorical distribution
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### Advanced Implementation with Safety Checks

```python
def safe_multinomial_sample(logits, temperature=1.0, min_prob=1e-6):
    """
    Multinomial sampling with safety checks and minimum probability.

    Args:
        logits: Model output logits [batch_size, vocab_size]
        temperature: Temperature parameter
        min_prob: Minimum probability threshold

    Returns:
        Sampled token indices [batch_size, 1]
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Apply softmax
    probs = torch.softmax(scaled_logits, dim=-1)

    # Apply minimum probability threshold
    probs = torch.clamp(probs, min=min_prob)

    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### Complete Decoding Loop

```python
def multinomial_decode(model, input_ids, max_length=100, temperature=1.2):
    """
    Complete multinomial sampling decoding loop.

    Args:
        model: Language model
        input_ids: Input token IDs [batch_size, seq_len]
        max_length: Maximum generation length
        temperature: Temperature for sampling

    Returns:
        Generated token IDs
    """
    for _ in range(max_length):
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

        # Multinomial sampling
        next_token = multinomial_sample(logits, temperature)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check for end of sequence
        if (next_token == eos_token_id).any():
            break

    return input_ids
```

### Integration with Repetition Penalty

```python
def multinomial_with_penalty(logits, input_ids, temperature=1.0, penalty=1.1):
    """
    Multinomial sampling with repetition penalty.

    Args:
        logits: Model logits [batch_size, vocab_size]
        input_ids: Previous tokens [batch_size, seq_len]
        temperature: Temperature parameter
        penalty: Repetition penalty factor

    Returns:
        Sampled token indices
    """
    # Apply repetition penalty
    for token_id in set(input_ids[0].tolist()):
        logits[0, token_id] /= penalty

    # Apply temperature and sample
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

---

## Best Practices

### 🧠 **Mastering Multinomial Sampling**

1. **Temperature Tuning**: Start with T=1.2-1.5 for creative tasks
2. **Combine with Top-k/p**: Use filtering to avoid low-probability tokens
3. **Repetition Penalty**: Implement penalties to avoid loops
4. **Safety Checks**: Add minimum probability thresholds
5. **Multiple Samples**: Generate multiple candidates and select best

### **Temperature Guidelines**

| Task Type | Recommended Temperature | Reasoning |
|-----------|------------------------|-----------|
| **Factual QA** | T = 0.7-0.9 | Reduce randomness, increase accuracy |
| **Creative Writing** | T = 1.2-1.5 | Balance creativity and coherence |
| **Brainstorming** | T = 1.5-2.0 | Maximize diversity and exploration |
| **Chatbots** | T = 1.0-1.3 | Natural variation without incoherence |

### **Common Pitfalls**

1. **Too High Temperature**: Can produce incoherent or nonsensical output
2. **No Filtering**: May select very low-probability tokens
3. **Repetition Loops**: Without penalties, can get stuck repeating
4. **Inconsistent Temperature**: Varying T during generation can cause issues

### **Debugging Tips**

```python
def debug_multinomial_step(logits, temperature, tokenizer):
    """
    Debug multinomial sampling step.
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    # Show top probabilities
    top_k_probs, top_k_indices = torch.topk(probs, k=10, dim=-1)

    print(f"Temperature: {temperature}")
    print("Top 10 tokens and probabilities:")
    for i, (prob, idx) in enumerate(zip(top_k_probs[0], top_k_indices[0])):
        token = tokenizer.decode([idx])
        print(f"{i+1}. {token}: {prob:.4f}")

    # Show sampled token
    sampled_idx = torch.multinomial(probs, num_samples=1)
    sampled_token = tokenizer.decode([sampled_idx])
    sampled_prob = probs[0, sampled_idx]
    print(f"Sampled: {sampled_token} (prob: {sampled_prob:.4f})")
```

### **Integration with Other Strategies**

```python
def hybrid_sampling(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    Combine multinomial sampling with top-k and top-p filtering.
    """
    # Apply top-k filtering
    if top_k > 0:
        top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
        logits = torch.where(logits < top_k_logits[..., -1, None],
                           -float('inf'), logits)

    # Apply top-p filtering
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        sorted_logits[mask] = -float('inf')

        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    # Apply temperature and sample
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

---

## Summary

Multinomial sampling is a **powerful technique** for generating diverse and creative text, making it ideal for:

- **Creative writing** and storytelling
- **Chatbots** and dialogue systems
- **Brainstorming** and idea generation
- **Open-ended tasks** requiring variety

However, its **stochastic nature** requires careful tuning and safety measures:

- **Temperature control** to balance creativity and coherence
- **Filtering strategies** (top-k, top-p) to avoid low-probability tokens
- **Repetition penalties** to prevent loops
- **Safety checks** to maintain output quality

When properly implemented, multinomial sampling can produce human-like, engaging, and diverse text that goes beyond the limitations of deterministic greedy decoding.
