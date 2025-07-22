# Decoding Strategies in Language Models

## Table of Contents

1. [Top-k Sampling](#top-k-sampling)
2. [Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)
3. [Numeric Examples](#numeric-examples)
4. [Comparison: Top-k vs Top-p](#comparison-top-k-vs-top-p)
5. [Advanced Decoding Strategies Comparison](#advanced-decoding-strategies-comparison)
6. [Implementation Details](#implementation-details)
7. [FAQ](#faq)

---

## Top-k Sampling

Top-k sampling restricts the token selection to the **top k most likely tokens**, then samples from that set using softmax.

**Algorithm:**

1. Keep top-k tokens only:
   $$
   z_i^{(k)} = \begin{cases}
   z_i & \text{if } i \in \text{TopK}(z, k) \\
   -\infty & \text{otherwise}
   \end{cases}
   $$
2. Apply softmax on the top-k logits:
   $$
   p_i^{(k)} = \frac{\exp(z_i^{(k)})}{\sum_j \exp(z_j^{(k)})}
   $$
3. Sample a token:
   $$
   \text{next\_token} \sim \text{Categorical}(p^{(k)})
   $$

---

## Top-p (Nucleus) Sampling

Top-p (nucleus) sampling chooses the **smallest set of tokens** whose cumulative probability reaches or exceeds $p$, then samples from that set.

**Algorithm:**

1. Softmax the logits:
   $$
   p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
   $$
2. Sort tokens by $p_i$ (descending).
3. Find the smallest prefix such that:
   $$
   \sum_{j=1}^k p_{j} \geq p
   $$
4. Mask all others, renormalize, and sample:
   $$
   \text{next\_token} \sim \text{Categorical}(p^{(p)})
   $$

---

## Numeric Examples

### Top-p Example (p = 0.90)

| Token | $p_i$ | Cumulative |
|-------|-------|------------|
| the   | 0.55  | 0.55       |
| mat   | 0.20  | 0.75       |
| on    | 0.17  | 0.92       |
| cat   | 0.05  | 0.97       |
| sat   | 0.02  | 0.99       |
| .     | 0.01  | 1.00       |

- Smallest set that adds to $\geq 0.90$ is: `the`, `mat`, `on`
- Renormalize:
  $$
  p = \frac{[0.55, 0.20, 0.17]}{0.92} \approx [0.60, 0.22, 0.18]
  $$
- Token is sampled from this set.

### Top-k Example (k = 3)

Assume vocab = 6 tokens, logits:

| Token | Logit $z_i$ |
|-------|-------------|
| the   | 4.2         |
| mat   | 3.1         |
| on    | 2.7         |
| cat   | 1.3         |
| .     | 0.8         |
| sat   | 0.2         |

- Keep: `the`, `mat`, `on`
- Softmax over these:
  $$
  p = \text{softmax}([4.2, 3.1, 2.7]) \approx [0.61, 0.26, 0.13]
  $$
- 61% chance to pick `the`, 26% `mat`, 13% `on`.
- Other tokens (`cat`, `.`, `sat`) get zero probability.

---

## Comparison: Top-k vs Top-p

| Category              | Top-k                        | Top-p (Nucleus)                |
|-----------------------|------------------------------|-------------------------------|
| Selection method      | Top k highest logits         | Smallest set where $\Sigma p \geq p$ |
| Diversity control     | Fixed size, static diversity | Adaptive to shape of distribution |
| Token count per step  | Always k                     | Varies depending on confidence |
| Safe token pruning    | ❌ Can drop strong tokens at rank (k+1) | ✅ Keeps based on actual prob |
| Adaptive?             | ❌ No                        | ✅ Yes                         |
| If distribution sharp | Only 1-2 tokens still selected | 1-2 tokens, so behaves like greedy |
| If distribution flat  | Still 3 tokens              | Can include 5–10 tokens or more |

---

## Advanced Decoding Strategies Comparison

| Category              | Greedy | Temperature | Multinomial Sampling | Top-k | Top-p (Nucleus) | Repetition Penalty |
|-----------------------|--------|-------------|---------------------|-------|-----------------|-------------------|
| Definition            | Picks argmax | Scales logits before softmax | Samples from softmax distribution | Samples from top-k logits only | Samples from top cumulative prob tokens | Penalizes repeated tokens |
| Formula               | $\arg\max(\text{softmax}(z))$ | $\text{softmax}(z/T)$ | $x \sim \text{Categorical}(\text{softmax}(z/T))$ | Remove all but top-k from logits | Remove tokens until $\Sigma p \geq p_{thresh}$ | $z_t = z_t/\text{penalty}$ for repeated $t$ |
| Output Type           | Deterministic | Deterministic (until sampling) | Stochastic | Stochastic | Stochastic | Deterministic or stochastic |
| Adds Diversity?       | ❌ No | ⚠️ Some (with $T>1$) | ✅ High | ✅ High (controlled) | ✅ High (adaptive) | ⚠️ Controls repetition |
| Controls Hallucination? | ✅ Best | ⚠️ Medium | ❌ Prone to hallucination | ✅ Stronger control than pure sampling | ✅ Stronger + context aware | ✅ Helps prevent repeating nonsense |
| Best for              | QA, Code | Chatbots, Summarization | Creative writing | Creative + structured | Conversational AI, dialogue | Chat + story generation + interviews |
| Worst for             | Stories, Open-ended | Exact outputs | QA, factual accuracy | QA with long tail | Structured answers | May under-penalize diverse tokens |
| Requires Softmax?     | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Temperature Used?     | ❌ No | ✅ $T>1$ = more diverse | ✅ Needed | ✅ Often used | ✅ Often used | ✅ Often used with sampling |

---

## Implementation Details

### PyTorch Implementation

The filtering operations can be efficiently implemented using PyTorch's `scatter_` operation:

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    """
    Apply top-k and top-p filtering to logits.
    """
    if top_k > 0:
        # Top-k filtering
        k_values, _ = torch.topk(logits, top_k)
        condition = logits < k_values[..., -1, None]
        logits = torch.where(condition, filter_value, logits)

    if 0.0 < top_p < 1.0:
        # Top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability above threshold
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        sorted_logits[mask] = filter_value

        # Restore original ordering
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits
```

### Key Implementation Points

1. **Efficient Filtering**: Use `torch.topk()` for top-k and `torch.sort()` for top-p
2. **In-place Operations**: `scatter_` modifies tensors in-place for memory efficiency
3. **Batch Processing**: Operations work on batched inputs `[batch_size, vocab_size]`
4. **Filter Value**: Use `-float("inf")` to ensure zero probability after softmax

---

## FAQ

**Q: Is Greedy the same as Top-k?**

- Greedy is Top-k with k=1 — it always picks the max token deterministically.

**Q: Can Top-k ever pick the max token?**

- Yes. The top token is included in the set. It has the highest probability, but not guaranteed to be picked (unless k=1).

**Q: How is Top-p different from Top-k?**

- Top-k selects a fixed number of tokens; Top-p selects a variable number based on cumulative probability. Top-p adapts to how confident the model is.

**Q: Does Top-p sample from full vocab?**

- No — only from the nucleus set (a subset). Multinomial sampling samples from full vocab. Top-p trims the low-probability tail.

**Q: Is Top-p better than Top-k?**

- Often yes — because it adapts. When the model is confident (sharp probs), it keeps few tokens. When uncertain (flat probs), it explores more.

**Q: Can both methods pick wrong tokens?**

- Yes — sampling introduces randomness. Lower probability ≠ zero probability.

---

## Time and Space Complexity

### Time Complexity

**Per Generation Step:**

**Top-k Filtering:**

- **Top-k operation**: $O(V \log k)$ where $V$ is vocabulary size, $k$ is top-k parameter
- **Masking operation**: $O(V)$ for applying filter
- **Overall top-k**: $O(V \log k + V) = O(V \log k)$

**Top-p (Nucleus) Filtering:**

- **Sorting logits**: $O(V \log V)$ for sorting vocabulary logits
- **Softmax computation**: $O(V)$ for probability calculation
- **Cumulative sum**: $O(V)$ for cumulative probability
- **Masking and restoration**: $O(V)$ for applying and restoring order
- **Overall top-p**: $O(V \log V + V + V + V) = O(V \log V)$

**Combined (Top-k + Top-p):**

- **Sequential application**: $O(V \log k + V \log V) = O(V \log V)$ (dominated by sorting)

**Full Generation:**

- **Total complexity**: $O(L \cdot V \log V)$ where $L$ is generation length
- **With model inference**: $O(L \cdot (S \cdot H^2 + V \log V))$ where $S$ is sequence length, $H$ is hidden size
- **Worst case**: $O(L \cdot V \log V)$ when filtering dominates
- **Best case**: $O(L \cdot S \cdot H^2)$ when model inference dominates

### Space Complexity

**Memory Usage:**

- **Logits storage**: $O(V)$ for vocabulary logits
- **Top-k auxiliary**: $O(k)$ for storing top-k values and indices
- **Top-p auxiliary**: $O(V)$ for sorted logits and indices
- **Probability storage**: $O(V)$ for softmax probabilities
- **Cumulative storage**: $O(V)$ for cumulative probabilities
- **Total additional space**: $O(V)$ (dominated by sorting operations)

**Comparison with Other Strategies:**

- **Top-k/Top-p**: $O(V \log V)$ time, $O(V)$ space
- **Greedy**: $O(V)$ time, $O(V)$ space
- **Temperature**: $O(V)$ time, $O(V)$ space
- **Repetition penalty**: $O(S \log S)$ time, $O(U)$ space where $U$ is unique tokens

### Optimization Strategies

**Efficient Top-k Implementation:**

```python
def optimized_top_k(logits, k):
    """Optimized top-k with minimal memory usage."""
    # Use torch.topk for efficient implementation
    values, indices = torch.topk(logits, k, dim=-1)
    # Create mask efficiently
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
    return logits.masked_fill(~mask, -float('inf'))
```

**Memory-Efficient Top-p:**

```python
def memory_efficient_top_p(logits, p):
    """Memory-efficient top-p implementation."""
    # Sort in-place to reduce memory usage
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Find cutoff point
    cutoff = torch.sum(cumulative_probs < p, dim=-1, keepdim=True)
    mask = torch.arange(sorted_logits.size(-1), device=logits.device) < cutoff

    # Apply mask and restore order
    sorted_logits[~mask] = -float('inf')
    logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits
```

### Performance Considerations

**Parameter Impact:**

- **Small k values**: Faster top-k filtering, less diversity
- **Large k values**: Slower filtering, more diversity
- **Low p values**: Faster top-p (fewer tokens to process)
- **High p values**: Slower top-p (more tokens to process)

**Practical Guidelines:**

- **Top-k**: Use when you need predictable token count
- **Top-p**: Use when you need adaptive filtering
- **Combined**: Apply top-k first, then top-p for best performance
- **Memory**: Top-p uses more memory due to sorting operations
