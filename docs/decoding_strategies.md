# Decoding Strategies in Language Models

## Table of Contents
1. [Top-k Sampling](#top-k-sampling)
2. [Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)
3. [Numeric Examples](#numeric-examples)
4. [Comparison: Top-k vs Top-p](#comparison-top-k-vs-top-p)
5. [Advanced Decoding Strategies Comparison](#advanced-decoding-strategies-comparison)
6. [FAQ](#faq)

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
