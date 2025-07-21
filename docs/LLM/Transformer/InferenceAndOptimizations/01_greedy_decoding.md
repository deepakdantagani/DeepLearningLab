# Greedy Decoding

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Step-by-Step Process](#step-by-step-process)
4. [When to Use Greedy Decoding](#when-to-use-greedy-decoding)
5. [Limitations](#limitations)
6. [Comparison with Other Strategies](#comparison-with-other-strategies)
7. [Implementation](#implementation)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

Greedy decoding is a **deterministic, token-by-token decoding algorithm** for autoregressive models like GPT. At each step, it selects the token with the highest probability (argmax of softmax over logits).

**Key Characteristics:**

- **Deterministic**: Always produces the same output for the same input
- **Fast**: Single forward pass per token
- **Locally optimal**: Chooses best next token, not best overall sequence
- **Baseline method**: Foundation for more advanced decoding strategies

---

## Mathematical Foundation

### Final Formula

**Let:**

- $h_t$: hidden state at time step $t$
- $W_o \in \mathbb{R}^{D \times |V|}$: vocabulary projection matrix
- $\text{logits}_t = h_t \cdot W_o$

**Then:**
$$
\hat{y}_t = \arg\max_i \text{Softmax}(h_t \cdot W_o)_i
$$

### Formula Breakdown

1. **Hidden State Projection**: $h_t \cdot W_o$ transforms the hidden state to logits
2. **Softmax Application**: $\text{Softmax}(\cdot)$ converts logits to probabilities
3. **Argmax Selection**: $\arg\max_i$ selects the index of the maximum probability
4. **Output**: $\hat{y}_t$ is the predicted token at time step $t$

---

## Step-by-Step Process

### Algorithm Flow

1. **Tokenize** input string

   ```
   "The capital of France" → [tok₁, tok₂, tok₃, tok₄]
   ```

2. **Forward pass** through model

   ```
   Get logits ∈ ℝ^{B × S × |V|}
   ```

3. **Select last token's logits**

   ```
   logits[:, -1, :] ∈ ℝ^{B × |V|}
   ```

4. **Apply softmax** to get probabilities

   ```
   probs = softmax(logits[:, -1, :])
   ```

5. **Select most probable token**

   ```
   next_token = argmax(probs, dim=-1)
   ```

6. **Append to input**

   ```
   input_ids = torch.cat([input_ids, next_token], dim=1)
   ```

7. **Repeat** until stop condition (`<eos>` or max length)

### PyTorch Implementation

```python
def greedy_decode(model, input_ids, max_length=100):
    """
    Greedy decoding implementation.

    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        max_length: Maximum generation length

    Returns:
        Generated token IDs
    """
    for _ in range(max_length):
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Get last token's logits
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Apply softmax and get argmax
        probs = torch.softmax(last_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]

        # Append to input
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check for end of sequence
        if (next_token == eos_token_id).any():
            break

    return input_ids
```

---

## When to Use Greedy Decoding

### ✅ **Optimal Use Cases**

| Task | Why It Works Well |
|------|-------------------|
| **Code Generation** | One correct token often exists (e.g., syntax) |
| **QA (Closed)** | Factual answer is deterministic |
| **Translation** | Literal translation often suffices |
| **Latency-sensitive tasks** | Very fast: 1 path, 1 token at a time |
| **Early prototyping** | Simple to implement and debug |
| **Consistent testing** | Reproducible results |

### **Example Scenarios**

**Code Generation:**

```
Input: "def calculate_area(radius):"
Greedy: "    return math.pi * radius ** 2"
```

**Question Answering:**

```
Input: "What is the capital of France?"
Greedy: "The capital of France is Paris."
```

---

## Limitations

### ❌ **Key Limitations**

1. **Short-Sightedness**: Picks best next token, not best overall sentence
2. **Repetition**: Can get stuck in loops without repetition penalties
3. **Lack of Diversity**: Always produces the same output
4. **Suboptimal Sequences**: May miss globally better solutions

### **Example of Short-Sightedness**

```
Input: "Once upon a"
Greedy → "time → there → was → a → man"
Beam  → "kingdom → where → dragons → lived"
```

- **Greedy**: Locally optimal but globally suboptimal
- **Beam Search**: Considers multiple paths for better overall sequence

---

## Comparison with Other Strategies

| Feature | Greedy Decoding | Beam Search | Top-k / Top-p Sampling |
|---------|----------------|-------------|------------------------|
| **Output Type** | Deterministic | Deterministic | Stochastic |
| **Speed** | ✅ Fastest | ❌ Slower | ✅ Fast |
| **Diversity** | ❌ None | ✅ Higher | ✅ High (if tuned) |
| **Reproducibility** | ✅ Yes | ✅ Yes | ❌ No |
| **Hallucination Risk** | ❌ Very low | ❌ Low | ✅ Higher |
| **Use Case Fit** | Code, QA | Stories, Answers | Chat, Creative Gen |
| **Memory Usage** | ✅ Low | ❌ High | ✅ Low |
| **Implementation Complexity** | ✅ Simple | ❌ Complex | ✅ Medium |

---

## Implementation

### Basic Implementation

```python
class GreedyStrategy(SamplingStrategy):
    """
    Greedy strategy: select the token with the highest probability.
    """
    def sample(self, logits: Tensor) -> Tensor:  # [B, S, V]
        return torch.argmax(logits, dim=-1, keepdim=True)  # [B, S, 1]
```

### Advanced Implementation with Repetition Penalty

```python
def greedy_decode_with_penalty(model, input_ids, repetition_penalty=1.1):
    """
    Greedy decoding with repetition penalty.
    """
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

        # Apply repetition penalty
        for token_id in set(input_ids[0].tolist()):
            logits[0, token_id] /= repetition_penalty

        # Greedy selection
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

### Performance Optimizations

```python
def optimized_greedy_decode(model, input_ids, max_length=100):
    """
    Optimized greedy decoding with caching.
    """
    # Use KV caching for efficiency
    past_key_values = None

    for _ in range(max_length):
        outputs = model(
            input_ids[:, -1:],  # Only pass last token
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        past_key_values = outputs.past_key_values

    return input_ids
```

---

## Tips and Best Practices

### 🧠 **Mastering Greedy Decoding**

1. **Use as Baseline**: Greedy is the foundation; all other strategies build on top
2. **Debug with Logits**: Always inspect `logits[:, -1, :]` to debug generation
3. **Early Prototyping**: Perfect for initial model testing and benchmarking
4. **Consistent Testing**: Use for reproducible evaluation and testing
5. **Monitor Repetition**: Implement repetition penalties for longer sequences

### **Common Pitfalls**

1. **Repetition Loops**: Without penalties, can generate infinite loops
2. **Overconfidence**: May produce confident but incorrect outputs
3. **Lack of Exploration**: Never considers alternative token sequences
4. **Context Blindness**: Doesn't consider future token implications

### **Debugging Tips**

```python
# Debug greedy decoding
def debug_greedy_step(logits, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=5, dim=-1)

    print("Top 5 tokens and probabilities:")
    for i, (prob, idx) in enumerate(zip(top_k_probs[0], top_k_indices[0])):
        token = tokenizer.decode([idx])
        print(f"{i+1}. {token}: {prob:.4f}")

    # Show selected token
    selected_idx = torch.argmax(probs, dim=-1)
    selected_token = tokenizer.decode([selected_idx])
    print(f"Selected: {selected_token}")
```

### **Integration with Other Strategies**

```python
# Combine greedy with temperature
def temperature_greedy(logits, temperature=1.0):
    """
    Apply temperature scaling before greedy selection.
    """
    scaled_logits = logits / temperature
    return torch.argmax(scaled_logits, dim=-1, keepdim=True)
```

---

## Summary

Greedy decoding is the **simplest and fastest** decoding strategy, making it ideal for:

- **Code generation** where syntax is deterministic
- **Question answering** where factual accuracy is paramount
- **Latency-sensitive applications** requiring fast response times
- **Early prototyping** and model evaluation

However, its **locally optimal nature** makes it unsuitable for creative tasks requiring diversity or tasks where global sequence optimization is important. For such cases, consider beam search or sampling-based strategies.
