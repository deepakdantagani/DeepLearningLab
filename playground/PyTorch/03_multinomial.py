#!/usr/bin/env python3
"""
PyTorch Multinomial Sampling Playground

This module demonstrates torch.multinomial, a function for sampling from
probability distributions. Commonly used in text generation, reinforcement
learning, and other probabilistic sampling scenarios.

Key Concepts:
- Sampling from probability distributions
- Temperature scaling effects
- Batch processing with multinomial
- Replacement vs non-replacement sampling
"""

import torch


def basic_multinomial_example() -> None:
    """Basic example of torch.multinomial with a simple probability distribution."""
    print("=== Basic Multinomial Example ===")

    # Simple probability distribution for 5 tokens
    probs = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])
    print(f"Probability distribution: {probs}")
    print(f"Sum of probabilities: {probs.sum():.3f}")

    # Sample 1 token
    sample = torch.multinomial(probs, num_samples=1)
    print(f"Sampled index: {sample}")
    print(f"Selected token probability: {probs[sample]:.3f}")

    # Sample multiple tokens with replacement
    samples = torch.multinomial(probs, num_samples=5, replacement=True)
    print(f"5 samples with replacement: {samples}")

    # Sample multiple tokens without replacement
    samples_no_replace = torch.multinomial(probs, num_samples=3, replacement=False)
    print(f"3 samples without replacement: {samples_no_replace}")
    print()


def temperature_sampling_example() -> None:
    """Demonstrate how temperature affects sampling behavior."""
    print("=== Temperature Sampling Example ===")

    # Original logits (before softmax)
    logits = torch.tensor([2.0, 1.0, 3.0, 0.5, 1.5])
    print(f"Original logits: {logits}")

    # Different temperature values
    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        # Apply temperature scaling
        scaled_logits = logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)

        print(f"\nTemperature: {temp}")
        print(f"Scaled logits: {scaled_logits}")
        print(f"Probabilities: {probs}")

        # Sample multiple times to show distribution
        samples = torch.multinomial(probs, num_samples=10, replacement=True)
        print(f"10 samples: {samples}")

        # Count occurrences
        unique, counts = torch.unique(samples, return_counts=True)
        print(f"Sample distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    print()


def batch_multinomial_example() -> None:
    """Demonstrate multinomial sampling with batch processing."""
    print("=== Batch Multinomial Example ===")

    # Batch of 3 sequences, each with 4 tokens
    batch_probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],  # Sequence 1
            [0.4, 0.3, 0.2, 0.1],  # Sequence 2
            [0.25, 0.25, 0.25, 0.25],  # Sequence 3 (uniform)
        ]
    )

    print(f"Batch probabilities shape: {batch_probs.shape}")
    print(f"Batch probabilities:\n{batch_probs}")

    # Sample 1 token per sequence
    batch_samples = torch.multinomial(batch_probs, num_samples=1)
    print(f"Batch samples: {batch_samples}")
    print(f"Batch samples shape: {batch_samples.shape}")

    # Sample 2 tokens per sequence
    batch_samples_2 = torch.multinomial(batch_probs, num_samples=2, replacement=True)
    print(f"Batch samples (2 per sequence): {batch_samples_2}")
    print(f"Batch samples shape: {batch_samples_2.shape}")
    print()


def text_generation_simulation() -> None:
    """Simulate text generation using multinomial sampling."""
    print("=== Text Generation Simulation ===")

    # Simulate vocabulary of 10 tokens
    vocab_size = 10

    # Simulate logits from a language model
    torch.manual_seed(42)  # For reproducible results
    logits = torch.randn(vocab_size)

    print(f"Model logits: {logits}")

    # Apply temperature
    temperature = 1.0
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    print(f"Temperature: {temperature}")
    print(f"Token probabilities: {probs}")

    # Generate sequence of 5 tokens
    sequence = []
    for i in range(5):
        sample = torch.multinomial(probs, num_samples=1)
        token_id = sample.item()
        sequence.append(token_id)
        print(f"Step {i+1}: Selected token {token_id} (prob: {probs[sample]:.3f})")

    print(f"Generated sequence: {sequence}")
    print()


def multinomial_vs_argmax_comparison() -> None:
    """Compare multinomial sampling vs argmax (greedy) selection."""
    print("=== Multinomial vs Argmax Comparison ===")

    # Create a probability distribution with clear winner
    probs = torch.tensor([0.05, 0.1, 0.6, 0.15, 0.1])
    print(f"Probability distribution: {probs}")

    # Argmax (greedy) - always selects the same token
    greedy_choice = torch.argmax(probs)
    print(f"Argmax (greedy) choice: {greedy_choice}")

    # Multinomial sampling - can select different tokens
    print("Multinomial samples (10 trials):")
    for i in range(10):
        sample = torch.multinomial(probs, num_samples=1)
        print(f"  Trial {i+1}: {sample.item()} (prob: {probs[sample]:.3f})")

    # Show distribution over many samples
    many_samples = torch.multinomial(probs, num_samples=1000, replacement=True)
    unique, counts = torch.unique(many_samples, return_counts=True)
    empirical_probs = counts.float() / 1000

    print("\nEmpirical probabilities (1000 samples):")
    for i, (token, count, emp_prob) in enumerate(zip(unique, counts, empirical_probs)):
        print(
            f"  Token {token}: {count} times ({emp_prob:.3f}) vs true prob {probs[i]:.3f}"
        )
    print()


def main() -> None:
    """Run all multinomial examples."""
    print("PyTorch Multinomial Sampling Playground")
    print("=" * 50)

    basic_multinomial_example()
    temperature_sampling_example()
    batch_multinomial_example()
    text_generation_simulation()
    multinomial_vs_argmax_comparison()

    print("Playground completed! 🎉")


if __name__ == "__main__":
    main()
