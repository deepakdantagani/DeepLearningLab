"""Playground for understanding torch.scatter_ operation."""

import torch


def basic_scatter_example():
    """Basic scatter_ example to understand the operation."""
    print("=== Basic Scatter Example ===")

    # Create a target tensor
    target = torch.zeros(2, 4)
    print(f"Target tensor:\n{target}")

    # Create source values and indices
    src = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    index = torch.tensor([[0, 2, 1, 3], [3, 1, 0, 2]])

    print(f"Source values:\n{src}")
    print(f"Indices:\n{index}")

    # Scatter values into target
    target.scatter_(dim=-1, index=index, src=src)
    print(f"After scatter_:\n{target}")
    print()


def top_p_filtering_scatter_example():
    """Demonstrate scatter_ in the context of top-p filtering."""
    print("=== Top-P Filtering Scatter Example ===")

    # Simulate logits from the filters.py example
    logits = torch.tensor([[1.2, 2.1, 0.5, 1.8], [0.8, 2.5, 1.1, 0.3]])
    print(f"Original logits:\n{logits}")

    # Step 1: Sort in descending order
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    print(f"Sorted logits:\n{sorted_logits}")
    print(f"Sorted indices (original positions):\n{sorted_idx}")

    # Step 2: Simulate top-p filtering (top_p = 0.8)
    # In real implementation, this would be based on cumulative
    # probabilities. Here we'll just filter the last position for
    # demonstration
    filter_value = -float("inf")
    sorted_logits_filtered = sorted_logits.clone()
    sorted_logits_filtered[:, -1] = filter_value  # Filter last position

    print(f"After filtering (last position set to -inf):\n{sorted_logits_filtered}")

    # Step 3: Scatter back to original positions
    logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits_filtered)
    print(f"After scatter_ (restored to original positions):\n{logits}")
    print()


def scatter_with_different_dims():
    """Show scatter_ with different dimensions."""
    print("=== Scatter with Different Dimensions ===")

    # 2D tensor
    target = torch.zeros(3, 4)
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Scatter along dimension 0 (rows)
    index_dim0 = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]])

    print(f"Target (before):\n{target}")
    print(f"Source:\n{src}")
    print(f"Indices (dim=0):\n{index_dim0}")

    target.scatter_(dim=0, index=index_dim0, src=src)
    print(f"After scatter_(dim=0):\n{target}")
    print()


def scatter_vs_scatter():
    """Compare scatter_ (in-place) vs scatter (out-of-place)."""
    print("=== Scatter_ vs Scatter ===")

    original = torch.zeros(2, 3)
    src = torch.tensor([[1, 2, 3], [4, 5, 6]])
    index = torch.tensor([[0, 2, 1], [1, 0, 2]])

    print(f"Original tensor:\n{original}")
    print(f"Source:\n{src}")
    print(f"Indices:\n{index}")

    # In-place scatter_
    tensor_inplace = original.clone()
    tensor_inplace.scatter_(dim=-1, index=index, src=src)
    print(f"After scatter_ (in-place):\n{tensor_inplace}")

    # Out-of-place scatter
    tensor_outplace = torch.scatter(original, dim=-1, index=index, src=src)
    print(f"After scatter (out-of-place):\n{tensor_outplace}")
    print(f"Original unchanged:\n{original}")
    print()


def interactive_scatter_experiment():
    """Interactive experiment to understand scatter_ behavior."""
    print("=== Interactive Scatter Experiment ===")

    # Create a simple example
    target = torch.zeros(2, 4)
    src = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])

    print("Try different index patterns:")
    print("1. Sequential: [[0,1,2,3], [0,1,2,3]]")
    print("2. Reverse: [[3,2,1,0], [3,2,1,0]]")
    print("3. Random: [[1,3,0,2], [2,0,3,1]]")

    # Example with reverse indices
    index = torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]])

    print(f"\nUsing reverse indices:\n{index}")
    target.scatter_(dim=-1, index=index, src=src)
    print(f"Result:\n{target}")
    print()


if __name__ == "__main__":
    basic_scatter_example()
    top_p_filtering_scatter_example()
    scatter_with_different_dims()
    scatter_vs_scatter()
    interactive_scatter_experiment()

    print("=== Key Takeaways ===")
    print("1. scatter_ is an in-place operation (modifies the tensor directly)")
    print("2. index tensor specifies where to place each value from src")
    print("3. dim parameter determines which dimension to scatter along")
    print("4. Useful for restoring original ordering after sorting operations")
    print("5. Common in ML for operations like top-k/top-p filtering")
