"""Playground for understanding torch.argmax operation."""

import torch


def basic_argmax_example():
    """Basic argmax example to understand the operation."""
    print("=== Basic Argmax Example ===")

    # 1D tensor
    x = torch.tensor([1, 5, 3, 9, 2])
    max_idx = torch.argmax(x)
    print(f"1D tensor: {x}")
    print(f"Argmax: {max_idx} (index of {x[max_idx]})")
    print()


def argmax_2d_example():
    """Demonstrate argmax with 2D tensors."""
    print("=== 2D Argmax Example ===")

    # 2D tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"2D tensor:\n{x}")

    # Find max along rows (dim=1)
    row_max_idx = torch.argmax(x, dim=1)
    print(f"Argmax along rows (dim=1): {row_max_idx}")

    # Find max along columns (dim=0)
    col_max_idx = torch.argmax(x, dim=0)
    print(f"Argmax along columns (dim=0): {col_max_idx}")
    print()


def classification_example():
    """Demonstrate argmax in classification context."""
    print("=== Classification Example ===")

    # Simulate logits from a model
    batch_size = 3
    num_classes = 5
    logits = torch.randn(batch_size, num_classes)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits:\n{logits}")

    # Get predicted class indices
    predictions = torch.argmax(logits, dim=1)
    print(f"Predicted class indices: {predictions}")

    # Show the actual maximum values
    max_values = torch.max(logits, dim=1)[0]
    print(f"Maximum values: {max_values}")
    print()


def keepdim_example():
    """Show the effect of keepdim parameter."""
    print("=== Keepdim Example ===")

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Input tensor:\n{x}")

    # Without keepdim
    result1 = torch.argmax(x, dim=1)
    print(f"Argmax without keepdim: {result1} (shape: {result1.shape})")

    # With keepdim
    result2 = torch.argmax(x, dim=1, keepdim=True)
    print(f"Argmax with keepdim: {result2} (shape: {result2.shape})")
    print()


def sequence_example():
    """Demonstrate argmax with sequence data."""
    print("=== Sequence Example ===")

    # Simulate attention weights or sequence scores
    seq_len = 10
    hidden_dim = 4
    sequence = torch.randn(seq_len, hidden_dim)
    print(f"Sequence shape: {sequence.shape}")

    # Find positions with maximum values for each hidden dimension
    max_positions = torch.argmax(sequence, dim=0)
    print(f"Max positions per dimension: {max_positions}")

    # Find the dimension with maximum value for each position
    max_dimensions = torch.argmax(sequence, dim=1)
    print(f"Max dimensions per position: {max_dimensions}")
    print()


def tie_breaking_example():
    """Show how argmax handles ties (multiple maximum values)."""
    print("=== Tie Breaking Example ===")

    # Tensor with multiple maximum values
    x = torch.tensor([1, 5, 5, 3, 5])
    print(f"Tensor with ties: {x}")

    max_idx = torch.argmax(x)
    print(f"Argmax returns first occurrence: {max_idx} (value: {x[max_idx]})")

    # Find all maximum values
    max_val = torch.max(x)
    all_max_indices = (x == max_val).nonzero(as_tuple=True)[0]
    print(f"All maximum indices: {all_max_indices}")
    print()


if __name__ == "__main__":
    basic_argmax_example()
    argmax_2d_example()
    classification_example()
    keepdim_example()
    sequence_example()
    tie_breaking_example()

    print("=== Key Takeaways ===")
    print("1. argmax returns indices of maximum values, not the values themselves")
    print("2. dim parameter specifies which dimension to reduce")
    print("3. keepdim=True preserves the reduced dimension")
    print("4. Returns LongTensor (integer tensor)")
    print("5. In case of ties, returns index of first occurrence")
    print("6. Common in classification and attention mechanisms")
