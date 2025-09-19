"""Playground: torch.arange

Run:
    python playground/tensor/05_torch_arange.py
"""

import torch


def basic_examples() -> None:
    print("=== Basic arange examples ===")
    print("arange(5):", torch.arange(5))
    print("arange(2, 6):", torch.arange(2, 6))
    print("arange(0, 10, 3):", torch.arange(0, 10, 3))
    print("arange(0.0, 1.0, 0.2):", torch.arange(0.0, 1.0, 0.2))
    print("arange(5, 0, -1):", torch.arange(5, 0, -1))
    print("arange(0, 4, dtype=int64):", torch.arange(0, 4, dtype=torch.int64))
    print()


def pitfalls_and_alternatives() -> None:
    print("=== Pitfalls and alternatives ===")
    # Floating step may accumulate error
    x = torch.arange(0.0, 1.0, 0.1)
    print("arange float step (0.1):", x)
    # If you need an exact number of points, prefer linspace
    y = torch.linspace(0.0, 1.0, steps=11)
    print("linspace steps=11:", y)
    print()


def shape_and_device() -> None:
    print("=== Shape and device ===")
    a = torch.arange(12)
    print("a shape:", a.shape)
    b = torch.arange(0, 12).reshape(3, 4)
    print("b reshape(3,4):\n", b)
    c = torch.arange(0, 6, device="cpu")
    print("device cpu:", c.device, c)
    print()


if __name__ == "__main__":
    basic_examples()
    pitfalls_and_alternatives()
    shape_and_device()


