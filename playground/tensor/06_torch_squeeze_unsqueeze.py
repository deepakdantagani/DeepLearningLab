import torch


def demo_unsqueeze():
    x = torch.tensor([1, 2, 3])  # [3]
    print("x shape:", x.shape)
    print("unsqueeze(0):", x.unsqueeze(0).shape)  # [1, 3]
    print("unsqueeze(1):", x.unsqueeze(1).shape)  # [3, 1]
    print("unsqueeze(-1):", x.unsqueeze(-1).shape)  # [3, 1]


def demo_squeeze():
    y = torch.randn(1, 3, 1, 5)  # [1, 3, 1, 5]
    print("y shape:", y.shape)
    print("squeeze():", y.squeeze().shape)  # [3, 5]
    print("squeeze(0):", y.squeeze(0).shape)  # [3, 1, 5]
    print("squeeze(2):", y.squeeze(2).shape)  # [1, 3, 5]


def demo_broadcasting():
    batch, seq, heads, d = 2, 4, 3, 6
    x = torch.randn(batch, seq, heads, d)  # [B, S, H, d]
    cos = torch.randn(seq, d // 2)  # [S, d/2]
    sin = torch.randn(seq, d // 2)  # [S, d/2]

    # Match shapes for broadcasting like in RoPE
    cos_b = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, d/2]
    sin_b = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, d/2]

    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b

    print("x_even shape:", x_even.shape)
    print("cos_b shape:", cos_b.shape)
    print("rot_even shape:", rot_even.shape)
    print("rot_odd shape:", rot_odd.shape)


if __name__ == "__main__":
    print("-- unsqueeze --")
    demo_unsqueeze()
    print("\n-- squeeze --")
    demo_squeeze()
    print("\n-- broadcasting --")
    demo_broadcasting()
