import torch

# Basic usage: select from a or b based on condition
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10, 20, 30, 40])
condition = a > 2
result = torch.where(condition, a, b)
print("Basic usage:")
print(result)  # tensor([10, 20, 3, 4])

# Broadcasting example
x = torch.arange(6).reshape(2, 3)
mask = x % 2 == 0
out = torch.where(mask, x, torch.tensor(-1))
print("\nBroadcasting example:")
print(out)
# tensor([[ 0, -1,  2],
#         [-1,  4, -1]])

# Common pattern: set all negatives to zero
z = torch.tensor([-2, -1, 0, 1, 2])
clipped = torch.where(z < 0, torch.tensor(0), z)
print("\nMasking negatives to zero:")
print(clipped)  # tensor([0, 0, 0, 1, 2])

# Top-k filtering example
batch = 10
vocab_size = 5
logits = torch.randn(batch, vocab_size)
print("logits", logits)
print("Logits Shape", logits.shape)
top_k = 2

kth_vals, _ = torch.topk(logits, top_k, dim=-1)
print("kth_vals", kth_vals)
print("kth_val shape", kth_vals.shape)

kth_vals_last = kth_vals[..., -1]
print("kth_vals_last", kth_vals_last)
print("kth_vals_last shape", kth_vals_last.shape)

kth_vals_last_dim = kth_vals_last[..., None]
print("kth_vals_last_dim", kth_vals_last_dim)
print("kth_vals_last_dim shape", kth_vals_last_dim.shape)

mask = logits < kth_vals_last_dim  # [10,5] < [10,1]
result = torch.where(mask, 1, logits)

print("result", result)
print("result shape", result.shape)
