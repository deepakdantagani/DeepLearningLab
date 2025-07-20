import torch

# Example tensor: each row is a batch, each column is a top-k value
kth_vals = torch.tensor(
    [[5, 4, 3], [8, 7, 6]]  # batch 0: top-3 values  # batch 1: top-3 values
)

print("kth_vals:")
print(kth_vals)
print("Shape:", kth_vals.shape)

# Select the last value in each row
last_vals = kth_vals[..., -1]
print("\nkth_vals[..., -1]:")
print(last_vals)
print("Shape:", last_vals.shape)

# Add a new axis at the end
last_vals_col = kth_vals[..., -1, None]
print("\nkth_vals[..., -1, None]:")
print(last_vals_col)
print("Shape:", last_vals_col.shape)
