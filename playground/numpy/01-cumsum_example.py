import numpy as np

# Example array
a = np.array([1, 2, 3, 4])

# Compute the cumulative sum
cumsum_a = np.cumsum(a)

print("Original array:", a)
print("Cumulative sum:", cumsum_a)

# 2D example
b = np.array([[1, 2], [3, 4]])
cumsum_b_axis0 = np.cumsum(b, axis=0)
cumsum_b_axis1 = np.cumsum(b, axis=1)

print("\n2D array:\n", b)
print("Cumulative sum along axis 0:\n", cumsum_b_axis0)
print("Cumulative sum along axis 1:\n", cumsum_b_axis1)

# Explanation as a comment:
# numpy.cumsum returns the cumulative sum of the elements along a given axis.
# For 1D arrays, it returns a running total.
# For 2D arrays, you can specify the axis (0 for columns, 1 for rows).
