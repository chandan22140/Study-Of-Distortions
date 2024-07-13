import numpy as np
import matplotlib.pyplot as plt

def generate_points_in_sphere(n, num_points, radius=10):
    """Generate num_points uniformly distributed within a n-dimensional sphere."""
    points = np.random.randn(num_points, n)
    points /= np.linalg.norm(points, axis=1).reshape(-1, 1)
    r = np.random.uniform(0, radius**n, num_points) ** (1/n)
    points *= r.reshape(-1, 1)
    return points

def compute_distortion_ratios(points, W, A):
    """Compute distortion ratios for points transformed by matrix W."""
    B = points
    original_distances = np.linalg.norm(B - A, axis=1)
    A_prime = A @ W
    B_prime = B @ W
    transformed_distances = np.linalg.norm(B_prime - A_prime, axis=1)
    distortion_ratios = (transformed_distances / original_distances) - 1
    return distortion_ratios

def compute_alpha(points, dominant_basis):
    """Compute angles (alpha) between points and the dominant basis."""
    cos_alpha = (points @ dominant_basis) / (np.linalg.norm(points, axis=1) * np.linalg.norm(dominant_basis))
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    return np.degrees(alpha)

# Parameters
n = 100
m = 5
num_points = 10000

# Generate n points in R^15 within a sphere with radius 10
points = generate_points_in_sphere(n, num_points)

# Generate random Gaussian matrix W of size 15x5
W = np.random.randn(n, m)

# Define a fixed point A
A = np.zeros(n)

# Compute distortion ratios
distortion_ratios = compute_distortion_ratios(points, W, A)

# Find the basis vector with the highest magnitude
basis_magnitudes = np.linalg.norm(W, axis=0)
highest_magnitude_index = np.argmax(basis_magnitudes)
highest_magnitude_basis = W[:, highest_magnitude_index]

# Compute angles (alpha) between points and the dominant basis
alphas = compute_alpha(points, highest_magnitude_basis)

# Calculate average distortion ratio for each angle bin
bins = np.linspace(0, 180, 36)  # Divide 0-180 degrees into 36 bins
bin_indices = np.digitize(alphas, bins)
bin_means = [distortion_ratios[bin_indices == i].mean() for i in range(1, len(bins))]

# Plot the relation between average distortion ratio and angle
plt.figure(figsize=(10, 6))
plt.plot(bins[:-1] + np.diff(bins)/2, bin_means, marker='o', linestyle='-')
plt.xlabel('Angle (alpha) in degrees')
plt.ylabel('Average Distortion Ratio (epsilon)')
plt.title('Relation between Average Distortion Ratio (epsilon) and Angle (alpha)')
plt.grid(True)
plt.show()
