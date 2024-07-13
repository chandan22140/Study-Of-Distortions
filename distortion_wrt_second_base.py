import numpy as np
import matplotlib.pyplot as plt

def generate_points_in_sphere(n, num_points, radius=10):
    """Generate num_points uniformly distributed within a n-dimensional sphere."""
    points = np.random.randn(num_points, n)
    points /= np.linalg.norm(points, axis=1).reshape(-1, 1)
    r = np.random.uniform(0, radius**n, num_points) ** (1/n)
    points *= r.reshape(-1, 1)
    return points

def compute_distortion_ratios(A, B, W):
    """Compute distortion ratios for points transformed by matrix W."""
    original_distances = np.linalg.norm(B - A, axis=1)
    A_prime = A @ W
    B_prime = B @ W
    transformed_distances = np.linalg.norm(B_prime - A_prime, axis=1)
    distortion_ratios = (transformed_distances / original_distances) - 1
    return distortion_ratios

def rotate_vector_around_axis(v, axis, theta):
    """Rotate vector v around given axis by theta degrees."""
    theta = np.radians(theta)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cross_product = np.cross(axis, v)
    dot_product = np.dot(axis, v)
    rotated_v = v * cos_theta + cross_product * sin_theta + axis * dot_product * (1 - cos_theta)
    return rotated_v

def compute_alpha(points, dominant_basis):
    """Compute angles (alpha) between points and the dominant basis."""
    cos_alpha = (points @ dominant_basis) / (np.linalg.norm(points, axis=1) * np.linalg.norm(dominant_basis))
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    return np.degrees(alpha)

# Parameters
n = 3
m = 2
num_points = 10000
num_angles = 36

points = generate_points_in_sphere(n, num_points)

# Generate random Gaussian matrix W of size nxm
W = np.random.randn(n, m)

# Define a fixed point A
A = np.zeros(n)

# Find the basis vector with the highest magnitude (dominant basis)
basis_magnitudes = np.linalg.norm(W, axis=0)
highest_magnitude_index = np.argmax(basis_magnitudes)
dominant_basis = W[:, highest_magnitude_index]

# Select a random line segment (pair of points) in R^n
line_segment = np.random.randn(2, n)
line_segment /= np.linalg.norm(line_segment, axis=1).reshape(-1, 1)
# ---------------------------------------------------------------------------

# Compute initial alpha for the line segment
alpha = compute_alpha(line_segment, dominant_basis)[0]

# Initialize an array to store distortion ratios for different beta values
distortion_ratios_beta = []

# Compute distortion ratios for different beta values
for beta in np.linspace(0, 180, num_angles):
    # Rotate the second basis vector around the dominant basis to change beta
    second_basis = W[:, (highest_magnitude_index + 1) % m]
    rotated_second_basis = rotate_vector_around_axis(second_basis, dominant_basis, beta)
    
    # Construct new transformation matrix W_beta
    W_beta = np.copy(W)
    W_beta[:, (highest_magnitude_index + 1) % m] = rotated_second_basis
    
    # Compute distortion ratios for the line segment with the new W_beta
    distortion_ratios = compute_distortion_ratios(line_segment[0], line_segment[1].reshape(1, -1), W_beta)
    distortion_ratios_beta.append(distortion_ratios[0])

# Plot the relation between distortion ratio and beta for the selected alpha
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 180, num_angles), distortion_ratios_beta, marker='o', linestyle='-')
plt.xlabel('Angle (beta) in degrees')
plt.ylabel('Distortion Ratio (epsilon)')
plt.title(f'Relation between Distortion Ratio (epsilon) and Angle (beta) for alpha = {alpha:.2f}')
plt.grid(True)
plt.show()
