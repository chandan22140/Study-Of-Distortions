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


n = 3
m = 2
num_points = 10000
num_angles = 36

# Generate n points in R^15 within a sphere with radius 10
points = generate_points_in_sphere(n, num_points)

# Generate random Gaussian matrix W of size 15x5
W = np.random.randn(n, m)

# Define a fixed point A
A = np.zeros(n)

# Find the basis vector with the highest magnitude (dominant basis)
basis_magnitudes = np.linalg.norm(W, axis=0)
highest_magnitude_index = np.argmax(basis_magnitudes)
dominant_basis = W[:, highest_magnitude_index]

# Select a random line segment (pair of points) in R^15
line_segment = np.random.randn(2, n)
line_segment /= np.linalg.norm(line_segment, axis=1).reshape(-1, 1)
# ---------------------------------------------------------------------------
# Compute initial alpha for the line segment
alpha_values = list(range(0, 181, 5))
alpha_values.append(0)
alpha_values.append(180)

distortion_ratios_alpha_beta = []

# Compute distortion ratios for different alpha and beta values
for alpha in alpha_values:
    # Rotate the line segment to achieve the desired alpha
    angle_to_rotate = alpha - compute_alpha(line_segment, dominant_basis)[0]
    rotated_line_segment = rotate_vector_around_axis(line_segment[1], dominant_basis, angle_to_rotate)
    rotated_line_segment = np.array([line_segment[0], rotated_line_segment])

    distortion_ratios_beta = []
    for beta in np.linspace(0, 180, num_angles):
        # Rotate the second basis vector around the dominant basis to change beta
        second_basis = W[:, (highest_magnitude_index + 1) % m]
        rotated_second_basis = rotate_vector_around_axis(second_basis, dominant_basis, beta)
        
        # Construct new transformation matrix W_beta
        W_beta = np.copy(W)
        W_beta[:, (highest_magnitude_index + 1) % m] = rotated_second_basis
        
        # Compute distortion ratios for the line segment with the new W_beta
        distortion_ratios = compute_distortion_ratios(rotated_line_segment[0], rotated_line_segment[1].reshape(1, -1), W_beta)
        distortion_ratios_beta.append(distortion_ratios[0])
    
    distortion_ratios_alpha_beta.append(distortion_ratios_beta)

# Plot the relation between distortion ratio and beta for different alpha values
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alpha_values):
    plt.plot(np.linspace(0, 180, num_angles), distortion_ratios_alpha_beta[i], marker='.', linestyle='-', label=f'alpha = {alpha}')
plt.xlabel('Angle (beta) in degrees')
plt.ylabel('Distortion Ratio (epsilon)')
plt.title('Relation between Distortion Ratio (epsilon) and Angle (beta) for different alpha values')
plt.legend()
plt.grid(True)
plt.show()



from mpl_toolkits.mplot3d import Axes3D

# Prepare data for 3D plot
alphas = np.repeat(alpha_values, num_angles)
betas = np.tile(np.linspace(0, 180, num_angles), len(alpha_values))
epsilons = np.array(distortion_ratios_alpha_beta).flatten()

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(alphas, betas, epsilons, c=epsilons, cmap='viridis')
ax.set_xlabel('Alpha (degrees)')
ax.set_ylabel('Beta (degrees)')
ax.set_zlabel('Distortion Ratio (epsilon)')
ax.set_title('3D Plot of Distortion Ratio (epsilon) with Alpha and Beta')
plt.colorbar(sc, label='Distortion Ratio (epsilon)')
plt.show()



# # Select specific alpha and beta for observation
# alpha = 45
# beta = 45

# # Rotate the line segment to achieve the desired alpha
# angle_to_rotate = alpha - compute_alpha(line_segment, dominant_basis)[0]
# rotated_line_segment = rotate_vector_around_axis(line_segment[1], dominant_basis, angle_to_rotate)
# rotated_line_segment = np.array([line_segment[0], rotated_line_segment])

# # Rotate the second basis vector around the dominant basis to achieve the desired beta
# second_basis = W[:, (highest_magnitude_index + 1) % m]
# rotated_second_basis = rotate_vector_around_axis(second_basis, dominant_basis, beta)

# # Observe changes in epsilon with respect to magnitude of second basis
# magnitudes = np.linspace(0, np.linalg.norm(dominant_basis), 50)
# distortion_ratios_magnitude = []

# for mag in magnitudes:
#     scaled_second_basis = rotated_second_basis / np.linalg.norm(rotated_second_basis) * mag
#     W_scaled = np.copy(W)
#     W_scaled[:, (highest_magnitude_index + 1) % m] = scaled_second_basis
#     distortion_ratios = compute_distortion_ratios(rotated_line_segment[0], rotated_line_segment[1].reshape(1, -1), W_scaled)
#     distortion_ratios_magnitude.append(distortion_ratios[0])

# # Plot the relation between distortion ratio and magnitude of the second basis
# plt.figure(figsize=(10, 6))
# plt.plot(magnitudes, distortion_ratios_magnitude, marker='o', linestyle='-')
# plt.xlabel('Magnitude of Second Basis')
# plt.ylabel('Distortion Ratio (epsilon)')
# plt.title('Relation between Distortion Ratio (epsilon) and Magnitude of Second Basis')
# plt.grid(True)
# plt.show()
