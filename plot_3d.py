import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def generate_points_in_sphere(n, num_points, radius=10):
    """Generate num_points uniformly distributed within a n-dimensional sphere."""
    points = np.random.randn(num_points, n)
    points /= np.linalg.norm(points, axis=1).reshape(-1, 1)
    r = np.random.uniform(0, radius**n, num_points) ** (1/n)
    points *= r.reshape(-1, 1)
    return points

# Step 1: Generate  points in R^3 with coordinates ranging from -10 to +10
num_points = 10000
n = 3
m = 2
points = generate_points_in_sphere(n, num_points)

# Step 2: Generate random Gaussian matrix G of size 3x2
G = np.random.randn(3, 2)  # Entries drawn from a Gaussian distribution

# Step 3: Define a fixed point A
A = np.array([0, 0, 0])
B = points

# Step 4: Compute original distances |AB|
original_distances = np.linalg.norm(B - A, axis=1)

# Step 5: Transform points A and B using matrix G to get A' and B'
A_prime = A @ G
B_prime = B @ G

# Step 6: Compute transformed distances |A'B'|
transformed_distances = np.linalg.norm(B_prime - A_prime, axis=1)

# Step 7: Calculate distortion ratios
distortion_ratios = (transformed_distances / original_distances) - 1

B_valid = B
distortion_ratios_valid = distortion_ratios

# Directly map the distortion ratios to the colormap
colors = plt.cm.viridis((distortion_ratios_valid - distortion_ratios_valid.min()) / (distortion_ratios_valid.max() - distortion_ratios_valid.min()))

# Step 10: Create a 3D plot for the points B and color them by distortion ratio
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot points with valid distortion ratios
sc = ax.scatter(B_valid[:, 0], B_valid[:, 1], B_valid[:, 2], c=distortion_ratios_valid, cmap='viridis', marker='.')

# Plot point A
ax.scatter(A[0], A[1], A[2], c='red', marker='x', s=100, label='A')

# Calculate magnitudes of the basis vectors
magnitude_basis_1 = np.linalg.norm(G[:, 0])
magnitude_basis_2 = np.linalg.norm(G[:, 1])

# Plot the first two basis vectors of the Gaussian matrix G
ax.quiver(A[0], A[1], A[2], G[0, 0], G[1, 0], G[2, 0], 
          length=20, color='black', arrow_length_ratio=0.1, linewidth=2, 
          label=f'Basis 1 (|G1|={magnitude_basis_1:.2f})')

ax.quiver(A[0], A[1], A[2], G[0, 1], G[1, 1], G[2, 1], 
          length=20, color='black', arrow_length_ratio=0.1, linewidth=2, 
          label=f'Basis 2 (|G2|={magnitude_basis_2:.2f})')

# Construct and plot the third vector orthogonal to both basis vectors
basis_1 = G[:, 0]
basis_2 = G[:, 1]
basis_3 = np.cross(basis_1, basis_2)
ax.quiver(A[0], A[1], A[2], basis_3[0], basis_3[1], basis_3[2], 
          length=20, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Orthogonal Basis')

ax.quiver(A[0], A[1], A[2], -basis_3[0], -basis_3[1], -basis_3[2], 
          length=20, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Orthogonal Basis')

# Plot the resultant vector of the two basis vectors
resultant_vector = basis_1 + basis_2
ax.quiver(A[0], A[1], A[2], resultant_vector[0], resultant_vector[1], resultant_vector[2], 
          length=20, color='pink', arrow_length_ratio=0.1, linewidth=2, label='Resultant Vector')

# Set the axes' limits to center point A and increase the range
range_increase = 10
ax.set_xlim(A[0] - range_increase, A[0] + range_increase)
ax.set_ylim(A[1] - range_increase, A[1] + range_increase)
ax.set_zlim(A[2] - range_increase, A[2] + range_increase)

ax.set_xlabel('X coordinate of B')
ax.set_ylabel('Y coordinate of B')
ax.set_zlabel('Z coordinate of B')
ax.set_title('Distortion Ratios for A with points B in R3')
ax.legend()

# Add color bar with actual distortion ratio values
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Distortion Ratio (|A\'B\'|/|AB| - 1)')
cbar.set_ticks([distortion_ratios_valid.min(), 0, distortion_ratios_valid.max()])
cbar.set_ticklabels([f'{distortion_ratios_valid.min():.3f}', '0', f'{distortion_ratios_valid.max():.3f}'])

plt.show()
