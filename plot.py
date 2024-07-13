import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate n points in R^2 with coordinates ranging from -5 to +5
n = 10000
points = np.random.uniform(-5, 5, (n, 2))

# Step 2: Generate random Gaussian matrix G of size 2x1
G = np.random.randn(2, 1)

# Step 3: Define the number of different values of A to test
num_A_points = 1  # Let's say we test with 5 different values of A

# Step 4: Create a figure for the subplots
fig = plt.figure(figsize=(15, 10))

# Iterate over multiple values of A and create separate plots
for i in range(num_A_points):
    ax = fig.add_subplot(2, (num_A_points + 1) // 2, i + 1, projection='3d')
    A = points[i]
    B = np.delete(points, i, axis=0)

    # Compute original distances |AB|
    original_distances = np.linalg.norm(B - A, axis=1)

    # Transform points A and B using matrix G to get A' and B'
    A_prime = A @ G
    B_prime = B @ G

    # Compute transformed distances |A'B'|
    transformed_distances = np.linalg.norm(B_prime - A_prime, axis=1)

    # Calculate distortion ratios
    distortion_ratios = (transformed_distances / original_distances) - 1

    # Normalize distortion ratios for coloring
    norm = plt.Normalize(distortion_ratios.min(), distortion_ratios.max())
    colors = plt.cm.viridis(norm(distortion_ratios))

    # Plot with color variation based on distortion ratios
    sc = ax.scatter(B[:, 0], B[:, 1], distortion_ratios, c=colors, marker='o', label=f'A{i+1}')

    # Plot point A
    ax.scatter(A[0], A[1], 0, c='red', marker='x', s=100, label='A')

    # Plot the basis vectors of the Gaussian matrix G
    origin = np.zeros((1, 2))
    for j in range(G.shape[1]):
        ax.quiver(origin[0, 0], origin[0, 1], 0, G[0, j], G[1, j], 0, length=10, color='black', linewidth=2, arrow_length_ratio=0.1, label=f'Basis {j+1}')

    ax.set_xlabel('X coordinate of B')
    ax.set_ylabel('Y coordinate of B')
    ax.set_zlabel('Distortion Ratio (|A\'B\'|/|AB| - 1)')
    ax.set_title(f'Distortion Ratios for A{i+1}')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Add color bar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='Distortion Ratio (|A\'B\'|/|AB| - 1)')

plt.show()
