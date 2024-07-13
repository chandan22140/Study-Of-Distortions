import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.spatial.distance import pdist, squareform

# Load MNIST data and filter classes '1' and '7'
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data.to_numpy()
labels = mnist.target.to_numpy()

class_1_idx = np.where(labels == '1')[0]
class_7_idx = np.where(labels == '7')[0]

class_1_data = data[class_1_idx]
class_7_data = data[class_7_idx]

# Combine the data and labels
combined_data = np.vstack([class_1_data, class_7_data])
combined_labels = np.hstack([np.ones(len(class_1_data)), np.full(len(class_7_data), 7)])

print("combined_data", combined_data.shape)
print("combined_labels", combined_labels.shape)

# Compute all pairwise distances
distances = squareform(pdist(combined_data))

# Ensure distances greater than 0 for percentile calculation
non_zero_distances = distances[distances > 0]

# Determine the threshold for the smallest 7% of distances
threshold_percentile = 7
threshold_distance = np.percentile(non_zero_distances, threshold_percentile)

# Find point pairs within the threshold distance
close_pairs = np.argwhere((distances < threshold_distance) & (distances > 0))

# Compute weighted average of the close point pairs
weights = 1 / distances[close_pairs[:, 0], close_pairs[:, 1]]

# Initialize the weighted average vector
weighted_avg_vector = np.zeros(combined_data.shape[1])

# Process data in smaller chunks to avoid memory issues
chunk_size = 10000  # Adjust this value based on available memory
num_chunks = len(close_pairs) // chunk_size + 1

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(close_pairs))
    chunk_pairs = close_pairs[start_idx:end_idx]
    chunk_weights = weights[start_idx:end_idx]
    chunk_diff = combined_data[chunk_pairs[:, 1]] - combined_data[chunk_pairs[:, 0]]
    weighted_avg_vector += np.sum(chunk_weights[:, np.newaxis] * chunk_diff, axis=0)

# Normalize the weighted average vector
if np.linalg.norm(weighted_avg_vector) == 0:
    weighted_avg_vector = np.random.randn(combined_data.shape[1])
    print("problemmmmmm....")
else:
    weighted_avg_vector /= np.linalg.norm(weighted_avg_vector)

# Construct the transformation matrix W
n = combined_data.shape[1]
k = 10  # Desired lower dimensionality

W = np.zeros((n, k))
W[:, 0] = weighted_avg_vector * 2  # Make the dominant column twice as large in magnitude

# Construct other columns and ensure their mean is 0
for i in range(1, k):
    random_vector = np.random.randn(n)
    random_vector -= random_vector.mean()
    random_vector /= random_vector.std()
    W[:, i] = random_vector

# Subtract the mean from the rest k-1 columns
W[:, 1:] -= W[:, 1:].mean(axis=0)

# Normalize W to have a standard deviation of 1 across all entries
W /= W.std()

# Project the data points using W
projected_data = combined_data @ W

# Check for NaN values in projected_data
if np.any(np.isnan(projected_data)):
    print("NaN values found in projected_data. Exiting...")
    exit()

# Classify using Random Forest
rf = RandomForestClassifier()
rf.fit(projected_data, combined_labels)
predictions = rf.predict(projected_data)

# Print classification report
print(classification_report(combined_labels, predictions))

# Plot relation between average distortion ratio (epsilon) and angle (alpha)
dominant_basis = W[:, 0]
cos_alpha = (combined_data @ dominant_basis) / (np.linalg.norm(combined_data, axis=1) * np.linalg.norm(dominant_basis))
alphas = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
distortion_ratios = (np.linalg.norm(projected_data - projected_data.mean(axis=0), axis=1) / np.linalg.norm(combined_data - combined_data.mean(axis=0), axis=1)) - 1

bins = np.linspace(0, 180, 36)
bin_indices = np.digitize(alphas, bins)
bin_means = [distortion_ratios[bin_indices == i].mean() for i in range(1, len(bins))]

plt.figure(figsize=(10, 6))
plt.plot(bins[:-1] + np.diff(bins)/2, bin_means, marker='o', linestyle='-')
plt.xlabel('Angle (alpha) in degrees')
plt.ylabel('Average Distortion Ratio (epsilon)')
plt.title('Relation between Average Distortion Ratio (epsilon) and Angle (alpha)')
plt.grid(True)
plt.show()
