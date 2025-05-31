import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
    pairwise_distances,
)

# ------------------------------
# Load Data
# ------------------------------
assignments_df = pd.read_csv('./fed_cluster_assignments/test_assignments.csv')
y_test = pd.read_csv('./data/uci_har/processed/test/y_test.csv', header=None).squeeze()

# Extract cluster assignments and features
cluster_assignments = assignments_df['cluster_assignment'].values
features = assignments_df.drop(columns=['point_index', 'cluster_assignment']).values

# ------------------------------
# Metric 1: Adjusted Rand Index
# ------------------------------
ari = adjusted_rand_score(y_test, cluster_assignments)

# ------------------------------
# Metric 2: Normalized Mutual Information
# ------------------------------
nmi = normalized_mutual_info_score(y_test, cluster_assignments)

# ------------------------------
# Metric 3: Silhouette Score (unsupervised)
# ------------------------------
sil_score = silhouette_score(features, cluster_assignments)

# ------------------------------
# Print Metrics
# ------------------------------
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")

# ------------------------------
# Cluster Size Analysis
# ------------------------------
unique, counts = np.unique(cluster_assignments, return_counts=True)
print("\nCluster sizes:")
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} points")

# ------------------------------
# Silhouette Scores per Cluster
# ------------------------------
sil_samples = silhouette_samples(features, cluster_assignments)

# Boxplot of silhouette scores per cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x=cluster_assignments, y=sil_samples)
plt.title("Silhouette Score Distribution per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# Inter-Centroid Distances
# ------------------------------
centroids_df = pd.read_csv('./fed_cluster_assignments/final_centroids.csv')
centroids = centroids_df.values
centroid_distances = pairwise_distances(centroids)

print("\nPairwise centroid distances:")
dist_df = pd.DataFrame(
    centroid_distances,
    index=[f"C{i}" for i in range(len(centroids))],
    columns=[f"C{i}" for i in range(len(centroids))]
)
print(dist_df.round(2))
