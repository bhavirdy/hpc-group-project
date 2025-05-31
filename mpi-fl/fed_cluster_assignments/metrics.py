import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder

# Load data
assignments_df = pd.read_csv('./fed_cluster_assignments/test_assignments.csv')
y_test = pd.read_csv('./data/uci_har/processed/test/y_test.csv', header=None).squeeze()  # true labels

# Extract cluster assignments and features
cluster_assignments = assignments_df['cluster_assignment'].values
features = assignments_df.drop(columns=['point_index', 'cluster_assignment']).values

# ------------------------------
# Map true labels 1-6 to two groups (HAR Dataset)
# ------------------------------
# HAR Dataset activity mapping:
# Group 1 (Movement): [1,2,3] -> WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
# Group 0 (Stationary): [4,5,6] -> SITTING, STANDING, LAYING

mapped_labels = np.where(np.isin(y_test, [1,2,3]), 1, 0)

print(f"Original label distribution: {np.bincount(y_test)}")
print(f"Mapped label distribution: {np.bincount(mapped_labels)}")
print(f"Cluster assignment distribution: {np.bincount(cluster_assignments)}")

# ------------------------------
# Metric 1: Adjusted Rand Index
# ------------------------------
ari = adjusted_rand_score(mapped_labels, cluster_assignments)

# ------------------------------
# Metric 2: Normalized Mutual Information
# ------------------------------
nmi = normalized_mutual_info_score(mapped_labels, cluster_assignments)

# ------------------------------
# Metric 3: Silhouette Score (unsupervised)
# ------------------------------
sil_score = silhouette_score(features, cluster_assignments)

# ------------------------------
# Metric 4: Clustering Accuracy
# ------------------------------
# Create a contingency matrix
contingency = np.zeros((2, 2), dtype=np.int64)
for i in range(len(mapped_labels)):
    contingency[cluster_assignments[i], mapped_labels[i]] += 1

print(f"\nContingency Matrix:")
print(f"Cluster\\True  0    1")
print(f"0           {contingency[0,0]:4d} {contingency[0,1]:4d}")
print(f"1           {contingency[1,0]:4d} {contingency[1,1]:4d}")

# Solve the assignment problem (Hungarian algorithm)
row_ind, col_ind = linear_sum_assignment(-contingency)
optimal_match = contingency[row_ind, col_ind].sum()
accuracy = optimal_match / len(mapped_labels)

# ------------------------------
# Print Metrics
# ------------------------------
print(f"\n=== Clustering Evaluation Results ===")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Clustering Accuracy: {accuracy:.4f}")