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
# Metric 4: Clustering Accuracy
# ------------------------------

# Map true labels 1-6 to two groups
# Group 1: [1,2,3] -> label 0
# Group 2: [4,5,6] -> label 1
mapped_labels = y_test.map(lambda x: 0 if x in [1, 2, 3] else 1).values

# Create a contingency matrix
contingency = np.zeros((2, 2), dtype=np.int64)
for i in range(len(mapped_labels)):
    contingency[cluster_assignments[i], mapped_labels[i]] += 1

# Solve the assignment problem (Hungarian algorithm)
row_ind, col_ind = linear_sum_assignment(-contingency)
optimal_match = contingency[row_ind, col_ind].sum()
accuracy = optimal_match / len(mapped_labels)

# ------------------------------
# Print Metrics
# ------------------------------
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Clustering Accuracy: {accuracy:.4f}")
