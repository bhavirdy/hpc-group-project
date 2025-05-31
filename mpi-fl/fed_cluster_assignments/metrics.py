import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, accuracy_score
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
# Metric 4: Clustering Accuracy (1-to-1 label mapping)
# ------------------------------

# Ensure labels are from 0 to n-1 for both y_test and cluster_assignments
true_labels = LabelEncoder().fit_transform(y_test)

# Build contingency matrix (true labels vs cluster assignments)
num_classes = len(np.unique(true_labels))
contingency = np.zeros((num_classes, num_classes), dtype=np.int64)
for i in range(len(true_labels)):
    contingency[cluster_assignments[i], true_labels[i]] += 1

# Solve assignment problem to find optimal mapping
row_ind, col_ind = linear_sum_assignment(-contingency)
label_mapping = dict(zip(row_ind, col_ind))

# Map cluster assignments to true labels
mapped_preds = np.array([label_mapping[cluster] for cluster in cluster_assignments])

# Compute accuracy
accuracy = accuracy_score(true_labels, mapped_preds)

# ------------------------------
# Print Metrics
# ------------------------------
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Clustering Accuracy: {accuracy:.4f}")
