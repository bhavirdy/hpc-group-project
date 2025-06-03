import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Load data
assignments_df = pd.read_csv('./cent_cluster_assignments/test_assignments.csv')
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
# Print Metrics
# ------------------------------
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Silhouette Score: {sil_score:.4f}")