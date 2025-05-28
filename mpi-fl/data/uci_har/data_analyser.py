import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load data
DATA_PATH = './data/uci_har/UCI HAR Dataset'
X_train = pd.read_csv(f'{DATA_PATH}/train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv(f'{DATA_PATH}/train/y_train.txt', header=None)
activity_labels = np.loadtxt(f'{DATA_PATH}/activity_labels.txt', dtype=str)
activity_dict = {int(k): v for k, v in activity_labels}

print(f"Data shape: {X_train.shape}")
print(f"Activities: {[str(v) for v in activity_dict.values()]}")

# Standardise and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# PCA Analysis
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
n_90 = np.argmax(cumulative_var >= 0.9) + 1

print(f"Components for 90% variance: {n_90}")

# Find optimal clusters
k_range = range(2, 8)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    silhouette_scores.append(score)

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal clusters: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")

# Perform clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Analyse clusters
print("\nCluster sizes:", {int(k): int(v) for k, v in Counter(clusters).items()})

for cluster_id in range(optimal_k):
    mask = clusters == cluster_id
    activities = y_train.iloc[mask, 0].values
    activity_counts = Counter(activities)
    
    print(f"\nCluster {cluster_id}:")
    for activity_id, count in activity_counts.most_common():
        activity_name = activity_dict[activity_id]
        pct = count / len(activities) * 100
        print(f"  {activity_name}: {count} ({pct:.1f}%)")

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# PCA explained variance
axes[0,0].bar(range(1, 21), explained_var[:20], alpha=0.7)
axes[0,0].set_title('PCA: Top 20 Components')
axes[0,0].set_xlabel('Component')
axes[0,0].set_ylabel('Explained Variance')

# Cumulative variance
axes[0,1].plot(range(1, 101), cumulative_var, 'b-')
axes[0,1].axhline(y=0.9, color='r', linestyle='--', label='90%')
axes[0,1].set_title('Cumulative Explained Variance')
axes[0,1].set_xlabel('Component')
axes[0,1].legend()

# Silhouette scores
axes[1,0].plot(k_range, silhouette_scores, 'go-')
axes[1,0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
axes[1,0].set_title('Silhouette Score vs K')
axes[1,0].set_xlabel('Number of Clusters')
axes[1,0].legend()

# Cluster-activity heatmap
cluster_activity_matrix = np.zeros((len(activity_dict), optimal_k))
for cluster_id in range(optimal_k):
    mask = clusters == cluster_id
    activities = y_train.iloc[mask, 0].values
    for activity_id in activities:
        cluster_activity_matrix[activity_id-1, cluster_id] += 1

im = axes[1,1].imshow(cluster_activity_matrix, cmap='Blues', aspect='auto')
axes[1,1].set_title('Activity-Cluster Heatmap')
axes[1,1].set_xlabel('Cluster')
axes[1,1].set_ylabel('Activity')
axes[1,1].set_xticks(range(optimal_k))
axes[1,1].set_xticklabels(range(optimal_k))
axes[1,1].set_yticks(range(len(activity_dict)))
axes[1,1].set_yticklabels(list(activity_dict.values()), fontsize=8)
plt.colorbar(im, ax=axes[1,1])

plt.tight_layout()
plt.show()

# Summary
active_activities = {'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'}
print(f"\nSummary:")
for cluster_id in range(optimal_k):
    mask = clusters == cluster_id
    activities = [activity_dict[aid] for aid in y_train.iloc[mask, 0].values]
    active_count = sum(1 for a in activities if a in active_activities)
    static_count = len(activities) - active_count
    cluster_type = "Active-dominant" if active_count > static_count else "Static-dominant"
    print(f"Cluster {cluster_id}: {cluster_type} ({active_count} active, {static_count} static)")