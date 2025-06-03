import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Paths
RAW_DIR = "data/mnist/raw"
PROCESSED_DIR = "data/mnist/processed"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")
SPLIT_DIR = os.path.join(TRAIN_DIR, "split_data")

# Ensure directories exist
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(os.path.join(RAW_DIR, "mnist_train.csv"))
test_df = pd.read_csv(os.path.join(RAW_DIR, "mnist_test.csv"))

# Separate labels
y_train = train_df['label']
X_train = train_df.drop(columns=['label'])

y_test = test_df['label']
X_test = test_df.drop(columns=['label'])

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save full processed datasets
pd.DataFrame(X_train_pca).to_csv(os.path.join(TRAIN_DIR, "X_train_pca.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(TRAIN_DIR, "y_train.csv"), index=False)

pd.DataFrame(X_test_pca).to_csv(os.path.join(TEST_DIR, "X_test_pca.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(TEST_DIR, "y_test.csv"), index=False)

# Split X_train_pca into 60 parts
split_data = np.array_split(X_train_pca, 60)
for i, part in enumerate(split_data, start=1):
    split_path = os.path.join(SPLIT_DIR, f"X_train_{i}_pca.csv")
    pd.DataFrame(part).to_csv(split_path, index=False)

print("Preprocessing complete.")
