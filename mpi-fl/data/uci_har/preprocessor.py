import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Define paths
base_dir = './data/uci_har/UCI HAR Dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
output_dir = './data/uci_har/processed_data'
os.makedirs(output_dir, exist_ok=True)

# Load data function
def load_split(data_dir, split):
    X = pd.read_csv(os.path.join(data_dir, f'X_{split}.txt'), delim_whitespace=True, header=None)
    y = pd.read_csv(os.path.join(data_dir, f'y_{split}.txt'), header=None)
    subjects = pd.read_csv(os.path.join(data_dir, f'subject_{split}.txt'), header=None)
    return X, y, subjects

# Load train and test data
X_train, y_train, subj_train = load_split(train_dir, 'train')
X_test, y_test, subj_test = load_split(test_dir, 'test')

# Fit scaler and PCA on full training + test set
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(pd.concat([X_train, X_test], ignore_index=True))

pca = PCA(n_components=100)
X_all_pca = pca.fit_transform(X_all_scaled)

# Split PCA-transformed data back into train/test
X_train_pca = X_all_pca[:len(X_train)]
X_test_pca = X_all_pca[len(X_train):]

# Save full train/test sets
pd.DataFrame(X_train_pca).to_csv(os.path.join(output_dir, 'X_train_pca.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=False)

pd.DataFrame(X_test_pca).to_csv(os.path.join(output_dir, 'X_test_pca.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=False)

# Save training data split by subject
for subject_id in sorted(subj_train[0].unique()):
    subject_mask = (subj_train[0] == subject_id)
    subject_data = X_train_pca[subject_mask.values]
    pd.DataFrame(subject_data).to_csv(os.path.join(output_dir, f'subject_{subject_id}.csv'), index=False)

print(f"Saved X/y train/test and subject-level training data to: {output_dir}")