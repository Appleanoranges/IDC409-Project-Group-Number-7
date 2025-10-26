import pandas as pd
import numpy as np

df = pd.read_csv("data_hep.csv")
df["label"] = df["type"].apply(lambda x: 0 if x in [0, 1] else 1)
y = df.pop('label')      
event_type = df.pop('type')
'''print("Remaining columns for features:")
print(df.columns)'''

# Example: suppose df contains only your features (no labels)
df_norm = (df - df.mean()) / df.std()
# Convert to numpy array
X = df_norm.values

# Subtract mean again just in case
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Eigen decomposition
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

# Sort eigenvalues (and associated vectors) descending
sorted_idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[sorted_idx]
eig_vecs = eig_vecs[:, sorted_idx]

# Choose top k principal components
k = 18
principal_components = eig_vecs[:, :k]

# eig_vals: sorted list/array of all eigenvalues (descending)
efficiency = np.sum(eig_vals[:k]) / np.sum(eig_vals)
print(f"Fraction of variance explained by the first {k} PCs: {efficiency:.4f}")

# Project data X_centered onto the top k principal components
X_pca = X_centered.dot(principal_components)

# Convert to DataFrame for easy saving
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(k)])

# Optional: add the original labels back if you want
df_pca['label'] = y.values  # or however you have labels stored

# Save to CSV
df_pca.to_csv('pca_transformed_data.csv', index=False)

print("PCA transformed data saved to 'pca_transformed_data.csv'")

