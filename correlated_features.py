import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load your dataset
df = pd.read_csv('data_hep.csv')
df["label"] = df["type"].apply(lambda x: 0 if x in [0, 1] else 1)
# Remove non-feature columns like labels and event types if present
df_features = df.drop(columns=['label', 'type'], errors='ignore')

# Correlation matrix before reduction 
plt.figure(figsize=(12, 10))
sns.heatmap(df_features.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Before Feature Reduction')
plt.show()

corr_matrix = df_features.corr().abs()

# Use numpy.tril to get lower triangle mask
upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

threshold = 0.9
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

print(f"Highly correlated features to remove (threshold > {threshold}): {to_drop}")

# Drop the identified features
df_reduced = df_features.drop(columns=to_drop)

# Add the label and type columns back
df_reduced['label'] = df['label']
df_reduced['type'] = df['type']

# Ensures the output CSV is clean of any unwanted index columns if they appear
initial_cols = df_reduced.shape[1]
df_reduced = df_reduced.loc[:, ~df_reduced.columns.str.contains('^Unnamed')]
if df_reduced.shape[1] < initial_cols:
    print(f"Removed {initial_cols - df_reduced.shape[1]} 'Unnamed' columns from df_reduced.")

# Correlation matrix after reduction
plt.figure(figsize=(12, 10))
sns.heatmap(df_reduced.drop(columns=['label', 'type'], errors='ignore').corr(),
            annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix After Feature Reduction')
plt.show()


# save to CSV
print(f"Shape before removing features: {df_features.shape}")
print(f"Shape after removing features: {df_reduced.shape}")
df_reduced.to_csv('df_reduced_output.csv', index=False)
print("\nReduced DataFrame saved to 'df_reduced_output.csv'")