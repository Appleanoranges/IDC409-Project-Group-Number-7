import pandas as pd
import numpy as np
# Load your dataset
df = pd.read_csv('data_hep.csv')
df["label"] = df["type"].apply(lambda x: 0 if x in [0, 1] else 1)
# Remove non-feature columns like labels and event types if present
df_features = df.drop(columns=['label', 'type'], errors='ignore')

corr_matrix = df_features.corr().abs()

# Use numpy.tril to get lower triangle mask
upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

threshold = 0.9
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

print(f"Highly correlated features to remove: {to_drop}")

df_reduced = df_features.drop(columns=to_drop)
# Add the label and type columns back
df_reduced['label'] = df['label']
df_reduced['type'] = df['type']

# Save to CSV
print(f"Shape before removing features: {df_features.shape}")
print(f"Shape after removing features: {df_reduced.shape}")
df_reduced.to_csv('df_reduced_output.csv', index=False)
