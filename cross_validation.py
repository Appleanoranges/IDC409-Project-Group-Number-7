from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("df_reduced_output.csv")
X = df.drop(columns=["label", "type"], errors="ignore")
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                      activation='relu', solver='adam',
                      max_iter=500, random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

cv_results = cross_validate(model, X_scaled, y, cv=5,
                            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                            return_train_score=False)

for metric in cv_results:
    print(f"{metric}: {np.mean(cv_results[metric]):.4f}")
