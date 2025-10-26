import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Load your CSV file
df = pd.read_csv("df_reduced_output.csv")
#df= pd.read_csv("pca_transformed_data.csv")
#df= pd.read_csv("data_hep.csv")
#df["label"] = df["type"].apply(lambda x: 0 if x in [0, 1] else 1)
# Prepare features and labels
X = df.drop(columns=["label", "type"], errors="ignore")
y = df["label"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the neural network model
model = MLPClassifier(
    hidden_layer_sizes=(128,64,32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate on test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Get model predicted probabilities for the positive class (signal)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Separate predictions for signal and background
signal_scores = y_pred_proba[y_test == 1]
background_scores = y_pred_proba[y_test == 0]

# Histogram plot
plt.figure(figsize=(8, 6))
plt.hist(signal_scores, bins=50, range=(0, 1), color='r', alpha=0.6, label='Signal', density=False)
plt.hist(background_scores, bins=50, range=(0, 1), color='b', alpha=0.6, label='Background', density=False)

plt.title('Signal vs Background Event Distribution')
plt.xlabel('Model Predicted Probability')
plt.ylabel('Number of Events')
plt.legend(loc='upper center')
plt.grid(True)
plt.show()

# Optionally calculate and print AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {roc_auc:.4f}")
