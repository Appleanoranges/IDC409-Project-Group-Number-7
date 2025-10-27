import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report


# Load your dataset
df = pd.read_csv("pca_transformed_data.csv")
#df= pd.read_csv("pca_transformed_data.csv")

# Prepare features and labels
X = df.drop(columns=["label", "type"], errors="ignore")
y = df["label"]
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train BDT
bdt = GradientBoostingClassifier(n_estimators=100, max_depth=3)

# Cross-validation 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(bdt, X_train, y_train, cv=kf, scoring='roc_auc')

print("Cross-validation AUC scores:", auc_scores)
print(f"Mean AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# Train final model on all training data 
bdt.fit(X_train, y_train)

# Evaluate performance
y_pred_prob = bdt.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'BDT (AUC={auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Continuum Suppression')
plt.legend()
plt.show()

# Classification report
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['Signal', 'Continuum']))


# Add BDT score to dataframe or df
df['BDTscore'] = bdt.predict_proba(X)[:, 1]


# Plot BDT score distribution
plt.figure(figsize=(8,6))
plt.hist(df[df['label']==0]['BDTscore'], bins=30, range=(0,1), histtype='step', label='Signal')
plt.hist(df[df['label']==1]['BDTscore'], bins=30, range=(0,1), histtype='step', label='Continuum')
plt.xlabel('BDT score')
plt.ylabel('Number of events')
plt.title('BDT score distributions')
plt.legend()
plt.show()

# Save results
df.to_csv("continuum_with_BDT_CV.csv", index=False)
print("BDT scores added and saved to continuum_with_BDT_CV.csv")
