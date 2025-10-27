import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Load your dataset
df = pd.read_csv("pca_transformed_data.csv")
#df= pd.read_csv("pca_transformed_data.csv")

# Prepare features and labels
X = df.drop(columns=["label", "type"], errors="ignore")
y = df["label"]

# 4. Manual logistic regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X_mat, y_vec, lr=0.1, n_iter=500, lambda_reg=0.01):
    w = np.zeros(X_mat.shape[1])
    for _ in range(n_iter):
        z = np.dot(X_mat, w)
        y_pred = sigmoid(z)
        grad = np.dot(X_mat.T, (y_pred - y_vec)) / len(y_vec)
        regularization_term = (lambda_reg / len(y_vec)) * w
        regularization_term[0] = 0
        grad += regularization_term
        w -= lr * grad
    return w

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*p*r / (p + r) if (p + r) > 0 else 0.0

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

# 5. Stratified K-fold cross-validation
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1
    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    X_train_final = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_final = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    w = gradient_descent(X_train_final, y_train, lr=0.1, n_iter=500, lambda_reg=0.01)

    probs = sigmoid(np.dot(X_test_final, w))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy(y_test, preds)
    prec = precision(y_test, preds)
    rec = recall(y_test, preds)
    f1 = f1_score(y_test, preds)

    thresholds = np.linspace(0, 1, 100)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred_t = (probs >= t).astype(int)
        TP = np.sum((pred_t == 1) & (y_test == 1))
        FP = np.sum((pred_t == 1) & (y_test == 0))
        FN = np.sum((pred_t == 0) & (y_test == 1))
        TN = np.sum((pred_t == 0) & (y_test == 0))
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    roc_points = sorted(zip(fpr_list, tpr_list))
    sorted_fpr = [p[0] for p in roc_points]
    sorted_tpr = [p[1] for p in roc_points]

    if sorted_fpr[0] != 0.0 or sorted_tpr[0] != 0.0:
        sorted_fpr.insert(0, 0.0)
        sorted_tpr.insert(0, 0.0)

    if sorted_fpr[-1] != 1.0 or sorted_tpr[-1] != 1.0:
        sorted_fpr.append(1.0)
        sorted_tpr.append(1.0)

    auc = np.trapezoid(sorted_tpr, sorted_fpr)

    metrics['accuracy'].append(acc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f1'].append(f1)
    metrics['auc'].append(auc)

    cm = confusion_matrix(y_test, preds)
    print(f"Fold {fold}: Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    print(f"Confusion Matrix (TN, FP | FN, TP):\n{cm}\n")

print("\nMetrics averaged over {} folds:".format(k))
print(f"Accuracy: {np.mean(metrics['accuracy']):.3f}")
print(f"Precision: {np.mean(metrics['precision']):.3f}")
print(f"Recall: {np.mean(metrics['recall']):.3f}")
print(f"F1 Score: {np.mean(metrics['f1']):.3f}")
print(f"AUC: {np.mean(metrics['auc']):.3f}")

print(f"\nCross-validation mean accuracy score: {np.mean(metrics['accuracy']):.3f}")

# 6. Train main model on full data and plot graphs & confusion matrix
full_mean = np.mean(X, axis=0)
full_std = np.std(X, axis=0)
full_std[full_std == 0] = 1

X_scaled = (X - full_mean) / full_std
X_final = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

w_full = gradient_descent(X_final, y, lr=0.1, n_iter=500, lambda_reg=0.01)
probs_full = sigmoid(np.dot(X_final, w_full))
preds_full = (probs_full >= 0.5).astype(int)

thresholds = np.linspace(0, 1, 100)
tpr_list, fpr_list = [], []

for t in thresholds:
    pred_t = (probs_full >= t).astype(int)
    TP = np.sum((pred_t == 1) & (y == 1))
    FP = np.sum((pred_t == 1) & (y == 0))
    FN = np.sum((pred_t == 0) & (y == 1))
    TN = np.sum((pred_t == 0) & (y == 0))
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

roc_points = sorted(zip(fpr_list, tpr_list))
sorted_fpr = [p[0] for p in roc_points]
sorted_tpr = [p[1] for p in roc_points]

if sorted_fpr[0] != 0.0 or sorted_tpr[0] != 0.0:
    sorted_fpr.insert(0, 0.0)
    sorted_tpr.insert(0, 0.0)

if sorted_fpr[-1] != 1.0 or sorted_tpr[-1] != 1.0:
    sorted_fpr.append(1.0)
    sorted_tpr.append(1.0)

auc_main = np.trapezoid(sorted_tpr, sorted_fpr)

plt.figure(figsize=(8,6))
plt.plot(sorted_fpr, sorted_tpr, label=f'ROC (AUC={auc_main:.3f})')
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Main Model, All Data)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.hist(probs_full[y==0], bins=30, range=(0,1), histtype='step', label='Signal (y=0)')
plt.hist(probs_full[y==1], bins=30, range=(0,1), histtype='step', label='Continuum (y=1)')
plt.xlabel('Logistic Regression Score')
plt.ylabel('Number of Events')
plt.title('Score Distributions (Main Model, All Data)')
plt.legend()
plt.grid(True)
plt.show()

cm_main = confusion_matrix(y, preds_full)
print(f"Confusion Matrix for full-data model (TN, FP | FN, TP):\n{cm_main}")