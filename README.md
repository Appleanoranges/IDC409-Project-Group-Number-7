# IDC409-Project-Group-Number-7
## Continuum Suppression in High Energy Physics

***

### Project Overview
This project focuses on **continuum suppression** in high‑energy physics experiments using data recorded by the **Belle II detector**.  
The aim is to distinguish **Signal** events (detection and identification of B-mensos) from **Background** or **Continuum** events using machine‑learning techniques.  

We compared several models like — **Logistic Regression**, **Boosted Decision Trees (BDT)**, and **Deep Neural Networks (DNN)** — combined with **feature‑reduction** techniques (PCA and Correlation‑based elimination).

***

### Dataset Description
- We were provided with the data with each row as one recorded event and each column = feature values (event‑level variables such as thrust, R2, etc.)
- We were asked to put flag 0 and 1 in first category and 2 to 5 no flag in second category. Category one is signal event while category 2 is background event.
 
  
***

### Data Preprocessing
1. Removing non‑feature columns (`label`, `type`) before analysis.  
2. Normalizing features before training (mean 0, std 1).  
3. Apply dimensionality‑reduction / feature‑selection before model training (either PCA or Correlation filtering).

***

### Feature Reduction Methods

#### (1) Principal Component Analysis (PCA)
# Steps
1. Standardize dataset → mean 0, std 1.
2. Compute covariance matrix and eigen decomposition.
3. Sort eigenvectors by descending eigenvalues.
4. Select top k components (e.g., k = 18) that explain maximum variance.
```

**Outcome:**
- Transforms correlated features into new uncorrelated “principal components.”  
- Captures maximum variance while reducing dimensions.  
- Fraction of total variance explained is printed after transformation:
  ```
  Fraction of variance explained by the first 18 PCs: 0.94
  ```
- The transformed dataset is saved as:
  ```
  pca_transformed_data.csv
  ```

***

#### (2) Correlation‑Based Feature Elimination

# Steps
1. Compute absolute correlation matrix.
2. For each pair of features with correlation > 0.9, drop one.
3. Keep label and type columns.
4. Save reduced dataset to "df_reduced_output.csv"
```

**Outcome:**
- Removes only the most redundant variables, keeping others unchanged.  
- Preserves physical meaning of remaining features.



***

### Machine Learning Models Used

#### 1. Logistic Regression (Implemented manually)
- Uses gradient descent with L2 regularization.  
- Learns linear decision boundary between Signal and Background.  
- Evaluates metrics via **Stratified K‑Fold Cross‑Validation**.  
- Metrics computed for each fold: Accuracy, Precision, Recall, F1, ROC‑AUC.  
- Produces confusion matrices for error analysis.

**Notable Functions**
```python
gradient_descent()
sigmoid()
accuracy(), precision(), recall(), f1_score()
confusion_matrix()
```

**Outputs**
- Fold‑wise results with average metrics.
- ROC curve for final model.
- Distribution of logistic regression probabilities for both event classes.

***

#### 2. Boosted Decision Trees (BDT)
- Implemented with `GradientBoostingClassifier` from scikit‑learn.  
- Parameters: 100 estimators, max_depth = 3.  
- 5‑fold cross‑validated AUC computation.  
- Visualizes ROC curves and event‑score distributions.  
- Adds computed BDT scores to the dataset as a new column:
  ```
  df['BDTscore'] = model.predict_proba(X)[:, 1]
  ```
- Results stored in:
  ```
  continuum_with_BDT_CV.csv
  ```


***

#### 3. Deep Neural Network (DNN)
- Implemented using `MLPClassifier`.  
- Architecture: 3 hidden layers (128, 64, 32), ReLU activation, Adam optimizer.  
- Trained for 500 epochs with early‑stopping stability.  
- Uses standardized input data.  
- Outputs include Accuracy, Classification Report, ROC curve, AUC, and event distributions.


***

### Model Evaluation
Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC‑AUC**
- **Confusion Matrix**
  
Additionally, for every algorithm:
- ROC curves show discriminative power.  
- Histograms of Score/Probability distributions visualize separation between signal and background events.

***


### Results Summary (Example AUC Comparison)

| Model | PCA Dataset | Correlation Dataset |
|--------|--------------|----------------|
| Logistic Regression | 0.978 | 0.907 |
| BDT | 0.98 | 0.933 |
| DNN | 0.974 | 0.9104 |

***

### Conclusion
- All models demonstrated strong discrimination between Signal and Background, with ROC‑AUC values ranging between 0.90 to 0.98.

PCA emerged as the superior feature‑reduction approach for performance enhancement, especially for statistical modeling.

BDT provided the most powerful classifier, marginally outperforming others in both AUC and interpretability.

Logistic Regression remains an excellent interpretable baseline, and DNN adds flexibility for future large‑scale datasets.
