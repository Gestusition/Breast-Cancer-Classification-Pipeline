# Breast Cancer Machine Learning Classification Via Wisconsin Data

[![CI](https://github.com/Gestusition/Machine-Learning-Final-Odevi/actions/workflows/ci.yml/badge.svg)](https://github.com/Gestusition/Machine-Learning-Final-Odevi/actions/workflows/ci.yml)
[![Code Quality](https://github.com/Gestusition/Machine-Learning-Final-Odevi/actions/workflows/code-quality.yml/badge.svg)](https://github.com/Gestusition/Machine-Learning-Final-Odevi/actions/workflows/code-quality.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Project Purpose

This project implements an end-to-end machine learning pipeline on the **Breast Cancer Wisconsin** dataset. The objective is to perform exploratory data analysis (EDA), data preprocessing, feature engineering, model training, model comparison, cross-validation, hyperparameter tuning, model evaluation, and result interpretation within a clean, robust, and well-structured Python project.

> [!IMPORTANT]
> **Important Note:** This model is strictly for educational purposes. It CANNOT be used as a clinical diagnostic tool.

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Number of samples:** 569
- **Number of original features:** 30 (all numerical)
- **Target encoding:** `1 = malignant`, `0 = benign`
- **Class distribution:** Malignant 37.3% (212), Benign 62.7% (357)

### Original Features

30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses, describing characteristics of the cell nuclei present in the image across **mean**, **standard error**, and **worst** values:

`mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`, `radius error`, `texture error`, `perimeter error`, `area error`, `smoothness error`, `compactness error`, `concavity error`, `concave points error`, `symmetry error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst compactness`, `worst concavity`, `worst concave points`, `worst symmetry`, `worst fractal dimension`

## Installation and Execution

```bash
pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

### Verified Runtime Environment

End-to-end execution and outputs have been verified in the following environment. Since `requirements.txt` contains version ranges for broad compatibility, cross-validation scores may vary slightly across different library versions.

| Component | Version |
|---|---|
| Python | 3.14.6 |
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| shap | 0.52.0 |

When executed, the pipeline performs the following steps sequentially:
1. Load dataset, remap target encoding to malignant=1, benign=0
2. Perform EDA (`outputs/class_distribution.png`)
3. Inject ~1% synthetic missing values to demonstrate imputation
4. Feature engineering (3 new features)
5. Outlier analysis & reporting (descriptive)
6. Stratified train/validation/test split (60% / 20% / 20%)
7. Correlation heatmap (`outputs/correlation_heatmap.png`)
8. Pipelines + 3 model training & validation comparison
9. Hyperparameter tuning using GridSearchCV
10. Final model training on train+validation and evaluation on test set
11. Model explainability (feature importance + SHAP)
12. Results summary and clinical interpretation

## Engineered Features

| Feature | Formula | Description |
|---|---|---|
| `worst_to_mean_radius_ratio` | `worst radius / (mean radius + eps)` | Represents the relative difference between worst and mean radius summary metrics for the same sample; not a temporal growth rate |
| `worst_to_mean_area_ratio` | `worst area / (mean area + eps)` | Represents the relative difference between worst and mean area summary metrics for the same sample; not a temporal growth rate |
| `worst_minus_mean_compactness` | `worst compactness - mean compactness` | Represents the difference between worst and mean compactness summary metrics for the same sample |

Note: `mean` and `worst` metrics are not observations taken at different points in time; they are summary statistics computed for each FNA sample.

## Synthetic Categorical Feature

To demonstrate categorical preprocessing and encoding pipelines for educational purposes, a synthetic categorical feature is created:

- `radius_category`: Quantile-based 3-bin discretization of the `mean radius` feature (`small` / `medium` / `large`).
- Handled via custom scikit-learn transformer `RadiusCategoryTransformer` inside the pipeline.
- Bin edges are strictly fitted on the training set and applied to validation/test sets to prevent data leakage.
- Converted into one-hot numeric representation via `OneHotEncoder`.

## Synthetic Missing Value Injection

The original dataset has NO missing values. To demonstrate imputation pipelines, ~1% synthetic missing values (5 values per column) are injected into 5 randomly selected numerical columns using a fixed `random_state=42`:

| Column | Missing Count |
|---|---|
| mean perimeter | 5 |
| perimeter error | 5 |
| symmetry error | 5 |
| worst radius | 5 |
| worst fractal dimension | 5 |

Missing values are handled inside the pipeline via `SimpleImputer(strategy="median")` fitted solely on the training data.

## Preprocessing Steps

All preprocessing steps are encapsulated in scikit-learn `Pipeline` and `ColumnTransformer` objects. **Data leakage is strictly prevented**:
- Imputer, OutlierCapper, StandardScaler, and OneHotEncoder are **fitted only** on the training partition.
- Validation and test sets are exclusively **transformed**.
- GridSearchCV refits preprocessing pipelines per training fold.

### Pipeline Structure

**Logistic Regression and KNN:**
```
RadiusCategoryTransformer
  -> ColumnTransformer[
       numeric: SimpleImputer(median) -> OutlierCapper -> StandardScaler
       categorical: SimpleImputer(most_frequent) -> OneHotEncoder
     ]
  -> SelectKBest(k=15, score_func=f_classif)
  -> Classifier
```

**Random Forest:**
```
RadiusCategoryTransformer
  -> ColumnTransformer[
       numeric: SimpleImputer(median) -> OutlierCapper
       categorical: SimpleImputer(most_frequent) -> OneHotEncoder
     ]
  -> SelectKBest(k=15, score_func=f_classif)
  -> RandomForestClassifier
```

Random Forest does not require numerical feature scaling, so `StandardScaler` is omitted from its numeric branch.

### Outlier Handling

- During EDA, outliers across all numerical columns are reported using the IQR method (descriptive).
- Capping bounds (`OutlierCapper`) are learned strictly on the training set.
- IQR factor: 1.5.

## Train-Validation-Test Split Ratios

| Partition | Row Count | Ratio |
|---|---|---|
| Train | 341 | 60% |
| Validation | 114 | 20% |
| Test | 114 | 20% |

All splits use `stratify=y` to preserve the positive (malignant) class proportion across subsets.

## Compared Models

Three classification algorithms are trained and evaluated:

1. **Logistic Regression** (`max_iter=3000`, L2 regularization)
2. **K-Nearest Neighbors (KNN)** (`n_neighbors=5`)
3. **Random Forest** (`n_estimators=100`)

## Validation Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9846 |
| KNN | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9728 |
| Random Forest | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9923 |

All three models achieved identical Accuracy, Precision, Recall, and F1-Score on the validation set. In the case of a tie, the predefined model priority order **Logistic Regression > KNN > Random Forest** was used, selecting Logistic Regression.

Model selection criteria: Malignant F1-Score, tie-break on malignant Recall, secondary tie-break on explicit model priority.

## Hyperparameter Tuning

`GridSearchCV` was applied to the best model (Logistic Regression):

- **CV Strategy:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Scoring:** `f1` (malignant F1-score)
- **Best CV F1-Score:** 0.9724

### Best Hyperparameters

| Parameter | Value |
|---|---|
| `classifier__C` | 100 |
| `classifier__penalty` | l2 |
| `classifier__solver` | lbfgs |
| `selector__k` | 25 |

## Test Results (Selected Model: Logistic Regression)

| Metric | Value |
|---|---|
| Accuracy | 0.9737 |
| Precision | 0.9535 |
| Recall | 0.9762 |
| F1-Score | 0.9647 |
| ROC-AUC | 0.9954 |

### Confusion Matrix (Test Set)

| | Predicted: Benign | Predicted: Malignant |
|---|---|---|
| **Actual: Benign** | 70 | 2 |
| **Actual: Malignant** | 1 | 41 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Benign (0) | 0.99 | 0.97 | 0.98 | 72 |
| Malignant (1) | 0.95 | 0.98 | 0.96 | 42 |

## Key Features (Logistic Regression Absolute Coefficient Magnitudes)

| Rank | Feature | Absolute Coefficient Magnitude |
|---|---|---|
| 1 | num__worst radius | 13.0803 |
| 2 | num__mean concave points | 9.4186 |
| 3 | num__worst_to_mean_area_ratio | 7.0708 |
| 4 | num__area error | 6.9708 |
| 5 | num__worst_to_mean_radius_ratio | 6.1277 |
| 6 | num__worst concavity | 3.9796 |
| 7 | num__mean compactness | 3.9161 |
| 8 | num__mean area | 3.0456 |
| 9 | num__mean concavity | 2.8029 |
| 10 | num__mean radius | 2.7394 |

The most important features are strongly associated with tumor shape irregularity (concave points, concavity), tumor size (radius, area), and engineered ratio features. Values represent absolute coefficient magnitudes of standardized features, providing relative ranking of feature importance.

## Model Limitations

1. **Small Dataset Size:** 569 samples; generalization performance should be validated on larger external clinical cohorts.
2. **Educational Purpose:** This model is designed for educational demonstration and MUST NOT be used for real clinical medical diagnostics.
3. **Outlier Capping:** High biomarker values in malignant cases may represent genuine pathological signals; outlier capping must be interpreted cautiously.
4. **False Negative Risk:** 1 malignant case was misclassified as benign in the test set (Recall = 97.62%, 41 out of 42 malignant cases detected). In real medical diagnostics, every false negative carries significant clinical risk.

## Files in outputs/ Directory

| File | Description |
|---|---|
| `class_distribution.png` | Target class distribution bar chart |
| `correlation_heatmap.png` | Top 20 correlated features heatmap |
| `model_comparison.csv` | Validation comparison results table |
| `confusion_matrix.png` | Test confusion matrix visualization |
| `feature_importance.png` | Top 20 most important features horizontal bar chart |
| `shap_summary.png` | SHAP explainability summary plot (LinearExplainer for Logistic Regression) |

## Conclusion and Interpretation

Logistic Regression was chosen based on validation performance and model simplicity. On the independent test set, it achieved **97.37% Accuracy**, **97.62% Malignant Recall**, and **0.9954 ROC-AUC**. Out of 42 malignant cases, 41 were correctly identified with only 1 false negative and 2 false positives (Precision: 95.35%).

Concave points, tumor radius, area, and engineered ratio features (`worst_to_mean_area_ratio`, `worst_to_mean_radius_ratio`) are among the most influential predictors. The synthetic `radius_category_large` feature also ranked among the top 20 predictors.

A key advantage of Logistic Regression is direct interpretability through its coefficients and compatibility with SHAP `LinearExplainer`, enabling clear insight into the model's decision boundaries.

## License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Users/kirac/Desktop/Machine%20Learning%20final%20%C3%B6dev/LICENSE) file for details.

