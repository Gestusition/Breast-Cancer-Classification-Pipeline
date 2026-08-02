"""
Ön İşleme ve Pipeline Oluşturma
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from transformers import RadiusCategoryTransformer, OutlierCapper


def split_data(df):
    """Veriyi stratified olarak %60 train, %20 validation, %20 test seklinde boler."""
    print("\n" + "=" * 70)
    print("8. TRAIN / VALIDATION / TEST BOLME (Stratified)")
    print("=" * 70)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    n = len(df)
    print(f"Train seti:      {X_train.shape[0]} satir ({X_train.shape[0]/n*100:.0f}%)")
    print(f"Validation seti: {X_val.shape[0]} satir ({X_val.shape[0]/n*100:.0f}%)")
    print(f"Test seti:       {X_test.shape[0]} satir ({X_test.shape[0]/n*100:.0f}%)")

    print(f"\nTrain      malignant orani: {y_train.mean():.2%}")
    print(f"Validation malignant orani: {y_val.mean():.2%}")
    print(f"Test       malignant orani: {y_test.mean():.2%}")

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_scaled_preprocessor():
    """LR ve KNN icin: SimpleImputer(median) -> OutlierCapper -> StandardScaler + OHE."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_capper", OutlierCapper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=["category", "object"])),
    ])


def build_unscaled_preprocessor():
    """Random Forest icin: SimpleImputer(median) -> OutlierCapper + OHE (scaler yok)."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_capper", OutlierCapper(factor=1.5)),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=["category", "object"])),
    ])


def create_model_pipelines():
    """Tum modeller icin sklearn Pipeline'lari olusturur."""
    print("\n" + "=" * 70)
    print("9. PIPELINE INSASI")
    print("=" * 70)

    scaled_preprocessor = build_scaled_preprocessor()
    unscaled_preprocessor = build_unscaled_preprocessor()

    pipelines = {
        "Logistic Regression": Pipeline([
            ("add_categories", RadiusCategoryTransformer()),
            ("preprocessor", clone(scaled_preprocessor)),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("add_categories", RadiusCategoryTransformer()),
            ("preprocessor", clone(scaled_preprocessor)),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": Pipeline([
            ("add_categories", RadiusCategoryTransformer()),
            ("preprocessor", clone(unscaled_preprocessor)),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    print("Logistic Regression: RadiusCategory -> Imputer->OutlierCapper->Scaler + OHE -> SelectKBest -> LR")
    print("KNN:                RadiusCategory -> Imputer->OutlierCapper->Scaler + OHE -> SelectKBest -> KNN")
    print("Random Forest:      RadiusCategory -> Imputer->OutlierCapper + OHE -> SelectKBest -> RF")

    return pipelines, scaled_preprocessor, unscaled_preprocessor
