"""
Model Training and Hyperparameter Tuning
"""

import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config import OUTPUTS_DIR
from transformers import RadiusCategoryTransformer
from preprocessing import build_scaled_preprocessor, build_unscaled_preprocessor


MODEL_TIE_BREAK_PRIORITY = {
    "Logistic Regression": 0,
    "KNN": 1,
    "Random Forest": 2,
}


def compare_models(pipelines, X_train, y_train, X_val, y_val):
    """Trains all models and compares them on the validation set.
    Selection metric: malignant F1-score, tie-breaking on recall and explicit priority.
    """
    print("\n" + "=" * 70)
    print("11-12. MODEL TRAINING AND VALIDATION COMPARISON")
    print("=" * 70)

    results = []
    best_score_tuple = (-1.0, -1.0, float("-inf"))
    best_model_name = None

    for name, pipeline in pipelines.items():
        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        try:
            y_proba = pipe.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_proba)
        except (AttributeError, IndexError, ValueError):
            roc_auc = float("nan")

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
        })

        tie_break_priority = MODEL_TIE_BREAK_PRIORITY.get(name, len(pipelines))
        score = (f1, rec, -tie_break_priority)
        if score > best_score_tuple:
            best_score_tuple = score
            best_model_name = name

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["F1-Score", "Recall"], ascending=[False, False])
    results_df = results_df.reset_index(drop=True)

    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved as 'outputs/model_comparison.csv'.")
    print(
        f"\nBest model: {best_model_name} "
        f"(F1-Score: {best_score_tuple[0]:.4f})"
    )

    return results_df, best_model_name


def tune_hyperparameters(best_model_name, X_train, y_train):
    """Performs hyperparameter tuning for the best model using GridSearchCV."""
    print("\n" + "=" * 70)
    print("13. HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
            "selector__k": [10, 15, 20, 25],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
            "selector__k": [10, 15, 20, 25],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7, None],
            "classifier__min_samples_split": [2, 5, 10],
            "selector__k": [10, 15, 20, 25],
        },
    }

    if best_model_name not in param_grids:
        print(f"Warning: No parameter grid found for {best_model_name}.")
        return None, {}

    if best_model_name == "Random Forest":
        preprocessor = build_unscaled_preprocessor()
    else:
        preprocessor = build_scaled_preprocessor()

    pipeline = Pipeline([
        ("add_categories", RadiusCategoryTransformer()),
        ("preprocessor", clone(preprocessor)),
        ("selector", SelectKBest(score_func=f_classif, k=15)),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42) if best_model_name == "Logistic Regression"
         else KNeighborsClassifier() if best_model_name == "KNN"
         else RandomForestClassifier(random_state=42)),
    ])

    param_grid = param_grids[best_model_name]
    print(f"\nParameter grid for {best_model_name}:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest cross-validation F1-score: {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k} = {v}")

    return grid_search, grid_search.best_params_


def train_final_model(best_model_name, best_params, X_train, X_val, y_train, y_val):
    """Trains the final model on train+validation using the best hyperparameters."""
    print("\n" + "=" * 70)
    print("14. FINAL MODEL TRAINING (Train + Validation)")
    print("=" * 70)

    X_train_final = pd.concat([X_train, X_val], ignore_index=True)
    y_train_final = pd.concat([y_train, y_val], ignore_index=True)

    print(f"Train + Validation combined: {X_train_final.shape[0]} rows")

    if best_model_name == "Random Forest":
        preprocessor = build_unscaled_preprocessor()
        classifier = RandomForestClassifier(random_state=42)
    elif best_model_name == "KNN":
        preprocessor = build_scaled_preprocessor()
        classifier = KNeighborsClassifier()
    else:
        preprocessor = build_scaled_preprocessor()
        classifier = LogisticRegression(max_iter=3000, random_state=42)

    pipeline = Pipeline([
        ("add_categories", RadiusCategoryTransformer()),
        ("preprocessor", clone(preprocessor)),
        ("selector", SelectKBest(score_func=f_classif, k=15)),
        ("classifier", clone(classifier)),
    ])

    if best_params:
        pipeline.set_params(**best_params)

    pipeline.fit(X_train_final, y_train_final)
    print("Final model trained.")

    return pipeline
