"""
End-to-End Machine Learning Project - Breast Cancer Classification
==================================================================

Purpose:
    Executes an end-to-end machine learning pipeline on the Breast Cancer
    Wisconsin dataset: EDA, preprocessing, feature engineering, model training,
    model comparison, hyperparameter tuning, evaluation, and explainability.

Libraries Used:
    - pandas, numpy: Data manipulation
    - scikit-learn: Preprocessing, model training, evaluation
    - matplotlib, seaborn: Visualization
    - shap: Model explainability

Execution:
    1. pip install -r requirements.txt
    2. python main.py
"""

import config  # noqa: F401

from data_loading import load_and_prepare_data
from eda import perform_eda, report_outliers, save_correlation_heatmap
from feature_engineering import inject_missing_values, engineer_features
from preprocessing import split_data, create_model_pipelines
from model_training import compare_models, tune_hyperparameters, train_final_model
from evaluation import evaluate_final_model, explain_model
from reporting import print_result_summary


def main():
    """Runs the end-to-end machine learning workflow."""
    print("=" * 70)
    print("  BREAST CANCER CLASSIFICATION - END-TO-END ML PROJECT")
    print("=" * 70)

    df = load_and_prepare_data()
    perform_eda(df)
    df = inject_missing_values(df)
    df = engineer_features(df)
    report_outliers(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    save_correlation_heatmap(X_train, y_train)

    pipelines, _, _ = create_model_pipelines()
    validation_df, best_model_name = compare_models(
        pipelines, X_train, y_train, X_val, y_val
    )

    _, best_params = tune_hyperparameters(
        best_model_name, X_train, y_train
    )

    final_model = train_final_model(
        best_model_name, best_params, X_train, X_val, y_train, y_val
    )

    test_metrics, _ = evaluate_final_model(
        final_model, X_test, y_test, best_model_name
    )

    explain_model(final_model, X_train, X_test, y_test, best_model_name)
    print_result_summary(best_model_name, validation_df, test_metrics, best_params)


if __name__ == "__main__":
    main()
