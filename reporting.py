"""
Result Reporting
"""

from config import OUTPUTS_DIR


def print_result_summary(best_model_name, validation_df, test_metrics, best_params):
    """Prints comprehensive evaluation summary and interpretation."""
    print("\n" + "=" * 70)
    print("18. RESULTS & CONCLUSION")
    print("=" * 70)

    matching_rows = validation_df.loc[validation_df["Model"] == best_model_name]
    if matching_rows.empty:
        raise ValueError(f"Selected model not found in validation table: {best_model_name}")
    best_val = matching_rows.iloc[0]

    print(f"""
SUMMARY EVALUATION
------------------
Best Model: {best_model_name}

Validation Results:
  Accuracy:  {best_val['Accuracy']:.4f}
  Precision: {best_val['Precision']:.4f}
  Recall:    {best_val['Recall']:.4f}
  F1-Score:  {best_val['F1-Score']:.4f}
  ROC-AUC:   {best_val['ROC-AUC']:.4f}

Test Results (positive class = malignant):
  Accuracy:  {test_metrics['accuracy']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall:    {test_metrics['recall']:.4f}
  F1-Score:  {test_metrics['f1_score']:.4f}
  ROC-AUC:   {test_metrics['roc_auc']:.4f}

Best Hyperparameters:
""")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k} = {v}")
    else:
        print("  (default parameters)")

    print(f"""
INTERPRETATION & CONCLUSION
---------------------------
1. Model Selection: {best_model_name} was selected because it achieved the highest
   malignant F1 score on the validation set, with recall and explicit model priority
   used as tie-breakers.

2. Malignant Recall: Measures how many of the actual malignant tumors were correctly
   identified. This is critical in medical diagnostic settings.

3. Importance of False Negatives: Misclassifying a malignant tumor as benign is
   clinically dangerous and could lead to life-threatening delays in treatment.

4. Key Features: Strongly associated with tumor size (area, radius, perimeter),
   shape irregularity (compactness, concavity), and texture.

5. Limitations:
   - The dataset contains only 569 samples; generalization performance may vary across larger populations.
   - This model is strictly for educational purposes and CANNOT be used as a clinical diagnostic tool.
""")

    print("=" * 70)
    print("Pipeline run completed.")
    print(f"Output directory: {OUTPUTS_DIR}")
    print("=" * 70)
