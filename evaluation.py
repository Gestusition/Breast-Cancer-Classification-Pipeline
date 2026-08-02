"""
Model Değerlendirmesi ve Açıklanabilirlik
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from config import OUTPUTS_DIR


def evaluate_final_model(model, X_test, y_test, best_model_name):
    """Test seti uzerinde final modeli degerlendirir ve confusion matrix kaydeder."""
    print("\n" + "=" * 70)
    print("15. TEST DEGERLENDIRMESI")
    print("=" * 70)

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Performans Metrikleri (pozitif sinif = malignant):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"                   Tahmin: Benign  Tahmin: Malignant")
    print(f"  Gercek: Benign       {cm[0, 0]:^13}  {cm[0, 1]:^17}")
    print(f"  Gercek: Malignant    {cm[1, 0]:^13}  {cm[1, 1]:^17}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Benign (0)", "Malignant (1)"], zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gercek Deger")
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\n'outputs/confusion_matrix.png' kaydedildi.")

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "roc_auc": roc_auc,
    }
    return metrics, cm


def explain_model(model, X_train, y_train, X_test, y_test, best_model_name):
    """Feature importance ve SHAP analizi."""
    print("\n" + "=" * 70)
    print("16-17. ACIKLANABILIRLIK ANALIZI (Bonus)")
    print("=" * 70)

    X_train_processed = model.named_steps["add_categories"].transform(X_train)
    X_train_processed = model.named_steps["preprocessor"].transform(X_train_processed)
    selector_mask = model.named_steps["selector"].get_support()

    try:
        all_feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        all_feature_names = np.array([f"feature_{i}" for i in range(X_train_processed.shape[1])])

    selected_feature_names = np.array(all_feature_names)[selector_mask]
    X_train_selected = X_train_processed.iloc[:, selector_mask]
    X_test_processed = model.named_steps["add_categories"].transform(X_test)
    X_test_processed = model.named_steps["preprocessor"].transform(X_test_processed)
    X_test_selected = X_test_processed.iloc[:, selector_mask]

    try:
        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            importances = model.named_steps["classifier"].feature_importances_
        elif hasattr(model.named_steps["classifier"], "coef_"):
            coef = model.named_steps["classifier"].coef_
            importances = np.abs(coef).flatten()
        else:
            from sklearn.inspection import permutation_importance
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, scoring="f1"
            )
            importances = perm_result.importances_mean
    except Exception:
        from sklearn.inspection import permutation_importance
        perm_result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, scoring="f1"
        )
        importances = perm_result.importances_mean

    n_features = len(selected_feature_names)
    importances = np.array(importances)
    if len(importances) != n_features:
        importances = importances[:n_features]

    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(20, n_features)

    print(f"\nEn onemli {top_n} oznitelik:")
    for i in range(top_n):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {selected_feature_names[idx]:40s}  {importances[idx]:.6f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    top_features = [selected_feature_names[i] for i in sorted_idx[:top_n]]
    top_importances = importances[sorted_idx[:top_n]]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, top_n))
    ax.barh(range(top_n), top_importances[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_xlabel("Onem Degeri")
    ax.set_title(f"En Onemli {top_n} Oznitelik - {best_model_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"\n'outputs/feature_importance.png' kaydedildi.")

    shap_success = False
    try:
        import shap

        n_background = min(50, X_train_selected.shape[0])
        background = X_train_selected.iloc[:n_background]
        eval_sample = X_test_selected.iloc[:min(50, X_test_selected.shape[0])]
        best_clf = model.named_steps["classifier"]

        if best_model_name == "KNN":
            print("\nSHAP: KNN icin KernelExplainer cok yavas; atlaniyor.")
        elif best_model_name == "Random Forest":
            explainer = shap.TreeExplainer(best_clf)
            shap_values = explainer.shap_values(eval_sample)
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values_to_plot, eval_sample,
                              feature_names=selected_feature_names.tolist(),
                              show=False)
            fig.tight_layout()
            fig.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            shap_success = True
            print(f"\n'outputs/shap_summary.png' kaydedildi (SHAP TreeExplainer).")
        elif best_model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(best_clf, background)
            shap_values = explainer.shap_values(eval_sample)

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, eval_sample,
                              feature_names=selected_feature_names.tolist(),
                              show=False)
            fig.tight_layout()
            fig.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            shap_success = True
            print(f"\n'outputs/shap_summary.png' kaydedildi (SHAP LinearExplainer).")
        else:
            from sklearn.inspection import permutation_importance as perm_imp
            perm_result = perm_imp(
                model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1"
            )
            print("\nSHAP basarisiz; permutation importance kullanildi.")
    except Exception as e:
        print(f"\nSHAP analizi basarisiz oldu: {e}")

    shap_path = OUTPUTS_DIR / "shap_summary.png"
    if not shap_success and shap_path.exists():
        shap_path.unlink()
    if not shap_success:
        print("shap_summary.png olusturulmadi.")

    return selected_feature_names, importances
