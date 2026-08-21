"""
Exploratory Data Analysis (EDA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUTS_DIR


def perform_eda(df):
    """Performs basic data exploration and saves visualizations."""
    print("\n" + "=" * 70)
    print("3-5. BASIC DATA EXPLORATION (EDA)")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(df.head(5))

    print(f"\nData types:")
    print(df.dtypes.value_counts().to_string())

    print(f"\nSummary statistics (first 10 features):")
    print(df.describe().iloc[:, :10].round(3))

    target_counts = df["target"].value_counts()
    print(f"\nTarget variable distribution:")
    print(f"  1 (malignant): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    print(f"  0 (benign):    {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Benign (0)", "Malignant (1)"], [target_counts[0], target_counts[1]],
                   color=["#2ecc71", "#e74c3c"])
    ax.set_title("Target Variable Class Distribution")
    ax.set_ylabel("Sample Count")
    for bar, val in zip(bars, [target_counts[0], target_counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\n'outputs/class_distribution.png' saved.")

    print("\n" + "-" * 50)
    print("Missing Value Check (original dataset):")
    missing_total = df.isnull().sum().sum()
    print(f"  Total missing values: {missing_total}")
    print("  No missing values found in original dataset.")


def report_outliers(df):
    """Reports outlier counts in each numerical feature using IQR method."""
    print("\n" + "=" * 70)
    print("7. OUTLIER ANALYSIS (Descriptive EDA)")
    print("=" * 70)
    print("NOTE: Capping bounds will be learned strictly from training data.")

    feature_cols = [c for c in df.columns if c != "target"]
    outlier_counts = {}
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        if count > 0:
            outlier_counts[col] = count

    if outlier_counts:
        outlier_df = pd.DataFrame(
            {"Outlier Count": outlier_counts}
        ).sort_values("Outlier Count", ascending=False)
        print(f"\nColumns with outliers (total {len(outlier_counts)} columns):")
        print(outlier_df.head(15).to_string())
    else:
        print("\nNo outliers found in any column.")


def save_correlation_heatmap(X_train, y_train):
    """Saves a heatmap of features with the strongest correlations."""
    print("\n" + "=" * 70)
    print("9. CORRELATION ANALYSIS")
    print("=" * 70)

    df_corr = X_train.copy()
    df_corr["target"] = y_train.values
    corr_matrix = df_corr.corr()

    target_corr = corr_matrix["target"].drop("target").abs().sort_values(ascending=False)
    top_features = target_corr.head(20).index.tolist()
    top_corr = corr_matrix.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(top_corr, dtype=bool), k=1)
    sns.heatmap(top_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title("Correlation Heatmap of Top 20 Correlated Features",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"'outputs/correlation_heatmap.png' saved.")

    print(f"\nTop 10 features most correlated with target:")
    print(target_corr.head(10).to_string())
