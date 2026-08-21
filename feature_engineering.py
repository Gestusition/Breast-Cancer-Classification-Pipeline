"""
Feature Engineering
"""

import numpy as np


def inject_missing_values(df):
    """Injects ~1% synthetic missing values into numerical features to
    demonstrate imputation. Target column is untouched.
    """
    print("\n" + "=" * 70)
    print("5b. SYNTHETIC MISSING VALUE INJECTION (To Demonstrate Imputation)")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c != "target"]
    n_rows = len(df)
    n_missing = max(1, int(n_rows * 0.01))
    rng = np.random.default_rng(42)
    cols_to_inject = rng.choice(feature_cols, size=min(5, len(feature_cols)), replace=False)

    df_missing = df.copy()
    for col in cols_to_inject:
        idx = rng.choice(n_rows, size=n_missing, replace=False)
        df_missing.loc[idx, col] = np.nan

    print(f"Injected columns: {list(cols_to_inject)}")
    print(f"Missing values added per column: {n_missing} (~1%)")

    missing_after = df_missing.isnull().sum()
    missing_after_nonzero = missing_after[missing_after > 0]
    print(f"\nMissing values after injection:")
    print(missing_after_nonzero.to_string())

    return df_missing


def engineer_features(df):
    """Generates meaningful new features:
      - worst_to_mean_radius_ratio: worst radius / mean radius
      - worst_to_mean_area_ratio: worst area / mean area
      - worst_minus_mean_compactness: worst compactness - mean compactness

    ``mean`` and ``worst`` values are summary measurements of the same sample;
    they should not be interpreted as consecutive time-series observations.
    """
    print("\n" + "=" * 70)
    print("6. FEATURE ENGINEERING")
    print("=" * 70)

    eps = 1e-10
    df_fe = df.copy()

    df_fe["worst_to_mean_radius_ratio"] = df_fe["worst radius"] / (df_fe["mean radius"] + eps)
    print("  + worst_to_mean_radius_ratio = worst radius / mean radius")

    df_fe["worst_to_mean_area_ratio"] = df_fe["worst area"] / (df_fe["mean area"] + eps)
    print("  + worst_to_mean_area_ratio = worst area / mean area")

    df_fe["worst_minus_mean_compactness"] = (
        df_fe["worst compactness"] - df_fe["mean compactness"]
    )
    print("  + worst_minus_mean_compactness = worst compactness - mean compactness")

    print(f"\nDataset after feature engineering: {df_fe.shape[1]} columns")
    return df_fe
