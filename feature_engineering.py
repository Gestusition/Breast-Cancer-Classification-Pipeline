"""
Öznitelik Mühendisliği
"""

import numpy as np
import pandas as pd


def inject_missing_values(df):
    """Imputation islemini gostermek amaciyla sayisal ozniteliklere ~%1
    oraninda yapay eksik deger ekler. Target sutununa dokunulmaz.
    """
    print("\n" + "=" * 70)
    print("5b. YAPAY EKSIK DEGER ENJEKSIYONU (Imputation Gostermek Icin)")
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

    print(f"Enjeksiyon yapilan sutunlar: {list(cols_to_inject)}")
    print(f"Her sutuna eklenen eksik deger sayisi: {n_missing} (~%1)")

    missing_after = df_missing.isnull().sum()
    missing_after_nonzero = missing_after[missing_after > 0]
    print(f"\nEnjeksiyon sonrasi eksik degerler:")
    print(missing_after_nonzero.to_string())

    return df_missing


def engineer_features(df):
    """Anlamli yeni oznitelikler uretir:
      - radius_growth_ratio: worst radius / mean radius
      - area_growth_ratio: worst area / mean area
      - compactness_change: worst compactness - mean compactness
    """
    print("\n" + "=" * 70)
    print("6. OZNITELIK MUHENDISLIGI")
    print("=" * 70)

    eps = 1e-10
    df_fe = df.copy()

    df_fe["radius_growth_ratio"] = df_fe["worst radius"] / (df_fe["mean radius"] + eps)
    print("  + radius_growth_ratio = worst radius / mean radius")

    df_fe["area_growth_ratio"] = df_fe["worst area"] / (df_fe["mean area"] + eps)
    print("  + area_growth_ratio = worst area / mean area")

    df_fe["compactness_change"] = df_fe["worst compactness"] - df_fe["mean compactness"]
    print("  + compactness_change = worst compactness - mean compactness")

    print(f"\nOznitelik muhendisligi sonrasi veri seti: {df_fe.shape[1]} sutun")
    return df_fe
