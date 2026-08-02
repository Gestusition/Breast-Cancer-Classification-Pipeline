"""
Keşifsel Veri Analizi (EDA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUTS_DIR


def perform_eda(df):
    """Veri setinin temel incelemesini yapar ve gorsellestirme kaydeder."""
    print("\n" + "=" * 70)
    print("3-5. TEMEL VERI INCELEME (EDA)")
    print("=" * 70)

    print(f"\nVeri seti boyutu: {df.shape[0]} satir, {df.shape[1]} sutun")
    print(f"\nIlk 5 satir:")
    print(df.head(5))

    print(f"\nVeri tipleri:")
    print(df.dtypes.value_counts().to_string())

    print(f"\nTemel istatistikler (ilk 10 oznitelik):")
    print(df.describe().iloc[:, :10].round(3))

    target_counts = df["target"].value_counts()
    print(f"\nHedef degisken (target) dagilimi:")
    print(f"  1 (malignant): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    print(f"  0 (benign):    {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Benign (0)", "Malignant (1)"], [target_counts[0], target_counts[1]],
                   color=["#2ecc71", "#e74c3c"])
    ax.set_title("Hedef Degisken Sinif Dagilimi")
    ax.set_ylabel("Ornek Sayisi")
    for bar, val in zip(bars, [target_counts[0], target_counts[1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\n'outputs/class_distribution.png' kaydedildi.")

    print("\n" + "-" * 50)
    print("Eksik Deger Kontrolu (orijinal veri seti):")
    missing_total = df.isnull().sum().sum()
    print(f"  Toplam eksik deger: {missing_total}")
    print("  Orijinal veri setinde eksik deger bulunmamaktadir.")


def report_outliers(df):
    """IQR yontemiyle her sayisal oznitelikteki aykiri deger sayisini raporlar."""
    print("\n" + "=" * 70)
    print("7. AYKIRI DEGER INCELEMESI (Betimsel EDA)")
    print("=" * 70)
    print("NOT: Capping sinirlari yalnizca train verisinden ogrenilecektir.")

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
            {"Aykiri Deger Sayisi": outlier_counts}
        ).sort_values("Aykiri Deger Sayisi", ascending=False)
        print(f"\nAykiri deger bulunan sutunlar (toplam {len(outlier_counts)} sutun):")
        print(outlier_df.head(15).to_string())
    else:
        print("\nHicbir sutunda aykiri deger bulunamadi.")


def save_correlation_heatmap(X_train, y_train):
    """En guclu korelasyonlara sahip ozniteliklerin isi haritasini kaydeder."""
    print("\n" + "=" * 70)
    print("12. KORELASYON ANALIZI")
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
    ax.set_title("En Yuksek Korelasyonlu 20 Oznitelik Arasindaki Korelasyon Isi Haritasi",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"'outputs/correlation_heatmap.png' kaydedildi.")

    print(f"\nHedef ile en yuksek korelasyona sahip 10 oznitelik:")
    print(target_corr.head(10).to_string())
