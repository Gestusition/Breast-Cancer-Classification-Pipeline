"""
Veri Yükleme ve Hazırlama
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_prepare_data():
    """Breast Cancer Wisconsin veri setini yukler ve hedef kodlamasini
    malignant=1, benign=0 olacak sekilde cevirir.
    """
    print("=" * 70)
    print("1-2. VERI SETI YUKLEME VE HAZIRLAMA")
    print("=" * 70)

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    original_dist = df["target"].value_counts().to_dict()
    print(f"\nOrijinal hedef kodlamasi: 0=malignant, 1=benign")
    print(f"  0 (malignant): {original_dist[0]}")
    print(f"  1 (benign):    {original_dist[1]}")

    df["target"] = 1 - df["target"]

    flipped_dist = df["target"].value_counts().to_dict()
    print(f"\nCevrilmis hedef kodlamasi: 1=malignant, 0=benign")
    print(f"  1 (malignant): {flipped_dist[1]}")
    print(f"  0 (benign):    {flipped_dist[0]}")

    print(f"\nProblemin turu: Ikil siniflandirma (binary classification)")
    print(f"Hedef degisken: target (1=malignant/kotu huylu, 0=benign/iyi huylu)")

    return df
