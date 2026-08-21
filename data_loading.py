"""
Data Loading and Preparation
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_prepare_data():
    """Loads the Breast Cancer Wisconsin dataset and maps target encoding
    to malignant=1, benign=0.
    """
    print("=" * 70)
    print("1-2. DATASET LOADING AND PREPARATION")
    print("=" * 70)

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    original_dist = df["target"].value_counts().to_dict()
    print(f"\nOriginal target encoding: 0=malignant, 1=benign")
    print(f"  0 (malignant): {original_dist[0]}")
    print(f"  1 (benign):    {original_dist[1]}")

    df["target"] = 1 - df["target"]

    flipped_dist = df["target"].value_counts().to_dict()
    print(f"\nRemapped target encoding: 1=malignant, 0=benign")
    print(f"  1 (malignant): {flipped_dist[1]}")
    print(f"  0 (benign):    {flipped_dist[0]}")

    print(f"\nProblem type: Binary classification")
    print(f"Target variable: target (1=malignant, 0=benign)")

    return df
