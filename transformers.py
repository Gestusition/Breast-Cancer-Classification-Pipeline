"""
Özel Sklearn Transformer'lar
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RadiusCategoryTransformer(BaseEstimator, TransformerMixin):
    """mean radius ozniteligini quantile-based kategorilere donusturur.

    fit(): X_train uzerindeki mean radius degerlerinden qcut bin sinirlarini ogrenir.
    transform(): Ogrenilen sinirlari kullanarak radius_category sutununu ekler.
    """

    def __init__(self, source_column="mean radius", target_column="radius_category", n_bins=3):
        self.source_column = source_column
        self.target_column = target_column
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.labels_ = ["small", "medium", "large"]

    def fit(self, X, y=None):
        values = np.asarray(X[self.source_column], dtype=float)
        try:
            _, bin_edges = pd.qcut(values, q=self.n_bins, retbins=True, duplicates="drop")
        except (ValueError, IndexError):
            _, bin_edges = pd.cut(values, bins=self.n_bins, retbins=True)
        bin_edges = bin_edges.astype(float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.bin_edges_ = bin_edges
        actual_bins = len(bin_edges) - 1
        if actual_bins < self.n_bins:
            self.labels_ = self.labels_[:actual_bins]
        self._input_features_ = X.columns.tolist() if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.target_column] = pd.cut(
            X_copy[self.source_column],
            bins=self.bin_edges_,
            labels=self.labels_,
        )
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "_input_features_", None)
        if input_features is None:
            return np.array([self.target_column], dtype=object)
        return np.array(list(input_features) + [self.target_column], dtype=object)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """IQR tabanli aykiri deger sinirlandirmasi (capping).

    fit(): Train verisinden Q1, Q3 ve IQR hesaplar.
    transform(): Degerleri alt ve ust sinirlara clip eder.
    """

    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            q1 = float(X[col].quantile(0.25))
            q3 = float(X[col].quantile(0.75))
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.lower_bounds_:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col],
                )
        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return input_features
