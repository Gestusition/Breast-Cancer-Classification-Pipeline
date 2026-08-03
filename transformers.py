"""
Özel Sklearn Transformer'lar
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RadiusCategoryTransformer(BaseEstimator, TransformerMixin):
    """mean radius ozniteligini quantile-based kategorilere donusturur.

    fit(): X_train uzerindeki mean radius degerlerinden qcut bin sinirlarini ogrenir.
    transform(): Ogrenilen sinirlari kullanarak radius_category sutununu ekler.
    """

    def __init__(self, source_column="mean radius", target_column="radius_category", n_bins=3):
        self.source_column = source_column
        self.target_column = target_column
        self.n_bins = n_bins

    def fit(self, X, y=None):
        values = self._validated_source_values(X)
        if self.n_bins != 3:
            raise ValueError(
                "RadiusCategoryTransformer yalnizca n_bins=3 degerini "
                "destekler; kategoriler: small, medium, large."
            )

        usable_values = values[~np.isnan(values)]
        try:
            _, bin_edges = pd.qcut(
                usable_values,
                q=self.n_bins,
                retbins=True,
                duplicates="drop",
            )
            if len(bin_edges) != self.n_bins + 1:
                _, bin_edges = pd.cut(
                    usable_values,
                    bins=self.n_bins,
                    retbins=True,
                )
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"'{self.source_column}' sutunu icin kategori sinirlari "
                "olusturulamadi."
            ) from exc

        bin_edges = np.asarray(bin_edges, dtype=float)
        if len(bin_edges) != self.n_bins + 1 or np.any(np.diff(bin_edges) <= 0):
            raise ValueError(
                f"'{self.source_column}' sutunu icin {self.n_bins} gecerli "
                "kategori araligi olusturulamadi."
            )
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.bin_edges_ = bin_edges
        self.labels_ = ["small", "medium", "large"]
        self._input_features_ = X.columns.tolist()
        return self

    def transform(self, X):
        check_is_fitted(
            self,
            attributes=["bin_edges_", "labels_", "_input_features_"],
        )
        self._validated_source_values(X)
        X_copy = X.copy()
        X_copy[self.target_column] = pd.cut(
            X_copy[self.source_column],
            bins=self.bin_edges_,
            labels=self.labels_,
        )
        return X_copy

    def _validated_source_values(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RadiusCategoryTransformer girdisi bir pandas DataFrame olmalidir.")
        if self.source_column not in X.columns:
            raise ValueError(f"Kaynak sutun bulunamadi: '{self.source_column}'.")

        source = X[self.source_column]
        if not pd.api.types.is_numeric_dtype(source.dtype):
            raise TypeError(f"'{self.source_column}' sutunu sayisal olmalidir.")

        values = source.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"'{self.source_column}' sutunu sonsuz deger iceremez.")
        if np.isnan(values).all():
            raise ValueError(
                f"'{self.source_column}' sutunu en az bir kullanilabilir sayisal deger icermelidir."
            )
        return values

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
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
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
