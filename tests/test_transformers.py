"""RadiusCategoryTransformer lifecycle regression tests."""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from transformers import RadiusCategoryTransformer


class RadiusCategoryTransformerTest(unittest.TestCase):
    def test_fit_transform_adds_expected_category_column(self):
        data = pd.DataFrame({
            "mean radius": np.arange(1.0, 10.0),
            "other": np.arange(9),
        })

        transformed = RadiusCategoryTransformer().fit_transform(data)

        self.assertIn("radius_category", transformed.columns)
        self.assertEqual(
            transformed["radius_category"].astype(str).tolist(),
            ["small"] * 3 + ["medium"] * 3 + ["large"] * 3,
        )

    def test_refit_after_constant_data_rebuilds_all_learned_state(self):
        transformer = RadiusCategoryTransformer()
        constant = pd.DataFrame({"mean radius": [5.0] * 6})
        variable = pd.DataFrame({"mean radius": np.arange(1.0, 10.0)})

        constant_result = transformer.fit_transform(constant)
        self.assertFalse(constant_result["radius_category"].isna().any())

        transformed = transformer.fit(variable).transform(variable)

        self.assertEqual(transformer.labels_, ["small", "medium", "large"])
        self.assertEqual(len(transformer.bin_edges_), 4)
        self.assertFalse(transformed["radius_category"].isna().any())

    def test_transform_before_fit_raises_not_fitted_error(self):
        data = pd.DataFrame({"mean radius": [1.0, 2.0, 3.0]})

        with self.assertRaises(NotFittedError):
            RadiusCategoryTransformer().transform(data)

    def test_missing_source_column_raises_clear_error(self):
        transformer = RadiusCategoryTransformer()
        valid = pd.DataFrame({"mean radius": [1.0, 2.0, 3.0]})
        missing = pd.DataFrame({"other": [1.0, 2.0, 3.0]})

        with self.assertRaisesRegex(ValueError, "Source column not found"):
            transformer.fit(missing)

        transformer.fit(valid)
        with self.assertRaisesRegex(ValueError, "Source column not found"):
            transformer.transform(missing)

    def test_unsupported_n_bins_raises_clear_error(self):
        data = pd.DataFrame({"mean radius": [1.0, 2.0, 3.0, 4.0]})

        with self.assertRaisesRegex(ValueError, "only supports n_bins=3"):
            RadiusCategoryTransformer(n_bins=4).fit(data)

    def test_requires_dataframe_and_usable_numeric_source(self):
        transformer = RadiusCategoryTransformer()
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            transformer.fit(np.array([[1.0], [2.0]]))
        with self.assertRaisesRegex(TypeError, "must be numeric"):
            transformer.fit(pd.DataFrame({"mean radius": ["1", "2"]}))
        with self.assertRaisesRegex(ValueError, "must contain at least one usable numeric"):
            transformer.fit(pd.DataFrame({"mean radius": [np.nan, np.nan]}))

    def test_sklearn_clone_preserves_only_constructor_parameters(self):
        transformer = RadiusCategoryTransformer(target_column="size")
        transformer.fit(pd.DataFrame({"mean radius": [1.0, 2.0, 3.0]}))

        cloned = clone(transformer)

        self.assertEqual(cloned.get_params()["target_column"], "size")
        self.assertFalse(hasattr(cloned, "bin_edges_"))
        self.assertFalse(hasattr(cloned, "labels_"))


if __name__ == "__main__":
    unittest.main()
