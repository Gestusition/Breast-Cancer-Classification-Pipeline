"""Minimal smoke test for the end-to-end preprocessing/model path."""

import unittest

# The application configures sklearn to return pandas objects from transformers.
import config  # noqa: F401

from data_loading import load_and_prepare_data
from feature_engineering import engineer_features, inject_missing_values
from preprocessing import create_model_pipelines, split_data


class PipelineSmokeTest(unittest.TestCase):
    def test_logistic_pipeline_fits_and_predicts(self):
        data = engineer_features(inject_missing_values(load_and_prepare_data()))
        self.assertTrue(
            {
                "worst_to_mean_radius_ratio",
                "worst_to_mean_area_ratio",
                "worst_minus_mean_compactness",
            }.issubset(data.columns)
        )
        X_train, _, X_test, y_train, _, y_test = split_data(data)
        pipelines, _, _ = create_model_pipelines()

        model = pipelines["Logistic Regression"]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(set(predictions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
