"""SHAP output normalization and plot generation regression tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import config  # noqa: F401
from data_loading import load_and_prepare_data
from evaluation import (
    _positive_class_shap_values,
    _save_shap_summary_plot,
    explain_model,
)
from feature_engineering import engineer_features, inject_missing_values
from preprocessing import create_model_pipelines, split_data


class _ClassifierWithReversedClasses:
    classes_ = np.array([1, 0])


class ShapValueSelectionTest(unittest.TestCase):
    def test_selects_positive_class_from_list_using_classifier_order(self):
        negative = np.full((4, 3), -1.0)
        positive = np.full((4, 3), 1.0)

        selected = _positive_class_shap_values(
            [positive, negative],
            _ClassifierWithReversedClasses(),
        )

        np.testing.assert_array_equal(selected, positive)

    def test_selects_positive_class_from_three_dimensional_array(self):
        values = np.zeros((4, 3, 2))
        values[:, :, 0] = 7.0
        values[:, :, 1] = -7.0

        selected = _positive_class_shap_values(
            values,
            _ClassifierWithReversedClasses(),
        )

        self.assertEqual(selected.shape, (4, 3))
        np.testing.assert_array_equal(selected, values[:, :, 0])

    def test_keeps_two_dimensional_values_unchanged(self):
        values = np.arange(12).reshape(4, 3)

        selected = _positive_class_shap_values(
            values,
            _ClassifierWithReversedClasses(),
        )

        self.assertIs(selected, values)

    def test_rejects_unexpected_shap_dimensions(self):
        with self.assertRaisesRegex(ValueError, "Unexpected SHAP output dimension"):
            _positive_class_shap_values(
                np.zeros((2, 2, 2, 2)),
                _ClassifierWithReversedClasses(),
            )

    def test_saves_the_figure_created_by_summary_plot(self):
        class FakeShap:
            created_figure = None

            @classmethod
            def summary_plot(cls, *args, **kwargs):
                cls.created_figure, ax = plt.subplots()
                ax.plot([0, 1], [0, 1], linewidth=8)

        stale_figure = plt.figure()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("evaluation.OUTPUTS_DIR", Path(temp_dir)):
                    _save_shap_summary_plot(
                        FakeShap,
                        np.ones((2, 1)),
                        np.ones((2, 1)),
                        ["feature"],
                    )

                output_path = Path(temp_dir) / "shap_summary.png"
                self.assertGreater(output_path.stat().st_size, 1000)
                self.assertGreater(float(mpimg.imread(output_path).std()), 0.01)
                self.assertFalse(plt.fignum_exists(FakeShap.created_figure.number))
        finally:
            plt.close(stale_figure)


class ExplainModelShapIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = engineer_features(inject_missing_values(load_and_prepare_data()))
        cls.X_train, _, cls.X_test, cls.y_train, _, cls.y_test = split_data(data)

    def _assert_model_creates_nonempty_shap_png(self, model_name):
        pipelines, _, _ = create_model_pipelines()
        model = pipelines[model_name]
        model.fit(self.X_train, self.y_train)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch("evaluation.OUTPUTS_DIR", output_dir):
                explain_model(
                    model,
                    self.X_train,
                    self.X_test,
                    self.y_test,
                    model_name,
                )

            output_path = output_dir / "shap_summary.png"
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 1000)
            image = mpimg.imread(output_path)
            self.assertGreater(image.shape[0], 100)
            self.assertGreater(image.shape[1], 100)
            self.assertGreater(float(image.std()), 0.01)

    def test_random_forest_shap_path_creates_nonempty_png(self):
        self._assert_model_creates_nonempty_shap_png("Random Forest")

    def test_logistic_regression_shap_path_still_creates_nonempty_png(self):
        self._assert_model_creates_nonempty_shap_png("Logistic Regression")


if __name__ == "__main__":
    unittest.main()
