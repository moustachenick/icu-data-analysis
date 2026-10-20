import os
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from classification.classification_pipeline import ClassificationPipeline


class TestFeatureImportancePlot(unittest.TestCase):
    """Each XGBoost variant gets its own plot file; before, all three overwrote one file."""

    def test_plot_is_saved_to_the_given_filename(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = (X[:, 0] > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=5).fit(X, y)
        pipeline = ClassificationPipeline(data_dir_path="unused")

        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.png")
            second = os.path.join(tmp, "second.png")
            pipeline.plot_xgboost_feature_importances(model, filename=first)
            pipeline.plot_xgboost_feature_importances(model, title="Other", filename=second)

            self.assertTrue(os.path.exists(first))
            self.assertTrue(os.path.exists(second))


if __name__ == "__main__":
    unittest.main()
