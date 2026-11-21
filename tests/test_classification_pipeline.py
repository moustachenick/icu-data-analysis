import contextlib
import io
import os
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from classification.classification_pipeline import (
    CLINICAL_RULE_NAME, ClassificationPipeline, threshold_for_recall,
)

VARIABLES = ["icp", "cpp", "heart_rate"]


def _make_split(patient_ids, rng):
    rows = []
    for pid in patient_ids:
        level = rng.uniform(5, 30)  # some patients run high, most low
        for t in range(30):
            row = {"patient_id": pid, "timestamp": f"2020-01-01 {t // 2:02d}:{30 * (t % 2):02d}:00"}
            for var in VARIABLES:
                for lag in range(1, 6):
                    row[f"{var}_lag_{lag}"] = level + rng.normal(0, 4)
            row["icp_binary"] = int(level + rng.normal(0, 4) >= 22)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.drop(columns=["icp_binary"]), df["icp_binary"]


class TestThresholdForRecall(unittest.TestCase):

    def test_threshold_reaches_at_least_the_target_recall(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        score = np.array([0.9, 0.8, 0.4, 0.1, 0.7, 0.2])

        threshold = threshold_for_recall(y, score, 0.5)

        self.assertEqual(threshold, 0.8)  # the 2 highest-scoring positives give recall 0.5
        self.assertEqual(int(np.sum((score >= threshold) & (y == 1))), 2)

    def test_full_recall_uses_the_lowest_positive_score(self):
        self.assertEqual(threshold_for_recall([1, 0, 1], [0.3, 0.1, 0.6], 1.0), 0.3)


class TestClassificationPipeline(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.splits = (_make_split(range(0, 30), rng), _make_split(range(30, 40), rng), _make_split(range(40, 50), rng))
        self.pipeline = ClassificationPipeline(data_dir_path="unused")
        # A small grid keeps the test fast; the tuning logic is the same.
        self.pipeline.xgb_param_grid = {"max_depth": [2, 3], "n_estimators": [10], "scale_pos_weight": [1, 6]}

    def _run(self, cwd):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.splits
        old_cwd = os.getcwd()
        os.chdir(cwd)  # the feature-importance plots are written to the working directory
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                results = self.pipeline.run_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)
                self.pipeline.run_cross_validation_pipeline(X_train, y_train, n_splits=3)
        finally:
            os.chdir(old_cwd)
        return results, out.getvalue()

    def test_tuned_params_come_from_the_grid_and_reach_every_report(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            results, output = self._run(tmp)

        for params in self.pipeline.tuned_params.values():
            for key, values in self.pipeline.xgb_param_grid.items():
                self.assertIn(params[key], values)

        names = [r["name"] for r in results]
        self.assertIn(CLINICAL_RULE_NAME, names)
        self.assertIn("XGBoostClassificationPredictor_tuned", names)
        self.assertIn("XGBoostClassificationPredictor_lagonly_tuned", names)
        for res in results:
            self.assertEqual(len(res["y_score"]), len(res["y_true"]))
        self.assertIn("Matched Sensitivity (TEST)", output)
        self.assertIn("AUROC", output)
        self.assertIn("10-Fold Cross-Validation Results", output)

    def test_cross_validation_needs_tuned_params(self):
        (X_train, y_train), _, _ = self.splits
        with self.assertRaises(ValueError):
            ClassificationPipeline("unused").run_cross_validation_pipeline(X_train, y_train, n_splits=3)


if __name__ == "__main__":
    unittest.main()
