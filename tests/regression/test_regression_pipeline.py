import os
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from regression.regression_pipeline import RegressionPipeline

VARIABLES = ["icp", "cpp", "peep", "ph", "spo2", "temperature"]
HOURS = 5


def _make_split(patient_ids, rng):
    rows = []
    for pid in patient_ids:
        for t in range(20):
            row = {"patient_id": pid, "timestamp": f"2020-01-01 {t:02d}:00:00"}
            for var in VARIABLES:
                for lag in range(1, HOURS + 1):
                    row[f"{var}_lag_{lag}"] = rng.normal(10, 3)
            row["pathology_tbi"] = pid % 2
            row["pathology_other"] = 1 - pid % 2
            rows.append(row)
    X = pd.DataFrame(rows)
    y = X["icp_lag_1"] * 0.8 + rng.normal(0, 1, len(X))
    return X, y


class TestBuildFeatureConfigs(unittest.TestCase):

    def setUp(self):
        X, _ = _make_split([1], np.random.default_rng(0))
        self.columns = list(X.columns)
        self.configs = RegressionPipeline.build_feature_configs(self.columns)

    def test_drop_peep_ph_spo2_removes_all_their_lags(self):
        config = self.configs["Drop PEEP, PH, SPO2"]
        for var in ["peep", "ph", "spo2"]:
            self.assertFalse([c for c in config if c.startswith(f"{var}_lag_")], var)
        self.assertIn("icp_lag_1", config)
        self.assertNotEqual(set(config), set(self.configs["Full Dataset"]))

    def test_lags_1_3_5_keeps_only_those_lags(self):
        lags = {c.rsplit("_lag_", 1)[1] for c in self.configs["Lags 1-3-5"] if "_lag_" in c}
        self.assertEqual(lags, {"1", "3", "5"})

    def test_pathology_only_in_lags_plus_pathology(self):
        for name, config in self.configs.items():
            has_pathology = any(c.startswith("pathology_") for c in config)
            self.assertEqual(has_pathology, name == "Lags + Pathology", name)

    def test_every_config_keeps_id_columns(self):
        for name, config in self.configs.items():
            self.assertIn("patient_id", config, name)
            self.assertIn("timestamp", config, name)


class TestRegressionPipeline(unittest.TestCase):

    def test_linear_models_scale_inside_a_pipeline(self):
        pipeline = RegressionPipeline()
        for name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            model = pipeline.models[name]["model"]
            self.assertIsInstance(model, Pipeline, name)
            self.assertIsInstance(model[0], StandardScaler, name)

    def test_run_pipeline_reports_r2_and_all_configs(self):
        rng = np.random.default_rng(42)
        X_train, y_train = _make_split(range(0, 20), rng)
        X_val, y_val = _make_split(range(20, 25), rng)
        X_test, y_test = _make_split(range(25, 30), rng)

        results = RegressionPipeline().run_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)

        evaluation = results["evaluation_results"]
        self.assertIn("R²", evaluation.columns)
        self.assertNotIn("Accuracy (%)", evaluation.columns)
        self.assertTrue((evaluation["R²"] <= 1).all())
        self.assertIn("Lags + Pathology", results["feature_config_results"].columns)
        self.assertEqual(len(results["cross_validation_results"]), len(RegressionPipeline().models))


if __name__ == "__main__":
    unittest.main()
