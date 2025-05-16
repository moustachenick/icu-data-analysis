import os
import pandas as pd
from classification.latest_icp_baseline_predictor import LatestICPBaselinePredictor
from classification.lagged_icp_baseline_predictor import LaggedICPBaselinePredictor
from classification.xgboost_classification_predictor import XGBoostClassificationPredictor
from data_parser.binary_data_processor import BinaryDataProcessor
from helper.data_frame_printer import DataFramePrinter
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm

class ClassificationPipeline:
    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.train_data_classification_path = os.path.join(data_dir_path, "train_data_classification.csv")
        self.test_data_classification_path = os.path.join(data_dir_path, "test_data_classification.csv")

    def prepare_classification_datasets(self, X_train, X_test, y_train, y_test):
        if not os.path.exists(self.train_data_classification_path) or not os.path.exists(
                self.test_data_classification_path):
            print("\nCreating classification datasets...")
            binary_processor = BinaryDataProcessor()
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            train_data = binary_processor.create_binary_data(train_data)
            test_data = binary_processor.create_binary_data(test_data)
            train_data.to_csv(self.train_data_classification_path, index=False)
            test_data.to_csv(self.test_data_classification_path, index=False)
        else:
            print("\nClassification datasets already exist. Skipping dataset creation.")
        train_data = pd.read_csv(self.train_data_classification_path)
        test_data = pd.read_csv(self.test_data_classification_path)
        return train_data, test_data

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        train_data, test_data = self.prepare_classification_datasets(X_train, X_test, y_train, y_test)

        results = []
        daily_mean_predictor = LaggedICPBaselinePredictor()
        daily_mean_predictor.fit(X_train)
        results.append(daily_mean_predictor.run_pipeline(X_test, y_test))

        X_train_bin = train_data.drop(columns=["icp_binary"])
        y_train_bin = train_data["icp_binary"]
        X_test_bin = test_data.drop(columns=["icp_binary"])
        y_test_bin = test_data["icp_binary"]

        baseline_predictor = LatestICPBaselinePredictor()
        results.append(baseline_predictor.run_pipeline(X_test_bin, y_test_bin))

        predictor = XGBoostClassificationPredictor()
        results.append(predictor.run_pipeline(X_train_bin, X_test_bin, y_train_bin, y_test_bin))

        # Prepare a summary table
        summary = []
        for res in results:
            report = res['classification_report']
            summary.append({
                'Predictor': res['name'],
                'Accuracy': f"{report['accuracy']:.3f}",
                'Precision (1)': f"{report['1']['precision']:.3f}",
                'Recall (1)': f"{report['1']['recall']:.3f}",
                'F1-score (1)': f"{report['1']['f1-score']:.3f}"
            })
        summary_df = pd.DataFrame(summary)
        DataFramePrinter.print_dataframe_tabulated(summary_df, "Classification Models Comparison")

    def run_cross_validation_pipeline(self, X, y, n_splits=10):
        print("\nRunning 10-fold cross-validation...\n")

        # Preprocess once for binary classification
        binary_processor = BinaryDataProcessor()
        full_data = pd.concat([X, y], axis=1)
        full_data = binary_processor.create_binary_data(full_data)

        # Create binary labels ONLY for stratification
        y_stratify = (y >= 22).astype(int)

        # Prepare StratifiedKFold
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

        predictors = {
            "LaggedICPBaselinePredictor": LaggedICPBaselinePredictor(),
            "LatestICPBaselinePredictor": LatestICPBaselinePredictor(),
            "XGBoostClassificationPredictor": XGBoostClassificationPredictor()
        }
        fold_results = {name: [] for name in predictors}
        patient_ids = X["patient_id"]
        for fold, (train_idx, test_idx) in enumerate(
                tqdm(sgkf.split(X, y_stratify, groups=patient_ids), total=n_splits, desc="Cross-validation folds"), 1):

            # Get raw icp data
            X_train_raw = X.iloc[train_idx].copy()
            y_train_raw = y.iloc[train_idx].copy()
            X_test_raw = X.iloc[test_idx].copy()
            y_test_raw = y.iloc[test_idx].copy()

            # --- Restore patient/time context ---
            X_train_raw["timestamp"] = X_train_raw["timestamp"]
            X_train_raw["patient_id"] = X_train_raw["patient_id"]
            X_test_raw["timestamp"] = X_test_raw["timestamp"]
            X_test_raw["patient_id"] = X_test_raw["patient_id"]

            # === LaggedICPBaseline ===
            lagged = predictors["LaggedICPBaselinePredictor"]
            lagged.fit(X_test_raw)  # Fit on test fold itself to compute per-day means
            res = lagged.run_pipeline(X_test_raw, y_test_raw)
            fold_results["LaggedICPBaselinePredictor"].append(res["classification_report"])

            # === Binary processing for LatestICP + XGBoost ===
            binary_processor = BinaryDataProcessor()
            train_bin = binary_processor.create_binary_data(pd.concat([X_train_raw, y_train_raw], axis=1))
            test_bin = binary_processor.create_binary_data(pd.concat([X_test_raw, y_test_raw], axis=1))

            X_train_bin = train_bin.drop(columns=["icp_binary"])
            y_train_bin = train_bin["icp_binary"]
            X_test_bin = test_bin.drop(columns=["icp_binary"])
            y_test_bin = test_bin["icp_binary"]

            # === LatestICPBaseline ===
            latest = predictors["LatestICPBaselinePredictor"]
            res = latest.run_pipeline(X_test_bin, y_test_bin)
            fold_results["LatestICPBaselinePredictor"].append(res["classification_report"])

            # === XGBoost ===
            xgb = predictors["XGBoostClassificationPredictor"]
            res = xgb.run_pipeline(X_train_bin, X_test_bin, y_train_bin, y_test_bin)
            fold_results["XGBoostClassificationPredictor"].append(res["classification_report"])

        # === Aggregate summary ===
        summary = []
        for name, reports in fold_results.items():
            f1s = [r['1']['f1-score'] for r in reports]
            precisions = [r['1']['precision'] for r in reports]
            recalls = [r['1']['recall'] for r in reports]
            accuracies = [r['accuracy'] for r in reports]

            summary.append({
                "Predictor": name,
                "Accuracy": f"{np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}",
                "Precision (1)": f"{np.mean(precisions):.3f} ± {np.std(precisions):.3f}",
                "Recall (1)": f"{np.mean(recalls):.3f} ± {np.std(recalls):.3f}",
                "F1-score": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
            })

        summary_df = pd.DataFrame(summary)
        DataFramePrinter.print_dataframe_tabulated(summary_df, "10-Fold Cross-Validation Results (Mean ± Std Dev)")
