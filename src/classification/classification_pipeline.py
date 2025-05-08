import os
import pandas as pd
from classification.latest_icp_baseline_predictor import LatestICPBaselinePredictor
from classification.lagged_icp_baseline_predictor import LaggedICPBaselinePredictor
from classification.xgboost_classification_predictor import XGBoostClassificationPredictor
from data_parser.binary_data_processor import BinaryDataProcessor
from helper.data_frame_printer import DataFramePrinter

class ClassificationPipeline:
    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.train_data_classification_path = os.path.join(data_dir_path, "train_data_classification.csv")
        self.test_data_classification_path = os.path.join(data_dir_path, "test_data_classification.csv")

    def prepare_classification_datasets(self, X_train, X_test, y_train, y_test):
        if not os.path.exists(self.train_data_classification_path) or not os.path.exists(self.test_data_classification_path):
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