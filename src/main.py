import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from classification.classification_pipeline import ClassificationPipeline
from data_parser.data_parser import DataParser
from data_parser.data_pre_processor import DataPreProcessor
from data_parser.binary_data_processor import BinaryDataProcessor
from helper.config import load_config
from helper.data_frame_printer import DataFramePrinter
from helper.output_logger import build_output_path, tee_output
from tabulate import tabulate
from regression.regression_pipeline import RegressionPipeline


def main(config):
    """
    Main function to initialize and run the ICPPrediction pipeline.
    Args:
        config (AppConfig): Resolved run configuration (see helper.config).
    """
    mode = config.mode
    hours = config.hours

    data_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    data_parser = DataParser(
        data_dir_path,
        apply_instance_filtering=config.apply_instance_filtering,
        filter_by_pathology=config.filter_by_pathology,
    )
    raw_data_file_path = data_parser.run()

    print_statistics_about_the_data(raw_data_file_path)

    print("Running the Data Preprocessing pipeline...\n")
    data_pre_processor = DataPreProcessor(raw_data_file_path, config)

    train_data_path = os.path.join(data_dir_path, f"train_data_{mode}.csv")
    val_data_path = os.path.join(data_dir_path, f"validation_data_{mode}.csv")
    test_data_path = os.path.join(data_dir_path, f"test_data_{mode}.csv")

    if all(os.path.exists(p) for p in (train_data_path, val_data_path, test_data_path)):
        print(f"\nTrain/validation/test datasets for {mode} already exist. Skipping dataset creation.")
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)
    else:
        cleaned_df = data_pre_processor.pre_process_dataset(hours, mode)
        train_data, val_data, test_data = build_split_datasets(cleaned_df, data_pre_processor, config)
        train_data.to_csv(train_data_path, index=False)
        val_data.to_csv(val_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)
        print_dataset_statistics(train_data, val_data, test_data)

    if mode == "classification":
        for name, d in (("train", train_data), ("validation", val_data), ("test", test_data)):
            if "icp_binary" not in d.columns:
                raise ValueError(f"The 'icp_binary' column is missing in the {name} data. "
                                 "Please ensure the binary data is created correctly.")

        X_train, y_train = train_data.drop(columns=["icp_binary"]), train_data["icp_binary"]
        X_val, y_val = val_data.drop(columns=["icp_binary"]), val_data["icp_binary"]
        X_test, y_test = test_data.drop(columns=["icp_binary"]), test_data["icp_binary"]

        pipeline = ClassificationPipeline(data_dir_path)
        # Selection happens on the validation split; the test split is the final report.
        pipeline.run_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)
        if config.run_cross_validation:
            # Secondary robustness check: patient-grouped CV on the TRAINING split only.
            pipeline.run_cross_validation_pipeline(X_train, y_train)
    else:
        X_train, y_train = train_data.drop(columns=["icp"]), train_data["icp"]
        X_val, y_val = val_data.drop(columns=["icp"]), val_data["icp"]
        X_test, y_test = test_data.drop(columns=["icp"]), test_data["icp"]

        run_regression_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\nICP Prediction pipeline completed. ✅")


def build_split_datasets(cleaned_df, data_pre_processor, config):
    """
    Build the three leak-free datasets from the cleaned (pre-impute, pre-lag) frame:
    patient-level train/validation/test split -> causal train-fit imputation -> per-split
    lagged features -> (classification) binary label. Patients are disjoint across splits.
    """
    mode, hours = config.mode, config.hours

    print("\nCreating patient-level train/validation/test split...")
    unique_patients = cleaned_df["patient_id"].unique()
    # Carve out test patients first, then split the remainder into train/validation.
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=config.test_size, random_state=42
    )
    val_relative = config.val_size / (1.0 - config.test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_relative, random_state=42
    )

    def subset(patient_ids):
        return cleaned_df[cleaned_df["patient_id"].isin(patient_ids)].copy()

    train_raw, val_raw, test_raw = subset(train_patients), subset(val_patients), subset(test_patients)
    print(f"Patients — train: {len(train_patients)}, validation: {len(val_patients)}, test: {len(test_patients)}")

    # Causal, train-fit imputation (no future donors, no cross-split leakage).
    if config.impute_missing_values:
        print("\n~~~~ Imputing missing values (causal, train-fit) ~~~~")
        train_raw, val_raw, test_raw = data_pre_processor.causal_knn_impute(train_raw, val_raw, test_raw)
    else:
        print("\nImputation disabled (preprocessing.impute_missing_values = false).")

    # Lagged features per split (lags are within-patient, so per-split == whole-frame).
    print("\n~~~~ Creating lagged features per split ~~~~")
    train_data = data_pre_processor.add_lagged_features(train_raw, hours)
    val_data = data_pre_processor.add_lagged_features(val_raw, hours)
    test_data = data_pre_processor.add_lagged_features(test_raw, hours)

    # Classification label after lagging.
    if mode == "classification":
        binary_processor = BinaryDataProcessor()
        train_data = binary_processor.create_binary_data(train_data)
        val_data = binary_processor.create_binary_data(val_data)
        test_data = binary_processor.create_binary_data(test_data)

    return train_data, val_data, test_data


def print_dataset_statistics(train_data, val_data, test_data):
    def calculate_statistics(df):
        total_rows = len(df)
        one_null = (df.isnull().sum(axis=1) == 1).sum()
        more_than_one_null = (df.isnull().sum(axis=1) > 1).sum()
        one_null_percentage = (one_null / total_rows) * 100
        more_than_one_null_percentage = (more_than_one_null / total_rows) * 100
        return [total_rows, one_null, one_null_percentage, more_than_one_null, more_than_one_null_percentage]

    headers = ["Dataset", "Total Rows", "Rows with 1 Null", "Percentage", "Rows with >1 Null", "Percentage"]
    table = [
        ["Train"] + calculate_statistics(train_data),
        ["Validation"] + calculate_statistics(val_data),
        ["Test"] + calculate_statistics(test_data),
    ]

    print("\nDataset Statistics:\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def run_regression_pipeline(X_train, X_val, X_test, y_train, y_val, y_test):
    pipeline = RegressionPipeline()
    results = pipeline.run_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)

    DataFramePrinter.print_dataframe_tabulated(results["evaluation_results"], "Regression Predictions Results (Test)")
    DataFramePrinter.print_dataframe_tabulated(results["cross_validation_results"], "Cross-Validation Results (Train, grouped)")


def print_statistics_about_the_data(raw_data_file_path):
    df = pd.read_csv(raw_data_file_path, low_memory=False)
    print("\nStatistics about the raw data:\n")
    print(df.info())
    print("\nFirst 5 rows of the raw data:\n")
    print(df.head())
    total_rows = len(df)
    print(f"Total number of rows: {total_rows}")

    unique_patients = df['patient_id'].nunique()
    print(f"Number of unique patients: {unique_patients}")

    rows_per_patient = df['patient_id'].value_counts()

    print(f"Number of patients with 2 rows or less: {sum(rows_per_patient <= 2)}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['patient_id', 'timestamp'])
    df['time_diff'] = df.groupby('patient_id')['timestamp'].diff()

    thirty_minutes = pd.Timedelta(minutes=30)
    non_30_minute_intervals = df['time_diff'] != thirty_minutes
    percentage_non_30_minute_intervals = non_30_minute_intervals.mean() * 100

    print(
        f"Percentage of the dataset that does not have 30-minute interval timestamps: {percentage_non_30_minute_intervals:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ICP Prediction pipeline.")
    parser.add_argument("--mode", type=str, choices=["regression", "classification"],
                        help="Mode of operation: 'regression' or 'classification' (overrides config.toml)")
    parser.add_argument("--hours", type=int, default=None,
                        help="Number of hours to use for creating lag features (overrides config.toml)")
    args = parser.parse_args()

    # All run decisions come from config.toml; --mode/--hours override it when provided.
    config = load_config(cli_mode=args.mode, cli_hours=args.hours)

    output_path = build_output_path(config.mode)
    with tee_output(output_path) as log_path:
        main(config)
        print(f"\nFull run output saved to: {log_path}")