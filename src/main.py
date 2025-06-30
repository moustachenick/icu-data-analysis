import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from classification.classification_pipeline import ClassificationPipeline
from data_parser.data_parser import DataParser
from data_parser.data_pre_processor import DataPreProcessor
from helper.data_frame_printer import DataFramePrinter
from tabulate import tabulate
from regression.regression_pipeline import RegressionPipeline


def main(mode, hours):
    """
    Main function to initialize and run the ICPPrediction pipeline.
    Args:
        mode (str): Mode of operation, either "regression" or "classification".
        hours (int): Number of hours to use for creating lag features.
    """
    data_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    data_parser = DataParser(data_dir_path)
    raw_data_file_path = data_parser.run()

    print_statistics_about_the_data(raw_data_file_path)

    print("Running the Data Preprocessing pipeline...\n")
    data_pre_processor = DataPreProcessor(raw_data_file_path)
    cleaned_df_lagged = data_pre_processor.pre_process_dataset(hours, mode)

    train_data_path = os.path.join(data_dir_path, f"train_data_{mode}.csv")
    test_data_path = os.path.join(data_dir_path, f"test_data_{mode}.csv")

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("\nCreating train/test split and saving datasets...")
        # Patient-level train/test split
        unique_patients = cleaned_df_lagged["patient_id"].unique()
        train_patients, test_patients = train_test_split(
            unique_patients, train_size=0.8, random_state=42
        )

        # Filter rows by patient group
        train_data = cleaned_df_lagged[cleaned_df_lagged["patient_id"].isin(train_patients)].copy()
        test_data = cleaned_df_lagged[cleaned_df_lagged["patient_id"].isin(test_patients)].copy()

        # Final X/y splits
        X_train = train_data.drop(columns=["icp"])
        y_train = train_data["icp"]
        X_test = test_data.drop(columns=["icp"])
        y_test = test_data["icp"]

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(train_data_path, index=False)
        test_data.to_csv(test_data_path, index=False)

        print_dataset_statistics(train_data, test_data)
    else:
        print("\nTrain and test datasets already exist. Skipping dataset creation.")
        X_train = pd.read_csv(train_data_path).drop(columns=["icp"])
        y_train = pd.read_csv(train_data_path)["icp"]
        X_test = pd.read_csv(test_data_path).drop(columns=["icp"])
        y_test = pd.read_csv(test_data_path)["icp"]

    if mode == "classification":
        pipeline = ClassificationPipeline(data_dir_path)
        pipeline.run_pipeline(X_train, X_test, y_train, y_test)
        if input("Run also the 10-fold cross-validation? (y/n): ").lower() == 'y':
            # Reconstruct full X and y from train/test parts,
            # because the cross-validation pipeline expects the full dataset
            X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
            pipeline.run_cross_validation_pipeline(X, y)
    else:
        run_regression_pipeline(X_train, X_test, y_train, y_test)

    print("\nICP Prediction pipeline completed. ✅")


def print_dataset_statistics(train_data, test_data):
    def calculate_statistics(df):
        total_rows = len(df)
        one_null = (df.isnull().sum(axis=1) == 1).sum()
        more_than_one_null = (df.isnull().sum(axis=1) > 1).sum()
        one_null_percentage = (one_null / total_rows) * 100
        more_than_one_null_percentage = (more_than_one_null / total_rows) * 100
        return [total_rows, one_null, one_null_percentage, more_than_one_null, more_than_one_null_percentage]

    train_stats = calculate_statistics(train_data)
    test_stats = calculate_statistics(test_data)

    headers = ["Dataset", "Total Rows", "Rows with 1 Null", "Percentage", "Rows with >1 Null", "Percentage"]
    table = [
        ["Train"] + train_stats,
        ["Test"] + test_stats
    ]

    print("\nDataset Statistics:\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def run_regression_pipeline(X_train, X_test, y_train, y_test):
    if input("Do you want to continue with the Regression pipeline? (y/n): ").lower() == 'y':
        pipeline = RegressionPipeline()
        results = pipeline.run_pipeline(X_train, X_test, y_train, y_test)

        DataFramePrinter.print_dataframe_tabulated(results["evaluation_results"], "Regression Predictions Results")
        DataFramePrinter.print_dataframe_tabulated(results["cross_validation_results"], "Cross-Validation Results")


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
                        help="Mode of operation: 'regression' or 'classification'")
    parser.add_argument("--hours", type=int, default=5, help="Number of hours to use for creating lag features")
    args = parser.parse_args()

    if not args.mode:
        print("Which mode would you like to run?")
        print("1. Regression")
        print("2. Classification")
        choice = input("Enter the number of your choice: ").strip()
        if choice == "1":
            args.mode = "regression"
        elif choice == "2":
            args.mode = "classification"
        else:
            print("Invalid choice. Defaulting to regression.")
            args.mode = "regression"

    main(args.mode, args.hours)
