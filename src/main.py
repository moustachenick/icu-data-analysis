import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from icu_data_parser.DataParser import DataParser
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_icp_predictions.ICPRegressionPredictor import ICPRegressionPredictor
from icu_data_parser.BinaryDataProcessor import BinaryDataProcessor
from helper.DataFramePrinter import DataFramePrinter


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

    headers = ["Dataset", "Total Rows", "Rows with 1 Null", "Percentage", "Rows with >1 Null",
               "Percentage"]
    table = [
        ["Train"] + train_stats,
        ["Test"] + test_stats
    ]

    print("\nDataset Statistics:\n")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def main(mode):
    """
    Main function to initialize and run the ICPPrediction pipeline.
    Args:
        mode (str): Mode of operation, either "regression" or "classification".
    """
    # Path for the raw data directory (icu-data-analysis/data)
    data_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    # Use the DataParser class to process the raw data
    data_parser = DataParser(data_dir_path)
    raw_data_file_path = data_parser.run()

    print_statistics_about_the_data(raw_data_file_path)

    # Initialize the DataPreProcessor class and run the preprocess pipeline
    print("Running the Data Preprocessing pipeline...\n")
    data_pre_processor = DataPreProcessor(raw_data_file_path)

    cleaned_df_lagged = data_pre_processor.pre_process_dataset()

    # Create train/test split and save the datasets
    print("\nCreating train/test split and saving datasets...")
    train_size = 0.8  # 80/20 split
    random_state = 42
    X = cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"])
    y = cleaned_df_lagged["icp_next"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Save train and test datasets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(data_dir_path + "/train_data.csv", index=False)
    test_data.to_csv(data_dir_path + "/test_data.csv", index=False)

    # Print dataset statistics
    print_dataset_statistics(train_data, test_data)

    if mode == "classification":
        # Convert target variable to binary values
        binary_processor = BinaryDataProcessor()
        binary_processor.create_binary_data(train_data)
        binary_processor.create_binary_data(test_data)

        # Save the modified datasets with a "_classification" suffix
        train_data.to_csv(data_dir_path + "/train_data_classification.csv", index=False)
        test_data.to_csv(data_dir_path + "/test_data_classification.csv", index=False)

        if input("Do you want to continue with the Classification pipeline? (y/n): ").lower() == 'y':
            # TODO: Implement classification pipeline
            print("Classification mode is not implemented yet.")
    else:
        if input("Do you want to continue with the Regression pipeline? (y/n): ").lower() == 'y':
            # Initialize the ICPRegressionPredictor and run the pipeline
            predictor = ICPRegressionPredictor()
            results = predictor.run_pipeline(X_train, X_test, y_train, y_test)

            # Print the evaluation results
            DataFramePrinter.print_dataframe_tabulated(results["evaluation_results"], "Regression Predictions Results")
            DataFramePrinter.print_dataframe_tabulated(results["cross_validation_results"], "Cross-Validation Results")

    print("\nICP Prediction pipeline completed. ✅")


def print_statistics_about_the_data(raw_data_file_path):
    """
    Print statistics about the raw data.
    Args:
        raw_data_file_path (str): Path to the raw data file.
    """
    raw_data = pd.read_csv(raw_data_file_path)
    print("\nStatistics about the raw data:\n")
    print(raw_data.info())
    print("\nDescriptive statistics of the raw data:\n")
    print(raw_data.describe())
    print("\nFirst 5 rows of the raw data:\n")
    print(raw_data.head())

    df = pd.read_csv(raw_data_file_path)

    # Total number of rows
    total_rows = len(df)
    print(f"Total number of rows: {total_rows}")

    # Number of unique patients
    unique_patients = df['patient_id'].nunique()
    print(f"Number of unique patients: {unique_patients}")

    # Number of rows per patient
    rows_per_patient = df['patient_id'].value_counts()
    print("\nNumber of rows per patient:")
    print(rows_per_patient)

    print(f"Number of patients with 2 rows or less: {sum(rows_per_patient <= 2)}")

    # Convert the timestamp column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the DataFrame by patient_id and timestamp
    df = df.sort_values(by=['patient_id', 'timestamp'])

    # Calculate the time difference between consecutive timestamps for each patient
    df['time_diff'] = df.groupby('patient_id')['timestamp'].diff()

    # Define the 30-minute interval
    thirty_minutes = pd.Timedelta(minutes=30)

    # Check for rows that do not follow the 30-minute interval within each patient group
    non_30_minute_intervals = df['time_diff'] != thirty_minutes

    # Calculate the percentage of rows that do not follow the 30-minute interval
    percentage_non_30_minute_intervals = non_30_minute_intervals.mean() * 100

    # Print the statistics
    print(
        f"Percentage of the dataset that does not have 30-minute interval timestamps: {percentage_non_30_minute_intervals:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ICP Prediction pipeline.")
    parser.add_argument("--mode", type=str, choices=["regression", "classification"],
                        help="Mode of operation: 'regression' or 'classification'")
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

    main(args.mode)
