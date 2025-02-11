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
