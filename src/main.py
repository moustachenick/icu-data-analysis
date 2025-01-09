# main.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from icu_data_parser.DataParser import DataParser
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_icp_predictions.ICPPredictor import ICPPredictor
from helper.DataFramePrinter import DataFramePrinter


def compare_splits_and_random_states(cleaned_df_lagged, split_sizes, random_states, cv_folds=10):
    """
    Compare model performance across different train/test split sizes and random states.
    Perform cross-validation for each combination.

    Args:
        cleaned_df_lagged (DataFrame): Cleaned and lagged DataFrame.
        split_sizes (list): List of train/test split sizes (fractions for training data).
        random_states (list): List of random_state values to try.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        combined_results_df: DataFrame summarizing performance across splits and random states.
    """
    if not split_sizes:
        raise ValueError("No split sizes provided. Please pass a list of split sizes.")
    if not random_states:
        raise ValueError("No random states provided. Please pass a list of random states.")

    combined_metrics = []

    for split_size in split_sizes:
        for random_state in random_states:
            print(f"\nEvaluating for train size: {split_size:.2f}, test size: {1 - split_size:.2f}, random_state: {random_state}")

            # Split the data
            X = cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"])
            y = cleaned_df_lagged["icp_next"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_size, random_state=random_state)

            # Initialize the ICPPredictor class
            predictor = ICPPredictor()

            # Train models
            predictor.train_models(X_train, y_train)

            # Evaluate models
            results_df = predictor.evaluate_models(X_test, y_test, split_size=split_size, random_state=random_state)

            # Perform cross-validation
            X_scaled = predictor.scaler.fit_transform(X)
            for name, model in predictor.models.items():
                print(f"Cross-validating {name} for train size {split_size:.2f}, random_state {random_state}...")
                scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="neg_mean_squared_error")
                results_df.loc[results_df["Model"] == name, "Mean CV MSE"] = -scores.mean()

            # Add split size and random_state information
            results_df["Train Size"] = split_size
            results_df["Test Size"] = 1 - split_size
            results_df["Random State"] = random_state

            combined_metrics.append(results_df)

    # Combine results for all splits and random states into a single DataFrame
    combined_results_df = pd.concat(combined_metrics, ignore_index=True)

    # Display summary of results
    DataFramePrinter.print_dataframe_tabulated(combined_results_df, "Comparison of Train/Test Splits, Random States, and Cross-Validation")

    return combined_results_df


def main():
    """
    Main function to initialize and run the ICPPrediction pipeline.
    """
    # Path for the raw data directory (icu-data-analysis/data)
    raw_data_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    # Use the DataParser class to process the raw data
    data_parser = DataParser(raw_data_dir_path)
    raw_data_file_path = data_parser.run()

    # Initialize the DataPreProcessor class and run the preprocess pipeline
    print("Running the Data Preprocessing pipeline...\n")
    data_pre_processor = DataPreProcessor(raw_data_file_path)

    cleaned_df_lagged = data_pre_processor.pre_process_dataset()

    # Compare different train/test splits and random states
    print("\nComparing performance across different train/test splits and random states...")
    split_sizes = [0.5, 0.6, 0.7, 0.8]  # Example split sizes
    random_states = [42, 21, 0, 1]  # Example random states
    combined_results_df = compare_splits_and_random_states(cleaned_df_lagged, split_sizes, random_states)

    # Print the configuration with the lowest RMSE
    min_rmse = combined_results_df["RMSE"].min()
    best_config = combined_results_df[combined_results_df["RMSE"] == min_rmse]
    DataFramePrinter.print_dataframe_tabulated(best_config, "Best Configuration with Lowest RMSE")

    print("\nICP Prediction pipeline completed.")


if __name__ == "__main__":
    main()
