import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from helper.DataFramePrinter import DataFramePrinter
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_parser.TimeSeriesProcessor import TimeSeriesProcessor
from icu_icp_predictions.ICPPredictionsPlotter import ICPPredictionsPlotter


class ICPPredictor:
    def __init__(self, raw_data_file_path):
        """
        Initialize the ICPPrediction class.

        Args:
            raw_data_file_path (str): Path to the raw data file.
        """
        self.raw_data_file_path = raw_data_file_path
        self.lagged_data_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_df_lagged.csv")
        )
        self.cleaned_df_lagged = None
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(alpha=0.1),
        }
        self.scaler = None  # Initialize the scaler

    def preprocess_data(self):
        """
        Preprocess the dataset and create lagged features.
        """
        if os.path.isfile(self.lagged_data_file_path):
            print(f"{self.lagged_data_file_path} already exists. Loading the data from the file.")
            self.cleaned_df_lagged = pd.read_csv(self.lagged_data_file_path)
        else:
            print(f"{self.lagged_data_file_path} does not exist. Processing the data.")
            data_processor = DataPreProcessor(self.raw_data_file_path)
            self.cleaned_df_lagged = data_processor.pre_process_dataset()

            # Create lagged features
            lags = 5
            columns_to_lag = [
                "icp", "temperature", "mean_blood_pressure", "cpp", "glucose",
                "haemoglobin", "heart_rate", "paco2", "pao2", "peep", "ph", "spo2"
            ]
            time_series_processor = TimeSeriesProcessor()
            self.cleaned_df_lagged = time_series_processor.process_data(
                self.cleaned_df_lagged, lags=lags, columns_to_lag=columns_to_lag
            )
            self.cleaned_df_lagged.to_csv(self.lagged_data_file_path, index=False)

        print("Preprocessing complete.")
        print(f"Number of rows in the cleaned dataset: {self.cleaned_df_lagged.shape[0]}")
        print(f"Number of rows with NaN values: {self.cleaned_df_lagged.isnull().sum().sum()}")

    def prepare_data(self, test_size=0.3):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Fraction of data to use for testing.

        Returns:
            X_train, X_test, y_train, y_test: Split data for training and testing.
        """
        X = self.cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"])
        y = self.cleaned_df_lagged["icp_next"]

        train_size = int((1 - test_size) * len(X))
        print(
            "\nTotal size: ",
            X.shape[0],
            "\tTrain size:",
            train_size,
            "\tTest size:",
            X.shape[0] - train_size,
            "\tPercentage of the original dataset:",
            round((train_size / X.shape[0]) * 100),
            "%",
        )

        return train_test_split(X, y, train_size=train_size, random_state=42)

    def train_models(self, X_train, y_train):
        """
        Train all models on the training data.
        This method fits the scaler on the training data and then trains all models.

        Args:
            X_train: Training features.
            y_train: Training target variable.
        """
        # Fit the scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            print(f"{name} model trained successfully.")

    def evaluate_models(self, X_test, y_test, split_size=None, random_state=None):
        """
        Evaluate all models on the testing data.
        This method requires that the models have been trained first.

        Args:
            X_test: Testing features.
            y_test: Testing target variable.

        Returns:
            results_df: DataFrame containing evaluation metrics for all models.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Ensure you call train_models() before evaluate_models().")

        # Scale the test data using the fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        metrics = {"Model": [], "MSE": [], "MAE": [], "RMSE": [], "Accuracy (%)": []}

        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")

            predictions = model.predict(X_test_scaled)
            accuracy = model.score(X_test_scaled, y_test) * 100

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)

            metrics["Model"].append(name)
            metrics["MSE"].append(mse)
            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)
            metrics["Accuracy (%)"].append(accuracy)

            print(f"{name}: MSE = {mse:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, Accuracy = {accuracy:.2f}%")

        results_df = pd.DataFrame(metrics)
        if split_size is None and random_state is None:
            print("\nModels evaluation completed.")
        else:
            print(f"\nModels evaluation completed for train size {split_size:.2f}, random_state {random_state}.")
        return results_df


    def compute_feature_importance(self, X_train, y_train):
        """
        Compute and display feature importance for Linear Regression and Lasso models.

        Args:
            X_train: Training features.
            y_train: Training target variable.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Linear Regression Coefficients
        linear_model = self.models["Linear Regression"]
        linear_model.fit(X_train_scaled, y_train)
        linear_coefficients = pd.Series(linear_model.coef_, index=X_train.columns)
        print("\nLinear Regression Coefficients, ordered by importance:")
        print(linear_coefficients.abs().sort_values(ascending=False))

        # Lasso Regression Coefficients
        lasso_model = self.models["Lasso Regression"]
        lasso_model.fit(X_train_scaled, y_train)
        lasso_coefficients = pd.Series(lasso_model.coef_, index=X_train.columns)
        print("\nLasso Regression Coefficients, ordered by importance:")
        print(lasso_coefficients.abs().sort_values(ascending=False))

        # Identify important features from Lasso
        important_features = lasso_coefficients.abs().sort_values(ascending=False).index.tolist()
        print("\nImportant features selected by Lasso:")
        print(important_features)

    def evaluate_with_feature_configs(self, X, y, train_size=0.7):
        """
        Evaluate models with different feature configurations.

        Args:
            X: Full feature set.
            y: Target variable.
            train_size (float): Fraction of data to use for training.

        Returns:
            rmse_results_df: DataFrame of RMSE values for all models and configurations.
        """
        feature_configs = {
            "Full Dataset": X,
            "Lags 1-3-5": X[
                [col for col in X.columns if '_lag_' in col and any(x in col for x in ['_lag_1', '_lag_3', '_lag_5'])]],
            "Drop PEEP, PH, SPO2": X.drop(columns=["peep", "ph", "spo2"]),
            "Extensive Drop": X.drop(columns=[
                "spo2", "spo2_lag_1", "spo2_lag_2", "spo2_lag_3", "spo2_lag_4", "spo2_lag_5",
                "heart_rate", "heart_rate_lag_1", "heart_rate_lag_2", "heart_rate_lag_3", "heart_rate_lag_4",
                "heart_rate_lag_5",
                "paco2", "paco2_lag_1", "paco2_lag_2", "paco2_lag_3", "paco2_lag_4", "paco2_lag_5",
                "pao2", "pao2_lag_1", "pao2_lag_2", "pao2_lag_3", "pao2_lag_4", "pao2_lag_5",
                "temperature", "temperature_lag_1", "temperature_lag_4", "temperature_lag_5",
                "ph", "ph_lag_1", "ph_lag_2", "ph_lag_3", "ph_lag_4", "ph_lag_5",
                "peep", "peep_lag_1", "peep_lag_2", "peep_lag_3", "peep_lag_4", "peep_lag_5",
                "glucose", "glucose_lag_1", "glucose_lag_2", "glucose_lag_3", "glucose_lag_4", "glucose_lag_5",
                "haemoglobin", "haemoglobin_lag_1", "haemoglobin_lag_2", "haemoglobin_lag_3", "haemoglobin_lag_4",
                "haemoglobin_lag_5",
                "cpp", "cpp_lag_1", "cpp_lag_2", "cpp_lag_3", "cpp_lag_4", "cpp_lag_5",
                "mean_blood_pressure", "mean_blood_pressure_lag_1", "mean_blood_pressure_lag_2",
                "mean_blood_pressure_lag_3", "mean_blood_pressure_lag_5"]),
        }

        rmse_metrics = {"Model": list(self.models.keys())}

        for config_name, X_config in feature_configs.items():
            print(f"\nEvaluating for configuration: {config_name}")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_config, y, train_size=train_size, random_state=42)

            # Scale the features
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Collect RMSE for all models
            rmse_values = []
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                rmse_values.append(rmse)

            rmse_metrics[config_name] = rmse_values

        # Convert results to DataFrame
        rmse_results_df = pd.DataFrame(rmse_metrics)
        DataFramePrinter.print_dataframe_tabulated(rmse_results_df, "Feature Configuration RMSE Results")
        return rmse_results_df

    def perform_cross_validation(self, X, y, cv_folds=10):
        """
        Perform cross-validation for all models.

        Args:
            X: Features.
            y: Target variable.
            cv_folds (int): Number of cross-validation folds.

        Returns:
            cv_results_df: DataFrame containing cross-validation MSE for all models.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        cv_results = {"Model": [], "Mean CV MSE": []}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="neg_mean_squared_error")
            mean_mse = -scores.mean()
            cv_results["Model"].append(name)
            cv_results["Mean CV MSE"].append(mean_mse)

        cv_results_df = pd.DataFrame(cv_results)
        print("Cross-validation completed.")
        return cv_results_df

    def compare_splits_and_random_states(self, split_sizes, random_states, cv_folds=10):
        """
        Compare model performance across different train/test split sizes and random states.
        Perform cross-validation for each combination.

        Args:
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
                X_train, X_test, y_train, y_test = train_test_split(
                    self.cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"]),
                    self.cleaned_df_lagged["icp_next"],
                    train_size=split_size,
                    random_state=random_state
                )

                # Train models
                self.train_models(X_train, y_train)

                # Evaluate models
                results_df = self.evaluate_models(X_test, y_test, split_size=split_size, random_state=random_state)

                # Perform cross-validation
                X_scaled = self.scaler.fit_transform(
                    self.cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"])
                )
                y = self.cleaned_df_lagged["icp_next"]

                for name, model in self.models.items():
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

    def run(self):
        """
        Run the entire ICP Prediction pipeline, including comparisons across splits and random states.
        """
        self.preprocess_data()

        # Default train-test split
        X_train, X_test, y_train, y_test = self.prepare_data()

        self.train_models(X_train, y_train)

        results_df = self.evaluate_models(X_test, y_test)
        DataFramePrinter.print_dataframe_tabulated(results_df, "Model Evaluation Results with Accuracy")

        # Prepare predictions for plotting
        predictions_dict = {}
        X_test_scaled = self.scaler.transform(X_test)
        for name, model in self.models.items():
            predictions_dict[name] = model.predict(X_test_scaled)

        plotter = ICPPredictionsPlotter()

        # Plot model results
        plotter.plot_model_results(y_test, predictions_dict)

        # Plot residuals
        plotter.plot_residuals(y_test, predictions_dict)

        # Plot prediction error
        plotter.plot_prediction_error(y_test, predictions_dict)

        # Plot performance metrics
        plotter.plot_performance_metrics(results_df)

        # Feature Importance
        print("\nComputing feature importance...")
        self.compute_feature_importance(X_train, y_train)

        # Cross-validation
        print("\nPerforming cross-validation...")
        X = self.cleaned_df_lagged.drop(columns=["icp_next", "timestamp", "patient_id", "date_of_birth"])
        y = self.cleaned_df_lagged["icp_next"]
        cv_results_df = self.perform_cross_validation(X, y)
        DataFramePrinter.print_dataframe_tabulated(cv_results_df, "Cross-Validation Results")

        # Plot cross-validation results
        print("\nPlotting cross-validation results...")
        plotter.plot_cross_validation_results(cv_results_df)

        # Feature Configurations
        print("\nEvaluating with different feature configurations...")
        self.evaluate_with_feature_configs(X, y)

        # Compare different train/test splits and random states
        print("\nComparing performance across different train/test splits and random states...")
        split_sizes = [0.5, 0.6, 0.7, 0.8]  # Example split sizes
        random_states = [42, 21, 0, 1]  # Example random states
        combined_results_df = self.compare_splits_and_random_states(split_sizes, random_states)

        # Print the configuration with the lowest RMSE
        min_rmse = combined_results_df["RMSE"].min()
        best_config = combined_results_df[combined_results_df["RMSE"] == min_rmse]
        DataFramePrinter.print_dataframe_tabulated(best_config, "Best Configuration with Lowest RMSE")

        print("\nICP Prediction pipeline completed.")
