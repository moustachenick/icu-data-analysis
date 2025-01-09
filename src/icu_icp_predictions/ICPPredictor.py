import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from helper.DataFramePrinter import DataFramePrinter


class ICPPredictor:
    def __init__(self):
        """
        Initialize the ICPPrediction class.
        """
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(alpha=0.1),
        }
        self.scaler = None  # Initialize the scaler

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