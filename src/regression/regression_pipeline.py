import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from helper.data_frame_printer import DataFramePrinter
from regression.baseline_history_regression import BaselineHistoryRegression
from regression.baseline_mean_regression import BaselineMeanRegression

class RegressionPipeline:
    def __init__(self):
        """
        Initialize the ICPPrediction class.
        """
        self.models = {
            "Linear Regression": {
                "model": LinearRegression(),
                "column_selector": lambda X: X.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore"),
                "use_scaling": True
            },
            "Ridge Regression": {
                "model": Ridge(),
                "column_selector": lambda X: X.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore"),
                "use_scaling": True
            },
            "Lasso Regression": {
                "model": Lasso(alpha=0.1),
                "column_selector": lambda X: X.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore"),
                "use_scaling": True
            },
            "Baseline History Regression": {
                "model": BaselineHistoryRegression(),
                "column_selector": lambda X: X,
                "use_scaling": False
            },
            "Baseline Mean Regression": {
                "model": BaselineMeanRegression(),
                "column_selector": lambda X: X,
                "use_scaling": False
            }
        }
        self.scaler = None  # Initialize the scaler

    def __train_models(self, X_train, y_train):
        """
        Train all models on the training data.
        This method fits the scaler on the training data and then trains all models.

        Args:
            X_train: Training features.
            y_train: Training target variable.
        """
        for name, model_info in self.models.items():
            X_train_selected = model_info["column_selector"](X_train)
            if model_info.get("use_scaling", True):
                self.scaler = StandardScaler()
                X_train_processed = self.scaler.fit_transform(X_train_selected)
            else:
                X_train_processed = X_train_selected
            model_info["model"].fit(X_train_processed, y_train)
            print(f"{name} model trained successfully.")

    def __evaluate_models(self, X_test, y_test):
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

        metrics = {"Model": [], "MSE": [], "MAE": [], "RMSE": [], "Accuracy (%)": []}

        for name, model_info in self.models.items():
            print(f"\nEvaluating {name}...")
            X_test_selected = model_info["column_selector"](X_test)
            if model_info.get("use_scaling", True):
                X_test_processed = self.scaler.transform(X_test_selected)
            else:
                X_test_processed = X_test_selected
            predictions = model_info["model"].predict(X_test_processed)
            accuracy = model_info["model"].score(X_test_processed, y_test) * 100

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)

            metrics["Model"].append(name)
            metrics["MSE"].append(mse)
            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)
            metrics["Accuracy (%)"].append(accuracy)

            print(f"{name}: MSE = {mse:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, Accuracy = {accuracy:.2f}%")

        return pd.DataFrame(metrics)

    def __compute_feature_importance(self, X_train, y_train):
        """
        Compute and display feature importance for Linear Regression and Lasso models.

        Args:
            X_train: Training features.
            y_train: Training target variable.
        """
        # Linear Regression Coefficients
        linear_model_info = self.models["Linear Regression"]
        X_train_linear = linear_model_info["column_selector"](X_train)
        if linear_model_info.get("use_scaling", True):
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_linear)
        else:
            X_train_scaled = X_train_linear
        linear_model_info["model"].fit(X_train_scaled, y_train)
        linear_coefficients = pd.Series(linear_model_info["model"].coef_, index=X_train_linear.columns)
        print("\nLinear Regression Coefficients, ordered by importance:")
        print(linear_coefficients.abs().sort_values(ascending=False))

        # Lasso Regression Coefficients
        lasso_model_info = self.models["Lasso Regression"]
        X_train_lasso = lasso_model_info["column_selector"](X_train)
        if lasso_model_info.get("use_scaling", True):
            self.scaler = StandardScaler()
            X_train_scaled_lasso = self.scaler.fit_transform(X_train_lasso)
        else:
            X_train_scaled_lasso = X_train_lasso
        lasso_model_info["model"].fit(X_train_scaled_lasso, y_train)
        lasso_coefficients = pd.Series(lasso_model_info["model"].coef_, index=X_train_lasso.columns)
        print("\nLasso Regression Coefficients, ordered by importance:")
        print(lasso_coefficients.abs().sort_values(ascending=False))

    def __evaluate_with_feature_configs(self, X_train, X_val, y_train, y_val):
        """
        Compare feature configurations on the VALIDATION split (model/feature selection).
        The test split is never used here — it stays held out for the final report.

        Args:
            X_train: Training features.
            X_val: Validation features.
            y_train: Training target.
            y_val: Validation target.

        Returns:
            rmse_results_df: DataFrame of validation RMSE values for all models and configurations.
        """
        feature_configs = {
            "Full Dataset": X_train.columns,
            "Lags 1-3-5": [col for col in X_train.columns if '_lag_' in col and any(x in col for x in ['_lag_1', '_lag_3', '_lag_5']) or col in ["patient_id", "timestamp"]],
            "Drop PEEP, PH, SPO2": [col for col in X_train.columns if col not in ["peep", "ph", "spo2"] or col in ["patient_id", "timestamp"]],
            "Extensive Drop": [col for col in X_train.columns if col not in [
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
                "mean_blood_pressure_lag_3", "mean_blood_pressure_lag_5"] or col in ["patient_id", "timestamp"]]
        }

        rmse_metrics = {"Model": list(self.models.keys())}

        for config_name, columns in feature_configs.items():
            print(f"\nEvaluating for configuration: {config_name}")

            # Select the columns for the current configuration
            X_train_config = X_train[columns]
            X_val_config = X_val[columns]

            # Collect validation RMSE for all models
            rmse_values = []
            for name, model_info in self.models.items():

                # if the model needs scaling, then we should drop the non-numeric columns
                if model_info.get("use_scaling", True):
                    # Ensure we only select numeric columns for scaling
                    X_train_config = X_train_config.select_dtypes(include=[np.number])
                    X_val_config = X_val_config.select_dtypes(include=[np.number])
                else:
                    X_train_config = X_train[columns]
                    X_val_config = X_val[columns]

                X_train_selected = model_info["column_selector"](X_train_config)
                X_val_selected = model_info["column_selector"](X_val_config)
                if model_info.get("use_scaling", True):
                    self.scaler = StandardScaler()
                    X_train_scaled = self.scaler.fit_transform(X_train_selected)
                    X_val_scaled = self.scaler.transform(X_val_selected)
                else:
                    X_train_scaled = X_train_selected
                    X_val_scaled = X_val_selected
                model_info["model"].fit(X_train_scaled, y_train)
                predictions = model_info["model"].predict(X_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                rmse_values.append(rmse)

            rmse_metrics[config_name] = rmse_values

        # Convert results to DataFrame
        rmse_results_df = pd.DataFrame(rmse_metrics)
        DataFramePrinter.print_dataframe_tabulated(rmse_results_df, "Feature Configuration RMSE Results (Validation)")
        return rmse_results_df

    def __perform_cross_validation(self, X_train, y_train, cv_folds=10):
        """
        Patient-grouped cross-validation on the TRAINING split only (secondary robustness
        check). Folds are grouped by ``patient_id`` so no patient spans train/test folds,
        and ``patient_id``/``timestamp`` are excluded from the model features.

        Args:
            X_train: Training features (must include 'patient_id' for grouping).
            y_train: Training target.
            cv_folds (int): Number of cross-validation folds.

        Returns:
            cv_results_df: DataFrame with mean, SD and worst-fold RMSE across folds.
        """
        cv_results = {"Model": [], "Mean CV RMSE": [], "SD CV RMSE": [], "Worst Fold RMSE": []}
        groups = X_train["patient_id"].to_numpy()
        gkf = GroupKFold(n_splits=cv_folds)

        for name, model_info in self.models.items():
            X_selected = model_info["column_selector"](X_train)
            if model_info.get("use_scaling", True):
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_selected)
            else:
                X_scaled = X_selected
            scores = cross_val_score(
                model_info["model"], X_scaled, y_train, groups=groups, cv=gkf,
                scoring="neg_mean_squared_error",
            )
            # Per-fold RMSE, then mean +/- SD across folds. Reporting the spread (rather than
            # the single sqrt(mean fold MSE) figure used previously) is standard practice and
            # keeps one pathological fold from silently dominating the headline number — the
            # failure mode that hid the pH contamination bug. See
            # notes/linear_regression_cv_anomaly.md.
            fold_rmses = np.sqrt(-scores)
            cv_results["Model"].append(name)
            cv_results["Mean CV RMSE"].append(fold_rmses.mean())
            cv_results["SD CV RMSE"].append(fold_rmses.std())
            cv_results["Worst Fold RMSE"].append(fold_rmses.max())

        cv_results_df = pd.DataFrame(cv_results)
        print("Cross-validation completed.")
        return cv_results_df

    def run_pipeline(self, X_train, X_val, X_test, y_train, y_val, y_test, cv_folds=10):
        """
        Run the full pipeline with a three-way split:
        - train models on the training split,
        - select feature configurations on the VALIDATION split,
        - report final metrics ONCE on the held-out TEST split,
        - plus a patient-grouped CV robustness check on the training split.

        Args:
            X_train, X_val, X_test (DataFrame): Features per split.
            y_train, y_val, y_test (Series): Targets per split.
            cv_folds (int): Number of cross-validation folds.

        Returns:
            results: Dict with test evaluation, grouped-CV, and feature-config results.
        """

        lag_cols = [c for c in X_train.columns if ('_lag_' in c) or (c in ["patient_id", "timestamp"])]
        X_train = X_train[lag_cols]
        X_val = X_val[lag_cols]
        X_test = X_test[lag_cols]

        self.__train_models(X_train, y_train)

        # Final, single-use evaluation on the held-out test split.
        evaluation_results = self.__evaluate_models(X_test, y_test)
        self.__compute_feature_importance(X_train, y_train)

        print("Performing patient-grouped cross-validation on the training split...")
        cross_validation_results = self.__perform_cross_validation(X_train, y_train, cv_folds)

        print("Selecting feature configurations on the validation split...")
        feature_config_results = self.__evaluate_with_feature_configs(X_train, X_val, y_train, y_val)

        results = {
            "evaluation_results": evaluation_results,
            "cross_validation_results": cross_validation_results,
            "feature_config_results": feature_config_results
        }

        return results