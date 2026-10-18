import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from helper.data_frame_printer import DataFramePrinter
from regression.baseline_history_regression import BaselineHistoryRegression
from regression.baseline_mean_regression import BaselineMeanRegression

ID_COLUMNS = ["patient_id", "timestamp"]

# True-ICP ranges (mmHg) for the stratified test report.
ICP_RANGE_BINS = [-np.inf, 20, 30, np.inf]
ICP_RANGE_LABELS = ["≤20", "20-30", ">30"]

# (model, reference) pairs for the per-patient Wilcoxon signed-rank test: tree vs. linear
# models (reviewer question) and ML models vs. the persistence baseline.
WILCOXON_PAIRS = [
    ("Random Forest", "Linear Regression"),
    ("XGBoost", "Linear Regression"),
    ("Linear Regression", "Baseline History Regression"),
    ("Random Forest", "Baseline History Regression"),
]


def numeric_features(X):
    return X.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore")


def all_columns(X):
    return X


class RegressionPipeline:
    def __init__(self):
        """
        Initialize the ICPPrediction class.

        Each linear model is a Pipeline (StandardScaler -> model), so the scaler is always fitted
        on exactly the data the model is fitted on — including inside every CV training fold.
        Tree models need no scaling. Models with a "param_grid" are tuned on the validation split.
        """
        self.models = {
            "Linear Regression": {
                "model": make_pipeline(StandardScaler(), LinearRegression()),
                "column_selector": numeric_features,
            },
            "Ridge Regression": {
                "model": make_pipeline(StandardScaler(), Ridge()),
                "column_selector": numeric_features,
                "param_grid": {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
            },
            "Lasso Regression": {
                "model": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
                "column_selector": numeric_features,
                "param_grid": {"lasso__alpha": [0.001, 0.01, 0.1, 1]},
            },
            "Random Forest": {
                "model": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
                "column_selector": numeric_features,
                "param_grid": {
                    "max_depth": [None, 10, 20],
                    "min_samples_leaf": [1, 5, 20],
                    "max_features": [1.0, "sqrt"],
                },
            },
            "XGBoost": {
                "model": XGBRegressor(n_jobs=-1, random_state=42),
                "column_selector": numeric_features,
                "param_grid": {
                    "n_estimators": [200, 500],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.03, 0.1],
                    "subsample": [0.8, 1.0],
                },
            },
            "Baseline History Regression": {
                "model": BaselineHistoryRegression(),
                "column_selector": all_columns,
            },
            "Baseline Mean Regression": {
                "model": BaselineMeanRegression(),
                "column_selector": all_columns,
            }
        }

    def __tune_hyperparameters(self, X_train, X_val, y_train, y_val):
        """
        Grid search on the VALIDATION split: fit every combination on the training split and
        keep the one with the lowest validation RMSE. The selected parameters are set on the
        model, so training, CV and the feature configurations all use them.

        Args:
            X_train: Training features.
            X_val: Validation features.
            y_train: Training target.
            y_val: Validation target.

        Returns:
            tuning_df: DataFrame with the selected parameters and validation RMSE per model.
        """
        tuning = {"Model": [], "Selected Parameters": [], "Validation RMSE": []}

        for name, model_info in self.models.items():
            if "param_grid" not in model_info:
                continue
            X_train_selected = model_info["column_selector"](X_train)
            X_val_selected = model_info["column_selector"](X_val)

            best_params, best_rmse = None, np.inf
            for params in ParameterGrid(model_info["param_grid"]):
                model = clone(model_info["model"]).set_params(**params).fit(X_train_selected, y_train)
                rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val_selected)))
                if rmse < best_rmse:
                    best_params, best_rmse = params, rmse

            model_info["model"].set_params(**best_params)
            tuning["Model"].append(name)
            tuning["Selected Parameters"].append(best_params)
            tuning["Validation RMSE"].append(best_rmse)
            print(f"{name}: selected {best_params} (validation RMSE = {best_rmse:.4f})")

        tuning_df = pd.DataFrame(tuning)
        DataFramePrinter.print_dataframe_tabulated(tuning_df, "Hyperparameter Tuning Results (Validation)")
        return tuning_df

    def __train_models(self, X_train, y_train):
        """
        Train all models on the training data.

        Args:
            X_train: Training features.
            y_train: Training target variable.
        """
        for name, model_info in self.models.items():
            model_info["model"].fit(model_info["column_selector"](X_train), y_train)
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
        metrics = {"Model": [], "MSE": [], "MAE": [], "RMSE": [], "R²": []}

        for name, model_info in self.models.items():
            print(f"\nEvaluating {name}...")
            predictions = model_info["model"].predict(model_info["column_selector"](X_test))

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            metrics["Model"].append(name)
            metrics["MSE"].append(mse)
            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)
            metrics["R²"].append(r2)

            print(f"{name}: MSE = {mse:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")

        return pd.DataFrame(metrics)

    def __evaluate_by_icp_range(self, X_test, y_test):
        """
        RMSE and MAE on the test split, stratified by the TRUE ICP value. Aggregate RMSE is
        dominated by the few high-ICP rows, so this shows where each model gains or loses.

        Args:
            X_test: Testing features.
            y_test: Testing target variable.

        Returns:
            range_df: DataFrame with RMSE and MAE per ICP range for all models.
        """
        ranges = pd.cut(y_test, bins=ICP_RANGE_BINS, labels=ICP_RANGE_LABELS)
        metrics = {"Model": list(self.models.keys())}

        for label in ICP_RANGE_LABELS:
            in_range = (ranges == label).to_numpy()
            rmse_values, mae_values = [], []
            for model_info in self.models.values():
                if not in_range.any():
                    rmse_values.append(np.nan)
                    mae_values.append(np.nan)
                    continue
                predictions = model_info["model"].predict(model_info["column_selector"](X_test[in_range]))
                rmse_values.append(np.sqrt(mean_squared_error(y_test[in_range], predictions)))
                mae_values.append(mean_absolute_error(y_test[in_range], predictions))
            metrics[f"RMSE {label} (n={in_range.sum()})"] = rmse_values
            metrics[f"MAE {label}"] = mae_values

        range_df = pd.DataFrame(metrics)
        DataFramePrinter.print_dataframe_tabulated(range_df, "Regression Results by True ICP Range (Test)")
        return range_df

    def __compare_models_wilcoxon(self, X_test, y_test):
        """
        Wilcoxon signed-rank test on per-patient MAE in the test split. The patient is the unit
        of analysis, because rows of one patient are not independent.

        Args:
            X_test: Testing features (must include 'patient_id').
            y_test: Testing target variable.

        Returns:
            wilcoxon_df: DataFrame with the median per-patient MAE difference and p-value per pair.
        """
        patient_mae = {}
        for name, model_info in self.models.items():
            predictions = model_info["model"].predict(model_info["column_selector"](X_test))
            abs_errors = pd.Series(np.abs(y_test.to_numpy() - predictions), index=X_test.index)
            patient_mae[name] = abs_errors.groupby(X_test["patient_id"]).mean()

        comparisons = {"Model": [], "Reference": [], "Patients": [],
                       "Median MAE Difference": [], "p-value": []}
        for model, reference in WILCOXON_PAIRS:
            differences = patient_mae[model] - patient_mae[reference]
            comparisons["Model"].append(model)
            comparisons["Reference"].append(reference)
            comparisons["Patients"].append(len(differences))
            comparisons["Median MAE Difference"].append(differences.median())
            comparisons["p-value"].append(wilcoxon(patient_mae[model], patient_mae[reference]).pvalue)

        wilcoxon_df = pd.DataFrame(comparisons)
        DataFramePrinter.print_dataframe_tabulated(
            wilcoxon_df, "Wilcoxon Signed-Rank Test on Per-Patient MAE (Test; negative difference = model better)"
        )
        return wilcoxon_df

    def __compute_feature_importance(self, X_train):
        """
        Display the coefficients of the trained Linear Regression and Lasso models, ordered by
        absolute value. Coefficients are on the standardized scale, so they are comparable.

        Args:
            X_train: Training features (used for the column names only).
        """
        for name in ["Linear Regression", "Lasso Regression"]:
            model_info = self.models[name]
            columns = model_info["column_selector"](X_train).columns
            coefficients = pd.Series(model_info["model"][-1].coef_, index=columns)
            ordered = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
            print(f"\n{name} Coefficients (signed), ordered by absolute value:")
            print(ordered)

    def __compute_permutation_importance(self, X_val, y_val):
        """
        Permutation importance of the tree models on the VALIDATION split: the increase in RMSE
        when one feature is shuffled. Unlike impurity-based importance, it is not biased toward
        continuous features with many unique values.

        Args:
            X_val: Validation features.
            y_val: Validation target.
        """
        for name in ["Random Forest", "XGBoost"]:
            model_info = self.models[name]
            X_val_selected = model_info["column_selector"](X_val)
            result = permutation_importance(
                model_info["model"], X_val_selected, y_val,
                scoring="neg_root_mean_squared_error", n_repeats=5, random_state=42,
            )
            importance = pd.Series(result.importances_mean, index=X_val_selected.columns)
            print(f"\n{name} Permutation Importance (validation RMSE increase), ordered by importance:")
            print(importance.sort_values(ascending=False))

    @staticmethod
    def build_feature_configs(columns):
        """
        Build the feature configurations compared on the validation split.

        Args:
            columns: All available columns (lag features, pathology one-hot, ID columns).

        Returns:
            Dict of configuration name -> list of columns (ID columns always included).
        """
        def base_variable(col):
            return col.split("_lag_")[0]

        ids = [c for c in columns if c in ID_COLUMNS]
        lags = [c for c in columns if "_lag_" in c]
        pathology = [c for c in columns if c.startswith("pathology_")]

        extensive_drop = [
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
            "mean_blood_pressure_lag_3", "mean_blood_pressure_lag_5"]

        return {
            "Full Dataset": ids + lags,
            "Lags 1-3-5": ids + [c for c in lags if c.rsplit("_lag_", 1)[1] in ["1", "3", "5"]],
            # Drop every lag of these variables (the bare columns are never model inputs).
            # PEEP is excluded from the whole analysis in preprocessing.
            "Drop PH, SPO2": ids + [c for c in lags if base_variable(c) not in ["ph", "spo2"]],
            "Extensive Drop": ids + [c for c in lags if c not in extensive_drop],
            "Lags + Pathology": ids + lags + pathology,
        }

    def __evaluate_with_feature_configs(self, X_train, X_val, y_train, y_val):
        """
        Compare feature configurations on the VALIDATION split (model/feature selection).
        The test split is never used here — it stays held out for the final report.
        Each model is cloned, so the trained models used for the test report are not changed.

        Args:
            X_train: Training features.
            X_val: Validation features.
            y_train: Training target.
            y_val: Validation target.

        Returns:
            rmse_results_df: DataFrame of validation RMSE values for all models and configurations.
        """
        feature_configs = self.build_feature_configs(X_train.columns)

        rmse_metrics = {"Model": list(self.models.keys())}

        for config_name, columns in feature_configs.items():
            print(f"\nEvaluating for configuration: {config_name}")

            rmse_values = []
            for name, model_info in self.models.items():
                X_train_selected = model_info["column_selector"](X_train[columns])
                X_val_selected = model_info["column_selector"](X_val[columns])
                model = clone(model_info["model"]).fit(X_train_selected, y_train)
                predictions = model.predict(X_val_selected)
                rmse_values.append(np.sqrt(mean_squared_error(y_val, predictions)))

            rmse_metrics[config_name] = rmse_values

        # Convert results to DataFrame
        rmse_results_df = pd.DataFrame(rmse_metrics)
        DataFramePrinter.print_dataframe_tabulated(rmse_results_df, "Feature Configuration RMSE Results (Validation)")
        return rmse_results_df

    def __perform_cross_validation(self, X_train, y_train, cv_folds=10):
        """
        Patient-grouped cross-validation on the TRAINING split only (secondary robustness
        check). Folds are grouped by ``patient_id`` so no patient spans train/test folds,
        and ``patient_id``/``timestamp`` are excluded from the model features. Scaling is part
        of each model's Pipeline, so it is refitted on every training fold.

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
            scores = cross_val_score(
                model_info["model"], model_info["column_selector"](X_train), y_train,
                groups=groups, cv=gkf, scoring="neg_mean_squared_error",
            )
            # Per-fold RMSE, then mean +/- SD across folds. Reporting the spread (rather than
            # the single sqrt(mean fold MSE) figure used previously) is standard practice and
            # keeps one pathological fold from silently dominating the headline number.
            # See notes/linear_regression_cv_anomaly.md.
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
        - tune hyperparameters on the VALIDATION split,
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
        # The main models use lag features only; pathology columns are kept for the
        # "Lags + Pathology" feature configuration.
        lag_cols = [c for c in X_train.columns if ('_lag_' in c) or (c in ID_COLUMNS)]
        config_cols = lag_cols + [c for c in X_train.columns if c.startswith("pathology_")]
        X_train_configs, X_val_configs = X_train[config_cols], X_val[config_cols]
        X_train = X_train[lag_cols]
        X_val = X_val[lag_cols]
        X_test = X_test[lag_cols]

        print("Tuning hyperparameters on the validation split...")
        tuning_results = self.__tune_hyperparameters(X_train, X_val, y_train, y_val)

        self.__train_models(X_train, y_train)

        # Final, single-use evaluation on the held-out test split.
        evaluation_results = self.__evaluate_models(X_test, y_test)
        icp_range_results = self.__evaluate_by_icp_range(X_test, y_test)
        wilcoxon_results = self.__compare_models_wilcoxon(X_test, y_test)
        self.__compute_feature_importance(X_train)
        self.__compute_permutation_importance(X_val, y_val)

        print("Performing patient-grouped cross-validation on the training split...")
        cross_validation_results = self.__perform_cross_validation(X_train, y_train, cv_folds)

        print("Selecting feature configurations on the validation split...")
        feature_config_results = self.__evaluate_with_feature_configs(
            X_train_configs, X_val_configs, y_train, y_val
        )

        results = {
            "tuning_results": tuning_results,
            "evaluation_results": evaluation_results,
            "icp_range_results": icp_range_results,
            "wilcoxon_results": wilcoxon_results,
            "cross_validation_results": cross_validation_results,
            "feature_config_results": feature_config_results
        }

        return results
