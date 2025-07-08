import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class LaggedICPBaselinePredictor:
    """
    Baseline predictor using the daily mean ICP value.
    If the daily mean ICP exceeds a specified threshold (e.g. 22), predict abnormal (1), else normal (0).
    """
    def __init__(self, threshold=22):
        """
        Initialize the baseline with a threshold for classifying binary ICP.
        """
        self.threshold = threshold
        self.daily_icp_means = None

    def _select_features(self, X):
        """
        Select only the columns needed for the daily mean baseline, including all icp_lag_* columns.
        """
        required = ["patient_id", "timestamp"]
        lagged = [col for col in X.columns if col.startswith("icp_lag_")]
        return X[[col for col in required + lagged if col in X.columns]].copy()

    def _compute_daily_means(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._select_features(X)
        X["date"] = pd.to_datetime(X["timestamp"]).dt.date
        lagged = [col for col in X.columns if col.startswith("icp_lag_")]
        melted = X.melt(id_vars=["patient_id", "date"], value_vars=lagged,
                        var_name="lag", value_name="icp_lag_value")
        melted = melted.dropna(subset=["icp_lag_value"])
        daily_means = (
            melted.groupby(["patient_id", "date"])["icp_lag_value"]
            .mean()
            .reset_index()
            .rename(columns={"icp_lag_value": "daily_icp_lag_mean"})
        )
        return daily_means

    def fit(self, X: pd.DataFrame):
        self.daily_icp_means = self._compute_daily_means(X)

    def run_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the baseline predictor on the test set.

        Steps:
        1. Checks if the model has been fitted (daily means computed).
        2. Selects relevant features from X_test and extracts the date from the timestamp.
        3. Melts the lagged ICP columns into a long format for aggregation.
        4. Computes the mean of lagged ICP values per patient per day (predicted means).
        5. Aggregates the true labels per patient per day (using max as the daily label).
        6. Merges predicted means with true labels on patient and date.
        7. Binarizes the true and predicted means using the threshold.
        8. Returns a dictionary with the confusion matrix, classification report, and arrays of true and predicted labels.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.

        Returns:
            dict: Contains model name, confusion matrix, classification report, y_true, and y_pred arrays.
        """
        if self.daily_icp_means is None:
            raise ValueError("Must call fit() before run_pipeline().")

        X = self._select_features(X_test)
        X["date"] = pd.to_datetime(X["timestamp"]).dt.date

        lagged = [col for col in X.columns if col.startswith("icp_lag_")]
        melted = X.melt(id_vars=["patient_id", "date"], value_vars=lagged,
                        var_name="lag", value_name="icp_lag_value")
        melted = melted.dropna(subset=["icp_lag_value"])

        # Compute predicted means
        pred_means = (
            melted.groupby(["patient_id", "date"])["icp_lag_value"]
            .mean()
            .reset_index()
            .rename(columns={"icp_lag_value": "daily_icp_lag_mean"})
        )

        # Derive true labels per patient/day
        y_test_df = X_test.copy()
        y_test_df["date"] = pd.to_datetime(y_test_df["timestamp"]).dt.date
        y_test_df["y_true"] = y_test.values
        y_true_agg = y_test_df.groupby(["patient_id", "date"])["y_true"].max().reset_index()

        # Join predictions with true labels
        merged = pred_means.merge(y_true_agg, on=["patient_id", "date"], how="inner")

        y_true_bin = (merged["y_true"] >= self.threshold).astype(int)
        y_pred_bin = (merged["daily_icp_lag_mean"] >= self.threshold).astype(int).fillna(0)

        return {
            'name': 'LaggedICPBaselinePredictor',
            'confusion_matrix': confusion_matrix(y_true_bin, y_pred_bin),
            'classification_report': classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0),
            'y_true': np.array(y_true_bin),
            'y_pred': np.array(y_pred_bin),
            
        }