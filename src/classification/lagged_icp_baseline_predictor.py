import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class LaggedICPBaselinePredictor:
    """
    A simplified rule-based baseline predictor for Intracranial Pressure (ICP) classification.

    This predictor computes the row-wise mean of a selected number of lagged ICP values
    (e.g., icp_lag_1 to icp_lag_5) for each sample. If the average exceeds a specified threshold,
    it classifies the sample as abnormal (1), otherwise as normal (0).

    This approach mimics real-time decision making by only using past ICP values (lags)
    that would be available at prediction time.

    Attributes:
        threshold (float): Decision threshold for binary classification.
        max_lag (int): The number of lagged ICP features to include (starting from icp_lag_1).
    """

    def __init__(self, decision_threshold=16, max_lag=5):
        """
        Initialize the baseline predictor.

        Args:
            decision_threshold (float): Threshold above which the mean ICP is considered abnormal.
            max_lag (int): Number of lagged features to use (e.g., use icp_lag_1 to icp_lag_max_lag).
        """
        self.threshold = decision_threshold
        self.max_lag = max_lag

    def _select_features(self, X):
        """
        Select the required columns: patient_id, timestamp, and the specified ICP lag features.

        Args:
            X (pd.DataFrame): Input feature dataframe.

        Returns:
            pd.DataFrame: Subset of X containing only relevant columns.
        """
        required = ["patient_id", "timestamp"]
        lagged = [f"icp_lag_{i}" for i in range(1, self.max_lag + 1)]
        return X[[col for col in required + lagged if col in X.columns]].copy()

    def run_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Apply the baseline predictor to the test set.

        For each row, compute the mean of the specified lagged ICP values, compare it
        to the threshold, and generate a binary prediction.

        Args:
            X_test (pd.DataFrame): Test feature set containing lagged ICP columns.
            y_test (pd.Series): Ground truth continuous ICP values.

        Returns:
            dict: Contains predictor name, confusion matrix, classification report, and
                  arrays of binary true and predicted labels.
        """
        X = self._select_features(X_test)
        lagged_cols = [col for col in X.columns if col.startswith("icp_lag_")]

        # Row-wise mean of the lagged values
        icp_means = X[lagged_cols].mean(axis=1, skipna=True)

        # Apply threshold for binary classification
        y_pred_bin = (icp_means >= self.threshold).astype(int)
        y_true_bin = (y_test >= self.threshold).astype(int)

        return {
            'name': 'LaggedICPBaselinePredictor',
            'confusion_matrix': confusion_matrix(y_true_bin, y_pred_bin),
            'classification_report': classification_report(
                y_true_bin, y_pred_bin, output_dict=True, zero_division=0
            ),
            'y_true': y_true_bin.to_numpy(),
            'y_pred': y_pred_bin.to_numpy()
        }
