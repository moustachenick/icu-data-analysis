from sklearn.metrics import classification_report, confusion_matrix
import re

class BaselinePredictor:
    """
    Rule-based baseline using an aggregate of all lagged ICP values.
    Predicts abnormal if the mean of lagged ICPs exceeds a threshold.
    """

    def __init__(self, threshold=22):
        """
        Args:
            threshold (float): Threshold above which prediction is 'abnormal' (1).
        """
        self.threshold = threshold
        self.icp_lag_cols = []

    def _identify_icp_lag_columns(self, X):
        """
        Auto-detect all columns matching the 'icp_lag_*' pattern.
        """
        return [col for col in X.columns if re.match(r'icp_lag_\d+', col)]

    def run_pipeline(self, X_test, y_test):
        """
        Predict based on the average of all ICP lag columns and evaluate performance.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Ground truth binary labels.
        """
        self.icp_lag_cols = self._identify_icp_lag_columns(X_test)

        if not self.icp_lag_cols:
            raise ValueError("No 'icp_lag_*' columns found in the test set.")

        # Compute the mean ICP across all lagged columns
        icp_avg = X_test[self.icp_lag_cols].mean(axis=1)
        y_pred = (icp_avg >= self.threshold).astype(int)

        print(f"\nRule-Based Prediction (avg({self.icp_lag_cols}) >= {self.threshold} → predict 1):")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        return y_pred
