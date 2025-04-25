from sklearn.metrics import classification_report, confusion_matrix

class BaselinePredictor:
    """
    Real-life like baseline predictor using the latest ICP value (icp_lag_1).
    If icp_lag_1 exceeds a specified threshold (e.g. 16 or 18), predict abnormal (1), else normal (0).
    """

    def __init__(self, icp_lag_col="icp_lag_1", decision_threshold=16):
        """
        Args:
            icp_lag_col (str): The lag column to use (typically 'icp_lag_1').
            decision_threshold (float): Threshold above which prediction is abnormal.
        """
        self.icp_lag_col = icp_lag_col
        self.threshold = decision_threshold

    def run_pipeline(self, X_test, y_test):
        """
        Applies the rule: if icp_lag_1 > threshold → predict 1, else 0.

        Args:
            X_test (pd.DataFrame): Test features.
            Y_test (pd.Series): Ground truth binary labels.
        """
        if self.icp_lag_col not in X_test.columns:
            raise ValueError(f"'{self.icp_lag_col}' column is missing from test data.")

        y_pred = (X_test[self.icp_lag_col] > self.threshold).astype(int)

        print(f"\nRule-Based Prediction: if {self.icp_lag_col} > {self.threshold} → predict 1")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        return y_pred

