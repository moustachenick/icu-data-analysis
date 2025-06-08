from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

class LatestICPBaselinePredictor:
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
        Returns a dict with confusion matrix, classification report, y_true, y_pred, and the predictor name.
        """
        if self.icp_lag_col not in X_test.columns:
            raise ValueError(f"'{self.icp_lag_col}' column is missing from test data.")
        
        if "patient_id" not in X_test.columns or "timestamp" not in X_test.columns:
            raise ValueError("Missing 'patient_id' or 'timestamp' in X_test for McNemar comparison.")

        y_pred = (X_test[self.icp_lag_col] > self.threshold).astype(int)

        # Get patient_id and date for alignment
        patient_ids = X_test["patient_id"].values
        dates = pd.to_datetime(X_test["timestamp"]).dt.date.values

        return {
            'name': 'LatestICPBaselinePredictor',
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'y_true': np.array(y_test),
            'y_pred': np.array(y_pred),
            
        }

