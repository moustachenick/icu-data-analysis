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

    def fit(self, X: pd.DataFrame):
        """
        Precompute the mean ICP per patient per day from the provided data.
        """
        # Use only required columns
        X = self._select_features(X)
        if 'timestamp' not in X.columns or 'patient_id' not in X.columns:
            raise ValueError("Required columns: 'patient_id' and 'timestamp'.")
        df = X.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        lagged = [col for col in df.columns if col.startswith("icp_lag_")]
        melted = df.melt(id_vars=["patient_id", "date"], value_vars=lagged, var_name="lag", value_name="icp_lag_value")
        # Remove NaNs
        melted = melted.dropna(subset=["icp_lag_value"])
        self.daily_icp_means = (
            melted.groupby(["patient_id", "date"])['icp_lag_value']
            .mean()
            .rename("daily_icp_lag_mean")
        )

    def run_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Predicts based on whether the daily mean ICP ≥ threshold.
        Returns a dict with confusion matrix, classification report, y_true, y_pred, and the predictor name.
        """
        X = self._select_features(X_test)
        if self.daily_icp_means is None:
            raise ValueError("Must call fit() before run_pipeline().")
        X['date'] = pd.to_datetime(X['timestamp']).dt.date
        lagged = [col for col in X.columns if col.startswith("icp_lag_")]
        melted = X.melt(id_vars=["patient_id", "date"], value_vars=lagged, var_name="lag", value_name="icp_lag_value")
        melted = melted.dropna(subset=["icp_lag_value"])
        joined = melted.set_index(["patient_id", "date"]) \
                   .join(self.daily_icp_means, how='left') \
                   .reset_index()
        pred_df = joined[["patient_id", "date", "daily_icp_lag_mean"]].drop_duplicates()
        y_pred = (pred_df['daily_icp_lag_mean'] >= self.threshold).astype(int).fillna(0)

        y_test_df = X_test.copy()
        y_test_df['date'] = pd.to_datetime(y_test_df['timestamp']).dt.date
        y_test_df['y_true'] = y_test.values
        y_true_agg = y_test_df.groupby(['patient_id', 'date'])['y_true'].max().reset_index()
        merged = pred_df.merge(y_true_agg, on=['patient_id', 'date'], how='inner')
        y_true_bin = (merged['y_true'] >= self.threshold).astype(int)
        y_pred_bin = (merged['daily_icp_lag_mean'] >= self.threshold).astype(int)

        return {
            'name': 'LaggedICPBaselinePredictor',
            'confusion_matrix': confusion_matrix(y_true_bin, y_pred_bin),
            'classification_report': classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0),
            'y_true': np.array(y_true_bin),
            'y_pred': np.array(y_pred_bin)
        }
