import re

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from regression.regression import Regression

_ICP_LAG_RE = re.compile(r"^icp_lag_(\d+)$")


# Persistence (last-value-carried-forward) baseline: predicts the patient's most recent past
# ICP, read from the `icp_lag_1` feature already present in each row. Because the train/val/test
# split is patient-level disjoint, a cross-patient `patient_id` lookup is always empty, so we use
# the per-row lagged ICP (strictly past data) instead. Falls back to the training-mean ICP only
# if no `icp_lag_*` column is available.
class BaselineHistoryRegression(Regression, BaseEstimator, RegressorMixin):

    def __init__(self):
        self._fallback = 0.0

    def fit(self, X_train, y_train):
        # The prediction comes from per-row lagged ICP features; we only need a scalar fallback
        # for the (unexpected) case where no icp_lag_* column is present at predict time.
        self._fallback = float(pd.Series(y_train).mean())
        return self

    def predict(self, X_test):
        lag_cols = self.__icp_lag_columns(X_test)
        if not lag_cols:
            return [self._fallback] * len(X_test)

        # Most recent past ICP = lowest-numbered available lag (normally icp_lag_1).
        most_recent_col = min(lag_cols, key=lambda c: int(_ICP_LAG_RE.match(c).group(1)))
        return X_test[most_recent_col].to_numpy()

    @staticmethod
    def __icp_lag_columns(X):
        return [c for c in X.columns if _ICP_LAG_RE.match(c)]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
