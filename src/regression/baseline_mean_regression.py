import re

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from regression.regression import Regression

_ICP_LAG_RE = re.compile(r"^icp_lag_(\d+)$")


# Recent-mean baseline: predicts the mean of the patient's available past ICP readings, taken
# from the `icp_lag_*` features already present in each row. Because the train/val/test split is
# patient-level disjoint, a cross-patient `patient_id` lookup is always empty, so we use the
# per-row lagged ICP (strictly past data) instead. Falls back to the training-mean ICP only if
# no `icp_lag_*` column is available.
class BaselineMeanRegression(Regression, BaseEstimator, RegressorMixin):
    def __init__(self):
        self._fallback = 0.0

    def fit(self, X_train, y_train):
        # The prediction comes from per-row lagged ICP features; we only need a scalar fallback
        # for the (unexpected) case where no icp_lag_* column is present at predict time.
        self._fallback = float(pd.Series(y_train).mean())
        return self

    def predict(self, X_test):
        lag_cols = [c for c in X_test.columns if _ICP_LAG_RE.match(c)]
        if not lag_cols:
            return [self._fallback] * len(X_test)

        # Mean of the patient's recent ICP history (all available lags).
        return X_test[lag_cols].mean(axis=1).to_numpy()

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
