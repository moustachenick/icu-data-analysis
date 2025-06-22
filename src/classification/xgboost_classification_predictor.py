from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import numpy as np

class XGBoostClassificationPredictor:

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values
    """

    def __init__(self, model_params=None, feature_selector=None):
        """
        Initializes the XGBoostClassificationPredictor with optional model parameters and feature selector.
        Args:
            model_params (dict): Optional dictionary of parameters for the XGBoost model.
            feature_selector (callable): Optional function to select features from the input DataFrame.
        """
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model = None
        self.feature_selector = feature_selector if feature_selector is not None else self._select_features

    @staticmethod
    def lag_only_feature_selector(X):
        """
        Select only icp_lag_* features (and optionally patient_id, timestamp if present).
        """
        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]
        lagged_cols = [col for col in X.columns if col.startswith('icp_lag_')]
        return X[lagged_cols + [col for col in columns_to_exclude if col in X.columns]]

    def _select_features(self, X):
        """Add commentMore actions
        Select only lagged features for XGBoost classification.
        """
        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]
        lagged_cols = [col for col in X.columns if 'lag' in col]
        return X[lagged_cols + [col for col in columns_to_exclude if col in X.columns]]

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """Add commentMore actions
        Run the pipeline to train and evaluate the model.
        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.
        """
        X_train = self.feature_selector(X_train)
        X_test = self.feature_selector(X_test)

        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]
        X_train_scaled, X_test_scaled = self._perform_normalization(X_train, X_test, columns_to_exclude)

        X_train_final = X_train_scaled.drop(columns=columns_to_exclude, errors='ignore')
        X_test_final = X_test_scaled.drop(columns=columns_to_exclude, errors='ignore')

        results = self.classification(X_train_final, X_test_final, y_train, y_test)

        return results

    def _perform_normalization(self, X_train, X_test, columns_to_exclude):
        """Add commentMore actions
        Performs MinMax scaling on the training and testing data.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            columns_to_exclude (list): List of columns to exclude from scaling.

        Returns:
            tuple: A tuple containing the scaled training DataFrame (pd.DataFrame)
                   and the scaled testing DataFrame (pd.DataFrame).
        """
        scaler = MinMaxScaler()
        columns_to_scale = [col for col in X_train.columns if col not in columns_to_exclude]

        # Create copies to avoid modifying original dataframes
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Fit scaler on training data and transform both train and test
        X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        # for the test set, we only transform (without fitting)
        X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        return X_train_scaled, X_test_scaled

    def classification(self, X_train, X_test, y_train, y_test):
        """
        Trains and evaluates the XGBoost classifier.
        Args:
            X_train: Normalized training features.
            X_test: Normalized testing features.
            y_train: Training target variable.
            y_test: Testing target variable.
        Returns:
            dict: A dictionary containing the results of the classification.
        """
        
        # here we use the ** notation, so that each key-value pair in self.model_params
        # is passed as a separate argument to the XGBClassifier constructor
        if not isinstance(self.model_params, dict):
            raise ValueError("model_params should be a dictionary of parameters for XGBoost.")
        model = xgb.XGBClassifier(**self.model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.model = model

        # Determine the correct name
        is_default_params = (not self.model_params or self.model_params == {})
        is_lagonly = self.feature_selector == self.lag_only_feature_selector
        if is_lagonly and is_default_params:
            name = 'XGBoostClassificationPredictor_lagonly'
        elif is_lagonly and not is_default_params:
            name = 'XGBoostClassificationPredictor_lagonly_with_params'
        elif not is_lagonly and not is_default_params:
            name = 'XGBoostClassificationPredictor_with_params'
        else:
            name = 'XGBoostClassificationPredictor'

        return {
            'name': name,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'y_true': np.array(y_test),
            'y_pred': np.array(y_pred)
        }
