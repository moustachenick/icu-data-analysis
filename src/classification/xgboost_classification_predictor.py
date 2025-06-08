from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import numpy as np

class XGBoostClassificationPredictor:

    def _select_features(self, X):
        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]
        lagged_cols = [col for col in X.columns if 'lag' in col]
        return X[lagged_cols + [col for col in columns_to_exclude if col in X.columns]]

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        X_train = self._select_features(X_train)
        X_test = self._select_features(X_test)

        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]
        X_train_scaled, X_test_scaled = self._perform_normalization(X_train, X_test, columns_to_exclude)

        X_train_final = X_train_scaled.drop(columns=columns_to_exclude, errors='ignore')
        X_test_final = X_test_scaled.drop(columns=columns_to_exclude, errors='ignore')

        results = self.classification(X_train_final, X_test_final, y_train, y_test)


        return results

    def _perform_normalization(self, X_train, X_test, columns_to_exclude):
        scaler = MinMaxScaler()
        columns_to_scale = [col for col in X_train.columns if col not in columns_to_exclude]

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        return X_train_scaled, X_test_scaled

    def classification(self, X_train, X_test, y_train, y_test):
        model_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'scale_pos_weight': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.model = model


        return {
            'name': 'XGBoostClassificationPredictor',
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'y_true': np.array(y_test),
            'y_pred': np.array(y_pred)
        }
