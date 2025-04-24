from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

class ClassificationPredictor:

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values
    """

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """
        Run the pipeline to train and evaluate the model.
        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.
        """

        columns_to_exclude = ["patient_id", "date_of_birth", "timestamp"]

        print("Dataframe before normalization:")
        print(X_train.head())

        # Normalize the data before training
        X_train_scaled, X_test_scaled = self._perform_normalization(X_train, X_test, columns_to_exclude)

        X_train_final = X_train_scaled.drop(columns=columns_to_exclude, errors='ignore')
        X_test_final = X_test_scaled.drop(columns=columns_to_exclude, errors='ignore')

        # Continue with training and evaluation
        model = self.classification(X_train_final, X_test_final, y_train, y_test)
        return model

    def _perform_normalization(self, X_train, X_test, columns_to_exclude):
        """
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

        print("Training and testing data successfully normalized.")
        print("Sample of normalized training data:")
        print(X_train_scaled.head())

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
            xgb.XGBClassifier: The trained model.
        """
        # Train XGBoost model
        # TODO tune the parameters for the XGBoost model according to [https://xgboost.readthedocs.io/en/latest/parameter.html]
        model_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        print("\n~~~~~~~~ XGBoost Predictor ~~~~~~~~\n")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        return model
        