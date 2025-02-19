import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ClassificationPredictor:

    """
    A class that uses the XGboost algorithm to predict abnormal Intracranial Pressure Values
    """

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """
        Run the pipeline to train and evaluate the model.
        Args:
            X_train: Training features.
            X_test: Testing features.
            y_train: Training target variable.
            y_test: Testing target variable.
        """

        # Normalize the data before training
        X_train = self.normalization(X_train)
        X_test = self.normalization(X_test)

        print(X_train.head())

        # Continue with training and evaluation

    def normalization(self, data, exclude_columns=None):
        """
        Normalize the data using Min-Max scaling while excluding specified columns.

        Args:
            data (pd.DataFrame): The data to be normalized.
            exclude_columns (list, optional): List of columns to exclude from normalization.

        Returns:
            pd.DataFrame: The normalized data.
        """
        if exclude_columns is None:
            exclude_columns = ["icp_next", "patient_id", "date_of_birth", "timestamp"]

        columns_to_scale = [col for col in data.columns if col not in exclude_columns]
        scaler = MinMaxScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

        print("Data successfully normalized.")
        return data