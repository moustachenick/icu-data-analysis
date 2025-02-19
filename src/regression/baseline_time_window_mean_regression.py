from regression.regression import Regression
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# This class is a regressor that predicts the ICP based on the mean ICP value from the last N days for each patient
class BaselineTimeWindowMeanICPRegression(Regression, BaseEstimator, RegressorMixin):

    def __init__(self, days_window=1):
        """
        Initialize the model with the number of days to consider before the test timestamp.
        :param days_window: The number of days before the test timestamp to use for prediction.
        """
        self.data = pd.DataFrame(columns=['patient_id', 'timestamp', 'icp_next'])
        self.days_window = days_window  # This defines how many past days to consider

    def fit(self, X_train, y_train):
        """
        Fit the model on the training data.
        :param X_train: Training features (patient_id, timestamp).
        :param y_train: Target ICP values.
        """
        # Keep only the relevant columns: patient_id and timestamp, and then concatenate with y_train (icp).
        X_train = X_train[['patient_id', 'timestamp']]
        X_train = pd.concat([X_train, y_train], axis=1)
        self.data = X_train

    def predict(self, X_test):
        """
        Predict the ICP values for the test data based on the training data within the specified time window.
        :param X_test: Test data (patient_id, timestamp).
        :return: Predicted ICP values.
        """
        predictions = []

        # Iterate over each row in the test data.
        for index, row in X_test.iterrows():
            patient_id = row['patient_id']
            timestamp = row['timestamp']

            # Filter the training data for the same patient.
            patient_data = self.data[self.data['patient_id'] == patient_id]

            # Filter the training data to include only rows before the test timestamp and within the time window (days_window).
            # Ensure the timestamp is in datetime format
            timestamp = pd.to_datetime(timestamp)
            
            # subtract the days_window from the timestamp to get the cutoff time
            # cutoff_time is the oldest timestamp that we will consider for the mean icp calculation
            # all the rows in the training data that are older than the cutoff_time will be ignored
            cutoff_time = timestamp - pd.Timedelta(days=self.days_window)

            # Ensure the 'timestamp' column is in datetime format
            patient_data_timestamp = pd.to_datetime(patient_data['timestamp'])
            patient_data = patient_data[(patient_data_timestamp < timestamp) &
                                        (patient_data_timestamp >= cutoff_time)]

            # If no data is available in the time window, use the average ICP of all patients.
            if patient_data.empty:
                prediction = self.data['icp_next'].mean()
                predictions.append(prediction)
            else:
                # Calculate the mean icp value within the time window.
                prediction = patient_data['icp_next'].mean()
                predictions.append(prediction)

        return predictions

    def get_params(self, deep=True):
        """
        Return model parameters (primarily the days_window parameter).
        :return: A dictionary of parameters.
        """
        return {'days_window': self.days_window}

    def set_params(self, **params):
        """
        Set the model parameters (primarily the days_window parameter).
        :param params: Dictionary of parameters to set.
        :return: The updated model instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
