from icu_data_regression_models.RegressionModel import RegressionModel
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# This class is a simple regressor that predicts the ICP based on the closest (timestamp) ICP value for each patient
# in the test set. If the patient is not in the training set, then the average icp value of this patient is predicted.
class BaselineHistoryRegressionModel(RegressionModel, BaseEstimator, RegressorMixin):

    def __init__(self):
        self.data = pd.DataFrame(columns=['patient_id', 'timestamp', 'icp'])

    def fit(self, X_train, y_train):
        # we will use all the data from X_train to make the predictions. we will filter the data to include only the
        # columns patient_id, timestamp and icp, since this Model only uses these columns.

        # drop all columns except patient_id and timestamp
        X_train = X_train[['patient_id', 'timestamp']]
        # concatenate X_train (columns patient_id and timestamp) and y_train (column icp)
        X_train = pd.concat([X_train, y_train], axis=1)

        self.data = X_train

    def predict(self, X_test):
        predictions = []

        # for each row in X_test, we want to find the closest timestamp in the training data (for the same patient) (
        # but not including the timestamp in the test data, we only want to use the data just before the timestamp in
        # the test data)
        for index, row in X_test.iterrows():
            patient_id = row['patient_id']
            timestamp = row['timestamp']

            # filter the training data to include only the rows of the same patient
            patient_data = self.data[self.data['patient_id'] == patient_id]

            # filter the training data to include only the rows before the timestamp in the test data
            patient_data = patient_data[patient_data['timestamp'] < timestamp]

            # if there are no rows before the timestamp, the prediction is the average icp value
            # of all patients
            if patient_data.empty:
                prediction = self.data['icp'].mean()
                predictions.append(prediction)
            else:
                # get the most recent icp value for the patient
                # order the data for this patient by timestamp
                patient_data = patient_data.sort_values(by='timestamp')
                prediction = patient_data['icp'].iloc[-1]
                predictions.append(prediction)

        return predictions

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
