from src.icu_data_regressor.Regressor import Regressor
import pandas as pd


class BaselineSimpleRegressor(Regressor):

    def __init__(self):
        self.data = pd.DataFrame(columns=['patient_id', 'timestamp', 'icp'])

    def fit(self, X_train, y_train):
        # drop all columns except patient_id and timestamp
        X_train = X_train[['patient_id', 'timestamp']]
        # concatenate X_train (columns patient_id and timestamp) and y_train
        X_train = pd.concat([X_train, y_train], axis=1)

        # for each row in X_train, check if the patient_id is already in self.data
        # if it is not, then add the row to self.data
        # if it is, then check the timestamp.
        # If the timestamp is more recent than the one in self.data, then update the row in self.data

        for index, row in X_train.iterrows():
            # if the patient_id is not in self.data, then add the row to self.data
            if row['patient_id'] not in self.data['patient_id'].values:
                self.data = pd.concat([self.data, row.to_frame().T], ignore_index=True)
            else:
                # if the patient_id is already in self.data, then check the timestamp
                # if the timestamp is more recent than the one in self.data, then update the row in self.data
                if row['timestamp'] > self.data[self.data['patient_id'] == row['patient_id']]['timestamp'].max():
                    # remove the row with the same patient_id from self.data
                    self.data = self.data[self.data['patient_id'] != row['patient_id']]
                    self.data = pd.concat([self.data, row.to_frame().T], ignore_index=True)

    def predict(self, X_test):
        predictions = []

        # for each row in X_test, get the corresponding row from self.data and add it to predictions
        for index, row in X_test.iterrows():
            patient_id = row['patient_id']
            icp = self.data[(self.data['patient_id'] == patient_id)]['icp']
            if not icp.empty:
                predictions.append(icp.iloc[-1])
            else:
                # if the patient_id is not in self.data, then add the average icp value to predictions
                predictions.append(self.data['icp'].mean())
        return pd.DataFrame(predictions, columns=['icp'])
