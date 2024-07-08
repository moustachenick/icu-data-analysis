import pandas as pd



class BaselineAdvancedRegressor:
    def __init__(self):
        self.data = pd.DataFrame(columns=['patient_id', 'timestamp', 'icp'])

    def fit(self, X_train, y_train):
        # Concatenate X_train (columns patient_id and timestamp) and y_train (icp)
        X_train = X_train[['patient_id', 'timestamp']]
        self.data = pd.concat([X_train, y_train], axis=1)

    def predict(self, X_test):
        predictions = []

        for index, row in X_test.iterrows():
            patient_id = row['patient_id']
            timestamp = row['timestamp']

            # Filter the training data to include only the rows of the same patient and before the current timestamp
            patient_data = self.data[(self.data['patient_id'] == patient_id) & (self.data['timestamp'] < timestamp)]

            # If there are no previous records, use the overall mean ICP as a fallback
            if patient_data.empty:
                prediction = self.data['icp'].mean()
            else:
                # Calculate the mean ICP value for all previous timestamps
                prediction = patient_data['icp'].mean()

            predictions.append(prediction)

        return predictions
