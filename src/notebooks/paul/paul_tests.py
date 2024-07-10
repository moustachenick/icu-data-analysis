import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_regression_models.BaselineHistoryRegressionModel import BaselineHistoryRegressionModel

data_processor = DataPreProcessor()

filtered_df = data_processor.replace_missing_values()

combined_df = data_processor.known_nearest_neighbor_imputer(filtered_df)

# initialize a new BaseSimpleRegressor object
baseline_regressor = BaselineHistoryRegressionModel()

# We want to predict the ICP value for patient with id "1001", at the timestamp "2013-12-21 11:30:00".
# So we need to train the Regressor on the data up to that timestamp.
# We will filter the data to include only the rows of this patient_id, and up to that timestamp.

# Filter the data to include only the rows of patient_id "1001"
patient_id = 1001
filtered_data = combined_df[combined_df['patient_id'] == patient_id]

# Filter the data to include only the rows up to ( and not including) timestamp "2013-12-21 11:30:00"
timestamp = '2013-12-21 11:00:00'
filtered_data = filtered_data[filtered_data['timestamp'] < timestamp]

# Split the filtered data into features (X) and target (y)
X_train = filtered_data.drop(columns=['icp'])
y_train = filtered_data['icp']

# Fit the Regressor on the filtered data
baseline_regressor.fit(X_train, y_train)

# Predict the ICP value for the patient_id at the timestamp

# Filter the data to include only the rows of this patient_id, and at that timestamp

X_test = combined_df[(combined_df['patient_id'] == patient_id) & (combined_df['timestamp'] == timestamp)]
y_test = X_test['icp']

prediction = baseline_regressor.predict(X_test.drop(columns=['icp']))

real = y_test.values[0]
# Get the predicted ICP value
predicted = prediction[0]

# Using f-strings for aligned printing
print(
    f"\n{'Predicted ICP value for patient_id:':<40} {patient_id:<10} {'at timestamp:':<20} {timestamp:<20} {'is:':<5} {predicted:<10}")
print(
    f"{'Real ICP value for patient_id:':<40} {patient_id:<10} {'at timestamp:':<20} {timestamp:<20} {'is:':<5} {real:<10}")

# We can now use the Regressor to predict the ICP values for the rest of the data.

# Split the data into features (X) and target (y)
X = combined_df.drop(columns=['icp'])
y = combined_df['icp']

# Let's use only the first 20% of the data, for speed (this will be removed in the final version)
X_original = X
X = X[:int(0.2 * X.shape[0])]
y = y[:int(0.2 * y.shape[0])]

# we need to create a train-test split of the data
# we will use the first 80% of the data for training and the rest for testing

train_size = int(0.8 * X.shape[0])

print('\nTotal size: ', X.shape[0], '\tTrain size:', train_size, '\tTest size:', X.shape[0] - train_size,
      '\tPercentage of the original dataset:', round((train_size / X_original.shape[0]) * 100), '%')

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# Use a BaselineHistoryRegressionModel model

baseline_regressor = BaselineHistoryRegressionModel()

# Fit the model on the training data
baseline_regressor.fit(X_train, y_train)

# Predict the target values on the test data
predictions = baseline_regressor.predict(X_test)

# now we need to compute the mean squared error of the predictions, without importing from sklearn

mse = np.mean((y_test - predictions) ** 2)

print('\nMean Squared Error from BaselineHistoryRegressionModel:', mse)

# ======================================================================================================================

# Use a Linear Regression model

model = LinearRegression()

# For the LinearRegression model, we need to drop the 'timestamp', 'patient_id', and 'date_of_birth' columns

X_train = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target values on the test data
predictions = model.predict(X_test)

# Calculate the mean squared error of the predictions

mse = mean_squared_error(y_test, predictions)

print('\nMean Squared Error from LinearRegression:', mse)
