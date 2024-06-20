import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.icu_data_regressor.BaselineSimpleRegressor import BaselineSimpleRegressor


# Function to print the percentage of rows that have more than 1 column with missing values,
# and the percentage of rows that have exactly 1 column with missing values.
def print_percentages_of_rows_with_missing_values(dataframe):
    # count the number of rows that have more than 1 column with a missing value
    number_of_rows_with_more_than_1_missing_value = dataframe[dataframe.isna().sum(axis=1) > 1].shape[0]
    # print the percentage of rows that have more than 1 column with missing values
    print('Percentage of rows that have more than 1 column with missing values: ',
          number_of_rows_with_more_than_1_missing_value / total_num_of_rows * 100)
    # number of rows with exactly 1 missing value
    number_of_rows_with_1_missing_value = dataframe[dataframe.isna().sum(axis=1) == 1].shape[0]
    # print the percentage of rows that have exactly 1 column with value -1
    print('Percentage of rows that have exactly 1 column with missing values: ',
          number_of_rows_with_1_missing_value / total_num_of_rows * 100)


# read the data from the csv file
df = pd.read_csv('../../../data/final_data.csv', engine='python')
df.head()

# The dataset contains missing values that are represented by different strings ("--", ".", etc).
# declare an array of strings that will be converted to NaN
missing_values_representations = [-1, '-1', '--', '-', '.', "/"]

# Replace missing values with NaN
df.replace(missing_values_representations, np.nan, inplace=True)

# print the total number of rows
total_num_of_rows = df.shape[0]
print()
print('Total number of rows: ', total_num_of_rows)
print()
print('Percentage of rows with missing values initially:')

print_percentages_of_rows_with_missing_values(df)

# drop the 'respiration_rate' column as it has many missing values, and is not needed for the analysis
df.drop(columns=['respiration_rate'], inplace=True)

print('\nDataFrame after dropping the "respiration_rate" column:')

print_percentages_of_rows_with_missing_values(df)

# Let's drop the rows with more than 1 missing value, since we cannot impute them.
missing_count_per_row = df.isnull().sum(axis=1)

filtered_df = df[missing_count_per_row <= 1]

print()
print('DataFrame after dropping rows with more than 1 missing value:')

print_percentages_of_rows_with_missing_values(filtered_df)

# A percentage of the rows have exactly 1 missing value.
# We will fill these missing values, by finding the nearest neighbor.
# We can find the nearest neighbor to the row with the missing value and replace it with that value.
# NOTE: We will use the nearest neighbor of any patient, not just the same patient.

# Separate the "date" and patient_id columns from the rest of the data
# (since they are not useful for the imputation, we will add them back later)
ignored_cols = ['patient_id', 'date_of_birth', 'timestamp']

ignored_data = filtered_df[ignored_cols]
numeric_df = filtered_df.drop(columns=ignored_cols)

missing_count_per_row = numeric_df.isnull().sum(axis=1)

# Split the DataFrame into 2 parts: one with no missing values and one with exactly one missing value
# Rows with no missing values
no_missing_df = numeric_df[missing_count_per_row == 0]

# Rows with exactly one missing value
one_missing_df = numeric_df[missing_count_per_row == 1]

# Initialize KNNImputer with a small number of neighbors (e.g., 1 or 2)
# This will find the nearest neighbor to the row with the missing value and replace it with that value.
imputer = KNNImputer(n_neighbors=1)

# Impute missing values in the DataFrame with one missing value
imputed_one_missing = imputer.fit_transform(one_missing_df)

# Convert the imputed array back to a DataFrame
imputed_one_missing_df = pd.DataFrame(imputed_one_missing, columns=numeric_df.columns, index=one_missing_df.index)

# Combine the DataFrames back together
combined_df = pd.concat([no_missing_df, imputed_one_missing_df])

# Sort the combined DataFrame to maintain the original order
combined_df = combined_df.sort_index()

# Add the ignored columns back to the combined DataFrame
combined_df = pd.concat([ignored_data, combined_df], axis=1)

print('\nDataFrame after imputing the missing values:')
print_percentages_of_rows_with_missing_values(combined_df)

# initialize a new BaseSimpleRegressor object
baseline_regressor = BaselineSimpleRegressor()

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
predicted = prediction['icp'].values[0]

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

# Use a BaselineSimpleRegressor model

baseline_regressor = BaselineSimpleRegressor()

# Fit the model on the training data
baseline_regressor.fit(X_train, y_train)

# Predict the target values on the test data
predictions = baseline_regressor.predict(X_test)

# now we need to compute the mean squared error of the predictions, without importing from sklearn

mse = np.mean((y_test - predictions['icp'].values) ** 2)

print('\nMean Squared Error from BaselineSimpleRegressor:', mse)

# ======================================================================================================================

# Use a LinearRegression model

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
