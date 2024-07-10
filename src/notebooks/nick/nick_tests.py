import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from icu_data_regression_models.baselineadvancedregressor import BaselineAdvancedRegressor
from icu_data_parser.datapreprocessor import DataPreProcessor



data_processor = DataPreProcessor()

filtered_df = data_processor.replace_missing_values(pd.DataFrame())

combined_df = data_processor.known_nearest_neighbor_imputer(filtered_df)

# Split the data into training and testing sets
train_size = int(len(combined_df) * 0.8)
train_df = combined_df.iloc[:train_size]
test_df = combined_df.iloc[train_size:]

# Prepare training and testing sets
X_train = train_df.drop(columns=['icp'])
y_train = train_df['icp']
X_test = test_df.drop(columns=['icp'])
y_test = test_df['icp']


# Initialize and train the Baseline Advanced Regressor
baseline_regressor = BaselineAdvancedRegressor()
baseline_regressor.fit(X_train, y_train)

# Make predictions with the Baseline Advanced Regressor
baseline_predictions = baseline_regressor.predict(test_df)

# For the LinearRegression model, we need to drop the 'timestamp', 'patient_id', and 'date_of_birth' columns

X_train = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Initialize and train the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Make predictions with the Linear Regression model
linear_predictions = linear_regressor.predict(X_test)

# Evaluate both models
baseline_mse = mean_squared_error(y_test, baseline_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)

print('Baseline Advanced Regressor Mean Squared Error:', baseline_mse)
print('Linear Regression Mean Squared Error:', linear_mse)

# Extract actual ICP values and patient_ids from the test set
actual_icp = y_test.values
patient_ids = test_df['patient_id'].values

# Plot 1: Actual ICP vs Baseline Advanced Regressor Predictions
plt.figure(figsize=(14, 7))
plt.plot(actual_icp, label='Actual ICP', color='blue', alpha=0.6)
plt.plot(baseline_predictions, label='Baseline Advanced Regressor Predictions', color='red', alpha=0.6)
plt.xticks(ticks=np.arange(len(patient_ids)), labels=patient_ids, rotation=90, fontsize=8, alpha=0.6)
plt.title('Actual ICP vs Baseline Advanced Regressor Predictions')
plt.xlabel('Patient IDs')
plt.ylabel('ICP')
plt.legend()
plt.show()

# Plot 2: Actual ICP vs Linear Regression Predictions
plt.figure(figsize=(14, 7))
plt.plot(actual_icp, label='Actual ICP', color='blue', alpha=0.6)
plt.plot(linear_predictions, label='Linear Regression Predictions', color='green', alpha=0.6)
plt.xticks(ticks=np.arange(len(patient_ids)), labels=patient_ids, rotation=90, fontsize=8, alpha=0.6)
plt.title('Actual ICP vs Linear Regression Predictions')
plt.xlabel('Patient IDs')
plt.ylabel('ICP')
plt.legend()
plt.show()
