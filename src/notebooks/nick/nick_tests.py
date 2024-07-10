import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

<<<<<<< HEAD

import pandas as pd
=======
>>>>>>> b6ad8bd9b86e8c0d2dfe962471dba1cdcd6ee9df
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from icu_data_regression_models.BaselineAdvancedRegressor import BaselineAdvancedRegressor
from icu_data_parser.DataPreProcessor import DataPreProcessor



data_processor = DataPreProcessor()

filtered_df = data_processor.replace_missing_values()

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
