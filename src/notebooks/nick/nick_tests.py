import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from icu_data_regression_models.BaselineAdvancedRegressor import BaselineAdvancedRegressor
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_regression_models.BaselineHistoryRegressionModel import BaselineHistoryRegressionModel

data_processor = DataPreProcessor()

filtered_df = data_processor.replace_missing_values()

combined_df = data_processor.known_nearest_neighbor_imputer(filtered_df)

# Split the data into features (X) and target (y)
X = combined_df.drop(columns=['icp'])
y = combined_df['icp']

# Let's use only the first 20% of the data, for speed (this will be removed in the final version)
X_original = X
X = X[:int(0.2 * X.shape[0])] 
y = y[:int(0.2 * y.shape[0])]

# Split the data into training and testing sets
train_size = int(0.8 * X.shape[0])

# Prepare training and testing sets
X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# Initialize and train the Baseline Advanced Regressor
baseline_regressor = BaselineAdvancedRegressor()
baseline_regressor.fit(X_train, y_train)

# Make predictions with the Baseline Advanced Regressor
baseline_predictions = baseline_regressor.predict(X_test)

#Initialize and train the Baseline History Regressor
history_regressor = BaselineHistoryRegressionModel()
history_regressor.fit(X_train, y_train)

# Make predictions with the Baseline History Regressor
history_predictions = history_regressor.predict(X_test)

# For the LinearRegression model, we need to drop the 'timestamp', 'patient_id', and 'date_of_birth' columns
X_train_lr = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test_lr = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Initialize and train the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_lr, y_train)

# Make predictions with the Linear Regression model
linear_predictions = linear_regressor.predict(X_test_lr)

# Evaluate both models
baseline_mse = mean_squared_error(y_test, baseline_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
history_mse = mean_squared_error(y_test, history_predictions)

print('Baseline Advanced Regressor Mean Squared Error:', baseline_mse)
print('Linear Regression Mean Squared Error:', linear_mse)
print('Baseline History Regression Mean Squared Error', history_mse)

# Calculate the residuals
history_residuals = y_test.values - history_predictions
advanced_residuals = y_test.values - baseline_predictions
linear_residuals = y_test.values - linear_predictions

# Perform paired samples t-test
t_statistic_ha, p_value_ha = ttest_rel(history_residuals, advanced_residuals)
t_statistic_hl, p_value_hl = ttest_rel(history_residuals, linear_residuals)
t_statistic_al, p_value_al = ttest_rel(advanced_residuals, linear_residuals)

print("Paired Samples T-Test Results:")
print(f"History vs Advanced: t-statistic = {t_statistic_ha:.3f}, p-value = {p_value_ha:.3f}")
print(f"History vs Linear: t-statistic = {t_statistic_hl:.3f}, p-value = {p_value_hl:.3f}")
print(f"Advanced vs Linear: t-statistic = {t_statistic_al:.3f}, p-value = {p_value_al:.3f}")

# Plotting
# Define the x values for the reference line
x = np.linspace(min(y_test), max(y_test), 400)
y_ref = x

# Plot the results for Linear Regression
plt.figure(figsize=(14, 8))
plt.scatter(y_test, linear_predictions, label='Linear Predictions')
plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.legend()
plt.show()

# Plot the results for Baseline Advanced Regressor
plt.figure(figsize=(14, 8))
plt.scatter(y_test, baseline_predictions, label='Baseline Advanced Predictions')
plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.legend()
plt.show()

# Plot the results for Baseline History Regressor
plt.figure(figsize=(14, 8))
plt.scatter(y_test, history_predictions, label='Baseline History Predictions')
plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.legend()
plt.show()

# Create a DataFrame with the predictions and actual values
results_df = pd.DataFrame({
    'Actual Values': y_test.values,
    'BaselineHistoryRegressionModel': history_predictions,
    'BaselineAdvancedRegressorModel': baseline_predictions,
    'LinearRegressionModel': linear_predictions
})

# Perform 10-fold cross-validation for all models
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Prepare data for Linear Regression model separately
X_lr = X.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Initialize lists to store cross-validation scores
linear_cv_scores = cross_val_score(linear_regressor, X_lr, y, cv=kf, scoring='neg_mean_squared_error')

# Convert scores to positive values
linear_cv_scores = -linear_cv_scores

print('Linear Regression 10-fold CV Mean Squared Error:', np.mean(linear_cv_scores))

# Display cross-validation results
cv_results_df = pd.DataFrame({
    'LinearRegressionModel': linear_cv_scores
})

