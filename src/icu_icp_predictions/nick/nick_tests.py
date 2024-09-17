import sys
import os

from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from icu_data_regression_models.BaselineMeanRegressionModel import BaselineMeanRegressionModel
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_regression_models.BaselineHistoryRegressionModel import BaselineHistoryRegressionModel
from icu_data_regression_models.RegressionModelPlotter import RegressionModelPlotter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV


file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'final_data.csv'))

data_processor = DataPreProcessor(file_path)

cleaned_df = data_processor.pre_process_dataset()

# Split the data into features (X) and target (y)
X = cleaned_df.drop(columns=['icp'])
y = cleaned_df['icp']

# Let's use only the first 20% of the data, for speed (TODO: remove in the final version)
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
baseline_mean_regression_model = BaselineMeanRegressionModel()
baseline_mean_regression_model.fit(X_train, y_train)

# Make predictions with the Baseline Advanced Regressor
baseline_predictions = baseline_mean_regression_model.predict(X_test)

# Initialize and train the Baseline History Regressor
history_regressor = BaselineHistoryRegressionModel()
history_regressor.fit(X_train, y_train)

# Make predictions with the Baseline History Regressor
history_predictions = history_regressor.predict(X_test)

# For the LinearRegression model, we need to drop the 'timestamp', 'patient_id', and 'date_of_birth' columns
X_train_lr = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test_lr = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_cleaned = X.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Scale the features (especially important for features on different scales)
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train_lr)
X_test_lr = scaler.transform(X_test_lr)


# Initialize and train the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train_lr, y_train)

# Make predictions with the Linear Regression model
linear_predictions = linear_regression.predict(X_test_lr)

# Get the coefficients
coefficients = pd.Series(linear_regression.coef_, index=X_cleaned.columns)

# Print the coefficients and their importance
print("Linear Regression Coefficients:")
print(coefficients)

# Let's also try another model from the sklearn library, the Ridge Regression model

# Initialize and train the Ridge Regression model
ridge_regression = Ridge()
ridge_regression.fit(X_train_lr, y_train)
ridge_regression_predictions = ridge_regression.predict(X_test_lr)

# Evaluate all models
baseline_mse = mean_squared_error(y_test, baseline_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
history_mse = mean_squared_error(y_test, history_predictions)
ridge_regression_mse = mean_squared_error(y_test, ridge_regression_predictions)

# Evaluate all models
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)
history_mae = mean_absolute_error(y_test, history_predictions)
ridge_regression_mae = mean_absolute_error(y_test, ridge_regression_predictions)

# Evaluate all models
baseline_rmse = root_mean_squared_error(y_test, baseline_predictions)
linear_rmse = root_mean_squared_error(y_test, linear_predictions)
history_rmse = root_mean_squared_error(y_test, history_predictions)
ridge_regression_rmse = root_mean_squared_error(y_test, ridge_regression_predictions)



print('\n\nMean Squared Error Results:')

print('Baseline Advanced Regression Mean Squared Error:', baseline_mse)
print('Linear Regression Mean Squared Error:', linear_mse)
print('Baseline History Regression Mean Squared Error', history_mse)
print('Ridge Regression Mean Squared Error:', ridge_regression_mse)

print('\n\nMean Absolute Error Results:')

print('Baseline Advanced Regression Mean Absolute Error:', baseline_mae)
print('Linear Regression Mean Absolute Error:', linear_mae)
print('Baseline History Regression Mean Absolute Error', history_mae)
print('Ridge Regression Mean Absolute Error:', ridge_regression_mae)

print('\n\nRoot Mean Squared Error Results:')

print('Baseline Advanced Regression Root Mean Squared Error:', baseline_rmse)
print('Linear Regression Root Mean Squared Error:', linear_rmse)
print('Baseline History Regression Root Mean Squared Error', history_rmse)
print('Ridge Regression Root Mean Squared Error:', ridge_regression_rmse)


# Calculate the residuals
history_residuals = y_test.values - history_predictions
advanced_residuals = y_test.values - baseline_predictions
linear_residuals = y_test.values - linear_predictions

# Perform paired samples t-test
t_statistic_ha, p_value_ha = ttest_rel(history_residuals, advanced_residuals)
t_statistic_hl, p_value_hl = ttest_rel(history_residuals, linear_residuals)
t_statistic_al, p_value_al = ttest_rel(advanced_residuals, linear_residuals)

print("\n\nPaired Samples T-Test Results:")
print(f"History vs Advanced: t-statistic = {t_statistic_ha:.3f}, p-value = {p_value_ha:.3f}")
print(f"History vs Linear: t-statistic = {t_statistic_hl:.3f}, p-value = {p_value_hl:.3f}")
print(f"Advanced vs Linear: t-statistic = {t_statistic_al:.3f}, p-value = {p_value_al:.3f}")
print("\n\n")

# Plotting
# Define the x values for the reference line
x = np.linspace(min(y_test), max(y_test), 400)
y_ref = x

# # Plot the results for Linear Regression
# plt.figure(figsize=(14, 8))
# plt.scatter(y_test, linear_predictions, label='Linear Predictions')
# plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
# plt.axhline(0, color='black', linewidth=1)
# plt.axvline(0, color='black', linewidth=1)
# plt.xlabel('Actual Values', fontsize=14)
# plt.ylabel('Predictions', fontsize=14)
# plt.legend()
# plt.show()

# # Plot the results for Baseline Advanced Regressor
# plt.figure(figsize=(14, 8))
# plt.scatter(y_test, baseline_predictions, label='Baseline Advanced Predictions')
# plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
# plt.axhline(0, color='black', linewidth=1)
# plt.axvline(0, color='black', linewidth=1)
# plt.xlabel('Actual Values', fontsize=14)
# plt.ylabel('Predictions', fontsize=14)
# plt.legend()
# plt.show()

# # Plot the results for Baseline History Regressor
# plt.figure(figsize=(14, 8))
# plt.scatter(y_test, history_predictions, label='Baseline History Predictions')
# plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
# plt.axhline(0, color='black', linewidth=1)
# plt.axvline(0, color='black', linewidth=1)
# plt.xlabel('Actual Values', fontsize=14)
# plt.ylabel('Predictions', fontsize=14)
# plt.legend()
# plt.show()

# Plot the lines for all models

# Dictionary of model predictions
predictions_dict = {
    'Linear Regression': linear_predictions,
    'Baseline Advanced Regressor': baseline_predictions,
    'Baseline History Regressor': history_predictions
}

# Plot the results for all models
RegressionModelPlotter.plot_regression_models(y_test, predictions_dict)

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
linear_cv_scores = cross_val_score(linear_regression, X_lr, y, cv=kf, scoring='neg_mean_squared_error')

print('\n\nLinear CV Scores:', linear_cv_scores)

# Convert scores to positive values
linear_cv_scores = -linear_cv_scores

# Display cross-validation results
cv_results_df = pd.DataFrame({
    'LinearRegressionModel': linear_cv_scores
})

# Now we want to compare the cross-validation results of the Linear Regression model
# with the Baseline Advanced Regressor and the Baseline History Regressor.

# Perform 10-fold cross-validation for the Baseline Advanced Regressor
advanced_cv_scores = cross_val_score(baseline_mean_regression_model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert scores to positive values
advanced_cv_scores = -advanced_cv_scores

print('\n\nAdvanced CV Scores:', advanced_cv_scores)

# Perform 10-fold cross-validation for the Baseline History Regressor

history_cv_scores = cross_val_score(history_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert scores to positive values
history_cv_scores = -history_cv_scores

print('\n\nHistory CV Scores:', history_cv_scores)

# Add the cross-validation results to the DataFrame

cv_results_df['BaselineAdvancedRegressorModel'] = advanced_cv_scores
cv_results_df['BaselineHistoryRegressionModel'] = history_cv_scores

# Display the cross-validation results

print('\nCross-Validation Results:')
print('Linear Regression 10-fold CV Mean Squared Error:', np.mean(linear_cv_scores))
print('Baseline Advanced Regressor 10-fold CV Mean Squared Error:', np.mean(advanced_cv_scores))
print('Baseline History Regressor 10-fold CV Mean Squared Error:', np.mean(history_cv_scores))
print('\n\n')

print('\nCross-Validation Results:')
print('Linear Regression 10-fold CV Mean Absolute Error:', np.absolute(linear_cv_scores))
print('Baseline Advanced Regressor 10-fold CV Mean Absolute Error:', np.absolute(advanced_cv_scores))
print('Baseline History Regressor 10-fold CV Mean Absolute Error:', np.absolute(history_cv_scores))
print('\n\n')

# Plot the cross-validation results (mean squared error) for all models

plt.figure(figsize=(14, 8))
plt.plot(linear_cv_scores, label='Linear Regression')
plt.plot(advanced_cv_scores, label='Baseline Advanced Regressor')
plt.plot(history_cv_scores, label='Baseline History Regressor')
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend()
plt.show()


# From the cleaned_df, we want to check the importance of the features in predicting the ICP values.
# Let's use the Lasso Regression model for this purpose.

# Remove the 'timestamp', 'patient_id', and 'date_of_birth' columns as they are not numeric and not useful for the model
X_train = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_cleaned = X.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Scale the features (important for Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Lasso regression model with a chosen alpha value
lasso = Lasso(alpha=0.1)  # You can experiment with different alpha values

# Fit the model
lasso.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = lasso.predict(X_train_scaled)
y_pred_test = lasso.predict(X_test_scaled)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MSE: {mse_train}")
print(f"Test MSE: {mse_test}")
print(f"Training R2 Score: {r2_train}")
print(f"Test R2 Score: {r2_test}")

# Get the coefficients from the Lasso model
coefficients = pd.Series(lasso.coef_, index=X_cleaned.columns)

# Display coefficients
print("Lasso Coefficients:")
print(coefficients)

# Identify important features
important_features = coefficients[coefficients != 0].index.tolist()
print("Important features selected by Lasso:", important_features)

# Now let's try to keep only the important features and see if the Linear model performance improves

# Prepare training and testing sets with only the important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Scale the features
scaler = StandardScaler()
X_train_important_scaled = scaler.fit_transform(X_train_important)
X_test_important_scaled = scaler.transform(X_test_important)

# Use Linear Regression model with only the important features
linear_regression_important = LinearRegression()
linear_regression_important.fit(X_train_important_scaled, y_train)

# Make predictions
linear_predictions_important = linear_regression_important.predict(X_test_important_scaled)

# Perform 10-fold cross-validation for the Linear Regression model with only the important features

linear_cv_scores_important = cross_val_score(linear_regression_important, X_lr, y, cv=kf, scoring='neg_mean_squared_error')

print('\n\nLinear CV Scores:', linear_cv_scores_important)

# Convert scores to positive values
linear_cv_scores_important = -linear_cv_scores_important

print('\nCross-Validation Results:')
print('Linear Regression 10-fold CV Mean Absolute Error (only important attributes):', np.absolute(linear_cv_scores_important))

# Evaluate the model
mse_important = mean_squared_error(y_test, linear_predictions_important)
linear_mae_important = mean_absolute_error(y_test, linear_predictions_important)

print(f"Linear Regression with Important Features Test MSE: {mse_important}")

# Compare the results with the original Linear Regression model
print('\n\nMean Squared Error Results (Compared):')
print('Linear Regression Mean Squared Error:', linear_mse)
print('Linear Regression with Important Features Mean Squared Error:', mse_important)

print('\n\nRoot Mean Squared Error Results (Compared):')
print('Linear Regression Root Mean Squared Error:', linear_rmse)
print('Linear Regression with Important Features Root Mean Squared Error:', root_mean_squared_error(y_test, linear_predictions_important))

print('\n\nAbsolute Error Results (Compared):')
print('Linear Regression Absolute Error:', linear_mae)
print('Linear Regression with Important Features Absolute Error:', linear_mae_important)
