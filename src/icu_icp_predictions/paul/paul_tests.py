import sys
import os

from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from icu_data_regression_classes.BaselineMeanRegression import BaselineMeanRegression
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_regression_classes.BaselineHistoryRegression import BaselineHistoryRegression
from icu_data_regression_classes.RegressionModelPlotter import RegressionModelPlotter
from icu_data_regression_classes.TimeWindowMeanICPRegression import TimeWindowMeanICPRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'final_data.csv'))

data_processor = DataPreProcessor(file_path)

cleaned_df = data_processor.pre_process_dataset()

# We can now use the Regressor to predict the ICP values for the rest of the data.

# Prepare the features and target variable
X = cleaned_df.drop(columns=['icp', 'icp_next'])  # Features
y = cleaned_df['icp_next']  # Target is the "next ICP" at the next valid timestamp

# Let's use only the first 50% of the data, for speed (this will be removed in the final version)
X_original = X
X = X[:int(0.5 * X.shape[0])]
y = y[:int(0.5 * y.shape[0])]

# we need to create a train-test split of the data
# we will use the first 80% of the data for training and the rest for testing

train_size = int(0.8 * X.shape[0])

print('\nTotal size: ', X.shape[0], '\tTrain size:', train_size, '\tTest size:', X.shape[0] - train_size,
      '\tPercentage of the original dataset:', round((train_size / X_original.shape[0]) * 100), '%')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=X.shape[0] - train_size, random_state=42)

# Initialize and train the Baseline Advanced Regression model
baseline_mean_regression_model = BaselineMeanRegression()
baseline_mean_regression_model.fit(X_train, y_train)

# Make predictions with the Baseline Advanced Regressor
baseline_predictions = baseline_mean_regression_model.predict(X_test)

# Initialize and train the Baseline History Regressor
history_regressor = BaselineHistoryRegression()
history_regressor.fit(X_train, y_train)

# Make predictions with the Baseline History Regressor
history_predictions = history_regressor.predict(X_test)

# Time Window Mean ICP Regression Model

print('\n\nTime Window Mean ICP Regression Model:\n')

# Define a dictionary to store the performance results for each time window
resultsPerTimeWindow = {}

# Loop over different time windows (e.g., 1 day, 2 days, 3 days, 4 days)
for days in [1, 2, 3, 4, 5, 6, 7]:
    # Initialize the model with the current time window
    model = TimeWindowMeanICPRegression(days_window=days)
    
    # Fit the model with the training data
    model.fit(X_train, y_train)
    
    # Predict ICP on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics (e.g., MSE, MAE)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Store the results for this time window
    resultsPerTimeWindow[days] = {
        'MSE': mse,
        'MAE': mae
    }

# Print the results for each time window
for days, results in resultsPerTimeWindow.items():
    print(f"Time Window: {days} days - MSE: {results['MSE']:.2f}, MAE: {results['MAE']:.2f}")

# Keep the best model based on the lowest MSE
best_time_window = min(resultsPerTimeWindow, key=lambda x: resultsPerTimeWindow[x]['MSE'])
print(f"Best Time Window: {best_time_window} days")
time_window_regression_model = TimeWindowMeanICPRegression(days_window=best_time_window)
time_window_regression_model.fit(X_train, y_train)
time_window_predictions = time_window_regression_model.predict(X_test)

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
time_window_predictions_mse = mean_squared_error(y_test, time_window_predictions)

# Evaluate all models
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)
history_mae = mean_absolute_error(y_test, history_predictions)
ridge_regression_mae = mean_absolute_error(y_test, ridge_regression_predictions)
time_window_predictions_mae = mean_absolute_error(y_test, time_window_predictions)

# Evaluate all models
baseline_rmse = root_mean_squared_error(y_test, baseline_predictions)
linear_rmse = root_mean_squared_error(y_test, linear_predictions)
history_rmse = root_mean_squared_error(y_test, history_predictions)
ridge_regression_rmse = root_mean_squared_error(y_test, ridge_regression_predictions)
time_window_predictions_rmse = root_mean_squared_error(y_test, time_window_predictions)


# Calculate the residuals
history_residuals = y_test.values - history_predictions
advanced_residuals = y_test.values - baseline_predictions
linear_residuals = y_test.values - linear_predictions
ridge_residuals = y_test.values - ridge_regression_predictions
time_window_residuals = y_test.values - time_window_predictions

# Plotting
# Define the x values for the reference line
x = np.linspace(min(y_test), max(y_test), 400)
y_ref = x
# Plot the lines for all models

# Dictionary of model predictions
predictions_dict = {
    'Linear Regression': linear_predictions,
    f'Time Window Mean ICP Regression ({best_time_window} days)': time_window_predictions
}

# Plot the results for all models
RegressionModelPlotter.plot_regression_models(y_test, predictions_dict)

# Create a DataFrame with the predictions and actual values
results_df = pd.DataFrame({
    'Actual Values': y_test.values,
    'BaselineHistoryRegressionModel': history_predictions,
    'BaselineAdvancedRegressorModel': baseline_predictions,
    'LinearRegressionModel': linear_predictions,
    'RidgeRegressionModel': ridge_regression_predictions,
    f'TimeWindowMeanICPRegressionModel ({best_time_window} days)': time_window_predictions
})

# Perform 10-fold cross-validation for all models
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Prepare data for Linear Regression model separately
X_lr = X.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Initialize lists to store cross-validation scores
linear_cv_scores = cross_val_score(linear_regression, X_lr, y, cv=kf, scoring='neg_mean_squared_error')
linear_cv_scores = -linear_cv_scores

# Now we want to compare the cross-validation results of the Linear Regression model
# with the Baseline Advanced Regressor and the Baseline History Regressor.

# Perform 10-fold cross-validation for the Baseline Advanced Regressor
advanced_cv_scores = cross_val_score(baseline_mean_regression_model, X, y, cv=kf, scoring='neg_mean_squared_error')
advanced_cv_scores = -advanced_cv_scores

# Perform 10-fold cross-validation for the Baseline History Regressor
history_cv_scores = cross_val_score(history_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')
history_cv_scores = -history_cv_scores

# Perform 10-fold cross-validation for the Ridge Regression model
ridge_cv_scores = cross_val_score(ridge_regression, X_lr, y, cv=kf, scoring='neg_mean_squared_error')
ridge_cv_scores = -ridge_cv_scores

# Perform 10-fold cross-validation for the Time Window Mean ICP Regression model
time_window_cv_scores = cross_val_score(time_window_regression_model, X, y, cv=kf, scoring='neg_mean_squared_error')
time_window_cv_scores = -time_window_cv_scores

# Add the cross-validation results to the DataFrame
cv_results_df = pd.DataFrame({
    'LinearRegressionModel': linear_cv_scores
})
cv_results_df['BaselineAdvancedRegressorModel'] = advanced_cv_scores
cv_results_df['BaselineHistoryRegressionModel'] = history_cv_scores
cv_results_df['RidgeRegressionModel'] = ridge_cv_scores
cv_results_df[f'TimeWindowMeanICPRegressionModel ({best_time_window} days)'] = time_window_cv_scores

# Plot the cross-validation results (mean squared error) for all models

plt.figure(figsize=(14, 8))
plt.plot(linear_cv_scores, label='Linear Regression')
plt.plot(advanced_cv_scores, label='Baseline Advanced Regressor')
plt.plot(history_cv_scores, label='Baseline History Regressor')
plt.plot(ridge_cv_scores, label='Ridge Regression')
plt.plot(time_window_cv_scores, label=f'Time Window Mean ICP Regression ({best_time_window} days)')
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend()
plt.show()

# Display the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) for all models, as a table

results = {
    'Model': ['Baseline Advanced Regressor', 'Baseline History Regressor', 'Linear Regression', 'Ridge Regression',
                f'Time Window Mean ICP Regression ({best_time_window} days)'],
    'MSE': [baseline_mse, history_mse, linear_mse, ridge_regression_mse, time_window_predictions_mse],
    'MAE': [baseline_mae, history_mae, linear_mae, ridge_regression_mae, time_window_predictions_mae],
    'RMSE': [baseline_rmse, history_rmse, linear_rmse, ridge_regression_rmse, time_window_predictions_rmse]
}

results_df = pd.DataFrame(results)
print(results_df)

# From the cleaned_df, we want to check the importance of the features in predicting the ICP values.
# Let's use the Lasso Regression model for this purpose.

# Remove the 'timestamp', 'patient_id', and 'date_of_birth' columns as they are not numeric and not useful for the model
X_train_lasso = X_train.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_test_lasso = X_test.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])
X_lasso = X.drop(columns=['timestamp', 'patient_id', 'date_of_birth'])

# Scale the features (important for Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lasso)
X_test_scaled = scaler.transform(X_test_lasso)

# Initialize Lasso regression model with a chosen alpha value
lasso = Lasso(alpha=0.1)  # You can experiment with different alpha values

# Fit the model
lasso.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = lasso.predict(X_train_scaled)
y_pred_test = lasso.predict(X_test_scaled)

# Get the coefficients from the Lasso model
coefficients = pd.Series(lasso.coef_, index=X_lasso.columns)

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
X_lr_important = X_lr[important_features]

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

linear_cv_scores_important = cross_val_score(linear_regression_important, X_lr_important, y, cv=kf, scoring='neg_mean_squared_error')

# Convert scores to positive values
linear_cv_scores_important = -linear_cv_scores_important

print('\nCross-Validation Results:\n')
print(f'Cross-validation MSE for Linear Model 1 (all columns): {linear_cv_scores.mean()}')
print(f'Cross-validation MSE for Linear Model 2 (important columns): {linear_cv_scores_important.mean()}')

print(f'Number of features in Model 1 (all columns): {linear_regression.coef_.shape[0]}')
print(f'Number of features in Model 2 (important columns): {linear_regression_important.coef_.shape[0]}')