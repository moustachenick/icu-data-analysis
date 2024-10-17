import sys
import os

from matplotlib import pyplot as plt

from icu_data_parser.TimeSeriesProcessor import TimeSeriesProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from icu_data_regression_classes.BaselineMeanRegression import BaselineMeanRegression
from icu_data_parser.DataPreProcessor import DataPreProcessor
from icu_data_regression_classes.BaselineHistoryRegression import (
    BaselineHistoryRegression,
)
from icu_data_regression_classes.RegressionModelPlotter import RegressionModelPlotter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "final_data.csv")
)

data_processor = DataPreProcessor(file_path)

cleaned_df = data_processor.pre_process_dataset()

time_series_processor = TimeSeriesProcessor()

# define the number of lags and columns to lag
lags = 5
columns_to_lag = ["cpp", "glucose", "haemoglobin", "icp", "heart_rate", "temperature", "mean_blood_pressure", "paco2", "pao2", "peep", "ph", "spo2"]

# Process the data by creating lag features
cleaned_df = time_series_processor.process_data(cleaned_df, lags=lags, columns_to_lag=columns_to_lag)

print(f"\nDataset pre-processed and lag features created.")
print(f"\nNumber of rows in the cleaned dataset: {cleaned_df.shape[0]}")

# print the number of rows with NaN or missing values
print("\nNumber of rows with NaN values: ", cleaned_df.isnull().sum().sum())

# We can now use the Regression to predict the ICP values for the rest of the data.

# Prepare the features and target variable
X = cleaned_df.drop(columns=["icp", "icp_next"])  # Features
y = cleaned_df["icp_next"]  # Target is the "next ICP" at the next valid timestamp

# Let's use only the first 50% of the data, for speed (TODO this will be removed in the final version)
X_original = X
# X = X[: int(0.5 * X.shape[0])]
# y = y[: int(0.5 * y.shape[0])]

# we need to create a train-test split of the data
# we will use the first 80% of the data for training and the rest for testing

train_size = int(0.8 * X.shape[0])

print(
    "\nTotal size: ",
    X.shape[0],
    "\tTrain size:",
    train_size,
    "\tTest size:",
    X.shape[0] - train_size,
    "\tPercentage of the original dataset:",
    round((train_size / X_original.shape[0]) * 100),
    "%",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=X.shape[0] - train_size, random_state=42
)

# Initialize and train the Baseline Advanced Regression model
baseline_mean_regression_model = BaselineMeanRegression()
baseline_mean_regression_model.fit(X_train, y_train)

# Make predictions with the Baseline Advanced Regression
baseline_predictions = baseline_mean_regression_model.predict(X_test)

# Initialize and train the Baseline History Regression
history_regression = BaselineHistoryRegression()
history_regression.fit(X_train, y_train)

# Make predictions with the Baseline History Regression
history_predictions = history_regression.predict(X_test)

# For the LinearRegression model, we need to drop the 'timestamp', 'patient_id', and 'date_of_birth' columns
X_train_lr = X_train.drop(columns=["timestamp", "patient_id", "date_of_birth"])
X_test_lr = X_test.drop(columns=["timestamp", "patient_id", "date_of_birth"])
X_cleaned = X.drop(columns=["timestamp", "patient_id", "date_of_birth"])

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
print("Linear Regression Coefficients, ordered by importance:")
print(coefficients.abs().sort_values(ascending=False))

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

# Calculate the residuals
history_residuals = y_test.values - history_predictions
advanced_residuals = y_test.values - baseline_predictions
linear_residuals = y_test.values - linear_predictions
ridge_residuals = y_test.values - ridge_regression_predictions

# get the accuracy of all models
accuracy_linear = linear_regression.score(X_test_lr, y_test)
accuracy_ridge = ridge_regression.score(X_test_lr, y_test)
accuracy_mean_baseline = baseline_mean_regression_model.score(X_test, y_test)
accuracy_history = history_regression.score(X_test, y_test)

# Plotting
# Define the x values for the reference line
x = np.linspace(min(y_test), max(y_test), 400)
# Plot the lines for all models

# Dictionary of model predictions
predictions_dict = {
    "Linear Regression": linear_predictions,
    "Baseline Advanced Regression": baseline_predictions,
    "Baseline History Regression": history_predictions,
    "Ridge Regression": ridge_regression_predictions,
}

# Plot the results for all models
RegressionModelPlotter.plot_regression_models(y_test, predictions_dict)

# Create a DataFrame with the predictions and actual values
results_df = pd.DataFrame(
    {
        "Actual Values": y_test.values,
        "BaselineHistoryRegression": history_predictions,
        "BaselineAdvancedRegression": baseline_predictions,
        "LinearRegressionModel": linear_predictions,
        "RidgeRegression": ridge_regression_predictions,
    }
)

# Perform 10-fold cross-validation for all models
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Prepare data for Linear Regression model separately
X_lr = X.drop(columns=["timestamp", "patient_id", "date_of_birth"])
# scale the features for the Linear Regression model
scaler = StandardScaler()
X_lr = scaler.fit_transform(X_lr)

# Initialize lists to store cross-validation scores
linear_cv_scores = cross_val_score(
    linear_regression, X_lr, y, cv=kf, scoring="neg_mean_squared_error"
)
linear_cv_scores = -linear_cv_scores

print("\nLinear Regression Model Cross-Validation MSE:", linear_cv_scores.mean())

# Now we want to compare the cross-validation results of the Linear Regression model
# with the Baseline Advanced Regression and the Baseline History Regression.

print("\nPerforming 10-fold cross-validation for all models.")
print("\nThis may take a few minutes...")

# Perform 10-fold cross-validation for the Baseline Advanced Regression
advanced_cv_scores = cross_val_score(
    baseline_mean_regression_model, X, y, cv=kf, scoring="neg_mean_squared_error"
)
advanced_cv_scores = -advanced_cv_scores

# Perform 10-fold cross-validation for the Baseline History Regression
history_cv_scores = cross_val_score(
    history_regression, X, y, cv=kf, scoring="neg_mean_squared_error"
)
history_cv_scores = -history_cv_scores

# Perform 10-fold cross-validation for the Ridge Regression model
ridge_cv_scores = cross_val_score(
    ridge_regression, X_lr, y, cv=kf, scoring="neg_mean_squared_error"
)
ridge_cv_scores = -ridge_cv_scores

# Add the cross-validation results to the DataFrame
cv_results_df = pd.DataFrame({"LinearRegressionModel": linear_cv_scores})
cv_results_df["BaselineAdvancedRegression"] = advanced_cv_scores
cv_results_df["BaselineHistoryRegression"] = history_cv_scores
cv_results_df["RidgeRegression"] = ridge_cv_scores

print(
    f"\n\nAccording to the cross-validation results, the model with the lowest MSE is: {cv_results_df.mean().idxmin()}\n"
)

# Plot the cross-validation results (mean squared error) for all models

plt.figure(figsize=(14, 8))
plt.plot(advanced_cv_scores, label="Baseline Advanced Regression")
plt.plot(linear_cv_scores, label="Linear Regression")
plt.plot(history_cv_scores, label="Baseline History Regression")
# plt.plot(ridge_cv_scores, label='Ridge Regression')
plt.xlabel("Fold", fontsize=14)
plt.ylabel("Mean Squared Error", fontsize=14)
plt.legend()
plt.show()

# Display the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) for all models, as a table

results = {
    "Model": [
        "Baseline Advanced Regression",
        "Baseline History Regression",
        "Linear Regression",
        "Ridge Regression",
    ],
    "MSE": [
        baseline_mse,
        history_mse,
        linear_mse,
        ridge_regression_mse,
    ],
    "MAE": [
        baseline_mae,
        history_mae,
        linear_mae,
        ridge_regression_mae,
    ],
    "RMSE": [
        baseline_rmse,
        history_rmse,
        linear_rmse,
        ridge_regression_rmse,
    ],
    "Accuracy": [
        f"{accuracy_mean_baseline * 100:.2f}%",
        f"{accuracy_history * 100:.2f}",
        f"{accuracy_linear * 100:.2f}%",
        f"{accuracy_ridge * 100:.2f}%",
    ],
}

algo_results_df = pd.DataFrame(results)
print(algo_results_df)

print(
    f'\nThe algorithm with the lowest MAE is: {algo_results_df["Model"][algo_results_df["MAE"].idxmin()]}'
)

# From the cleaned_df, we want to check the importance of the features in predicting the ICP values.
# Let's use the Lasso Regression model for this purpose.

# Remove the 'timestamp', 'patient_id', and 'date_of_birth' columns as they are not numeric and not useful for the model
X_train_lasso = X_train.drop(columns=["timestamp", "patient_id", "date_of_birth"])
X_test_lasso = X_test.drop(columns=["timestamp", "patient_id", "date_of_birth"])
X_lasso = X.drop(columns=["timestamp", "patient_id", "date_of_birth"])

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
print("\nLasso Coefficients, ordered by importance:")
print(coefficients.abs().sort_values(ascending=False))

# Identify important features, ordered by importance
important_features = coefficients.abs().sort_values(ascending=False).index.tolist()
print("Important features selected by Lasso:", important_features)
