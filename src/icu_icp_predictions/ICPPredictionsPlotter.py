import numpy as np
from matplotlib import pyplot as plt

class ICPPredictionsPlotter:

    def plot_model_results(self, y_test, predictions_dict):
        """
        Plot the results of different regression models to compare their predictions.

        Args:
            y_test: Actual target values.
            predictions_dict: Dictionary containing model names as keys and their predictions as values.
        """
        print("\nPlotting model results...")
        plt.figure(figsize=(12, 8))

        # Plot the actual values
        plt.plot(y_test.values, label="Actual Values", color="black", linewidth=2)

        # Plot predictions for each model
        for model_name, predictions in predictions_dict.items():
            plt.plot(predictions, label=model_name)

        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel("ICP Value", fontsize=14)
        plt.title("Comparison of Regression Model Predictions", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cross_validation_results(self, cv_results_df):
        """
        Plot cross-validation results for all models to compare their mean CV MSE.

        Args:
            cv_results_df: DataFrame containing cross-validation results with columns:
                - "Model": Model names.
                - "Mean CV MSE": Mean cross-validation MSE for each model.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(cv_results_df["Model"], cv_results_df["Mean CV MSE"], color='skyblue')
        plt.xlabel("Models", fontsize=14)
        plt.ylabel("Mean CV MSE", fontsize=14)
        plt.title("Cross-Validation Mean Squared Error Comparison", fontsize=16)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_test, predictions_dict):
        """
        Plot residuals for each regression model.

        Args:
            y_test: Actual target values.
            predictions_dict: Dictionary containing model names as keys and their predictions as values.
        """
        print("\nPlotting residuals...")
        plt.figure(figsize=(12, 8))

        for model_name, predictions in predictions_dict.items():
            residuals = y_test.values - predictions
            plt.scatter(range(len(residuals)), residuals, alpha=0.6, label=f"{model_name} Residuals")

        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel("Residual", fontsize=14)
        plt.title("Residuals of Regression Models", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_prediction_error(self, y_test, predictions_dict):
        """
        Plot prediction error for each regression model.

        Args:
            y_test: Actual target values.
            predictions_dict: Dictionary containing model names as keys and their predictions as values.
        """
        print("\nPlotting prediction error...")
        plt.figure(figsize=(12, 8))

        # Plot a perfect prediction line
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect Prediction", linewidth=2)

        for model_name, predictions in predictions_dict.items():
            plt.scatter(y_test, predictions, alpha=0.6, label=model_name)

        plt.xlabel("Actual Values", fontsize=14)
        plt.ylabel("Predicted Values", fontsize=14)
        plt.title("Prediction Error Plot", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_performance_metrics(self, results_df):
        """
        Plot performance metrics (MSE, MAE, RMSE) for each model.

        Args:
            results_df: DataFrame containing evaluation metrics for all models.
        """
        print("\nPlotting performance metrics...")
        metrics = ["MSE", "MAE", "RMSE"]
        x = np.arange(len(results_df["Model"]))  # Label locations

        plt.figure(figsize=(14, 8))

        for i, metric in enumerate(metrics):
            plt.bar(x + i * 0.25, results_df[metric], width=0.25, label=metric)

        plt.xticks(x + 0.25, results_df["Model"], rotation=45)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Metric Value", fontsize=14)
        plt.title("Performance Metrics for Regression Models", fontsize=16)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
