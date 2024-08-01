import matplotlib.pyplot as plt
import numpy as np
import itertools


class RegressionModelPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_regression_models(y_test, predictions_dict):
        """
        Plots the actual vs predicted values for multiple regression models.

        :param y_test: Actual values
        :param predictions_dict: Dictionary where keys are model names and values are predictions
        """
        x = np.linspace(min(y_test), max(y_test), 400)
        y_ref = x

        plt.figure(figsize=(14, 8))

        # Define a color cycle
        colors = itertools.cycle(plt.cm.tab10.colors)

        for model_name, predictions in predictions_dict.items():
            color = next(colors)
            plt.scatter(y_test, predictions, label=f'{model_name} Predictions', color=color)

        plt.plot(x, y_ref, color="black", linewidth=1, label='Perfect Prediction')
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predictions', fontsize=14)
        plt.legend()
        plt.show()
