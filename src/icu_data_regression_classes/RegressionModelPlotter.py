import matplotlib.pyplot as plt


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
        colors = ['red', 'green', 'orange', 'purple', 'brown']  # Add more colors if needed
        linestyles = ['-', '--', '-.', ':', '-']  # Different line styles

        plt.figure(figsize=(10, 6))

        plt.scatter(range(len(y_test)), y_test.values, label='Actual ICP', color='blue', marker='o')

        # Step 3: Plot line plot for each algorithm
        for i, (algo_name, y_pred) in enumerate(predictions_dict.items()):
            plt.plot(range(len(y_pred)), y_pred, 
            label=algo_name, 
            color=colors[i % len(colors)],  # Assign a color to each algorithm
            linestyle=linestyles[i % len(linestyles)],  # Assign a linestyle
            linewidth=2)  # Set line width for better visibility

        # Step 4: Add titles and labels
        plt.title('Actual vs Predicted ICP from Multiple Algorithms')
        plt.xlabel('Index')
        plt.ylabel('ICP')
        plt.legend()

        # Step 5: Display the plot
        plt.grid(True)
        plt.show(block=False)
