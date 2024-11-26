from tabulate import tabulate

class DataFramePrinter:

    @staticmethod
    def print_dataframe_tabulated(df, title=None):
        """
        Print a pandas DataFrame using the tabulate library for a pretty table format.

        Args:
            df (DataFrame): The DataFrame to print.
            title (str): Optional title to print above the table.
        """
        if title:
            print(f"\n{title}\n" + "=" * len(title))
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))