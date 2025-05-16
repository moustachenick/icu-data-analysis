import pandas as pd
from pathlib import Path
import os


class BinaryDataProcessor:
    """
    A class to process intracranial pressure (ICP) data and convert it into binary format.
    """

    def __init__(self, data=None, file_path=None):
        self.data = data
        self.file_path = file_path

        if self.data is None and self.file_path is not None:
            self.load_data()

    def load_data(self):
        """
        Load the CSV data into a pandas DataFrame if a file path is provided.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data successfully loaded from {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def create_binary_data(self, data=None):
        """
        Convert the 'icp' column into binary values (0 if < 22, 1 if >= 22).
        """
        if data is not None:
            self.data = data

        # Create a new binary column
        self.data['icp_binary'] = self.data['icp'].apply(lambda x: 1 if x >= 22 else 0)
        # remove the original 'icp' column, since it is no longer needed
        self.data.drop(columns=['icp'], inplace=True)
        return self.data


# --- TEST SECTION ---
if __name__ == "__main__":
    # Define the file path dynamically using os.path
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_df_lagged.csv")
    )
    print(f"File path resolved to: {file_path}")

    # Initialize the processor and load the data
    processor = BinaryDataProcessor(file_path=Path(file_path))  # Convert to Path object
    processor.load_data()

    # Create the binary column
    processor.create_binary_data()

    # Verify the result
    print(processor.data[['icp', 'icp_binary']].head())

    # (Optional) Save the updated DataFrame
    output_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "cleaned_df_lagged_binary.csv")
    )
    processor.data.to_csv(output_file, index=False)
    print(f"Updated DataFrame saved to {output_file}")
