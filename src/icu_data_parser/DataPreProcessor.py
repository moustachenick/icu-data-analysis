import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Get the current working directory
current_dir = Path(os.getcwd())

# Assuming the notebook is in a subdirectory of the project root, adjust the path accordingly
# Verify this path is correct for your project structure
project_root = current_dir.parent.parent.parent

# Print the project root for verification
print(f"Project Root: {project_root}")

# Convert the path to an absolute path and append it to sys.path if not already present
project_root_abs = str(project_root.resolve())
if project_root_abs not in sys.path:
    sys.path.append(project_root_abs)



class DataPreProcessor:
      
      def print_percentages_of_rows_with_missing_values(self, dataframe):
        # Function to print the percentage of rows that have more than 1 column with missing values
        # and the percentage of rows that have exactly 1 column with missing values.
        
        total_num_of_rows = dataframe.shape[0]
        number_of_rows_with_more_than_1_missing_value = dataframe[dataframe.isna().sum(axis=1) > 1].shape[0]
        print('Percentage of rows that have more than 1 column with missing values: ',
              number_of_rows_with_more_than_1_missing_value / total_num_of_rows * 100)
        
        number_of_rows_with_1_missing_value = dataframe[dataframe.isna().sum(axis=1) == 1].shape[0]
        print('Percentage of rows that have exactly 1 column with missing value: ', 
              number_of_rows_with_1_missing_value / total_num_of_rows * 100) 

      def replace_missing_values(self):
            # Construct the absolute path
            file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'final_data.csv'))
            
            # Read the data from the CSV file
            df = pd.read_csv(file_path, engine='python')
            df.head()
            # The dataset contains missing values that are represented by different strings ("--", ".", etc).
            # declare an array of strings that will be converted to NaN
            missing_values_representations = [-1, '-1', '--', '-', '.', "/"]
            # Replace missing values with NaN
            df.replace(missing_values_representations, np.nan, inplace=True)
            # print the total number of rows
            total_num_of_rows = df.shape[0]
            print()
            print('Total number of rows: ', total_num_of_rows)
            print()
            print('Percentage of rows with missing values initially:', 
            self.print_percentages_of_rows_with_missing_values(df))
            # drop the 'respiration_rate' column as it has many missing values, and is not needed for the analysis
            df.drop(columns=['respiration_rate'], inplace=True)
            print('\nDataFrame after dropping the "respiration_rate" column:')
            self.print_percentages_of_rows_with_missing_values(df)
            # Let's drop the rows with more than 1 missing value, since we cannot impute them.
            missing_count_per_row = df.isnull().sum(axis=1)
            filtered_df = df[missing_count_per_row <= 1]
            print()
            print('DataFrame after dropping rows with more than 1 missing value:')
            self.print_percentages_of_rows_with_missing_values(filtered_df)

            return filtered_df
      
      def known_nearest_neighbor_imputer(self,filtered_df):
            
            # A percentage of the rows have exactly 1 missing value.
            # We will fill these missing values, by finding the nearest neighbor.
            # We can find the nearest neighbor to the row with the missing value and replace it with that value.
            # NOTE: We will use the nearest neighbor of any patient, not just the same patient.

            # Separate the "date" and patient_id columns from the rest of the data
            # (since they are not useful for the imputation, we will add them back later)
            ignored_cols = ['patient_id', 'date_of_birth', 'timestamp']

            ignored_data = filtered_df[ignored_cols]
            numeric_df = filtered_df.drop(columns=ignored_cols)

            missing_count_per_row = numeric_df.isnull().sum(axis=1)

            # Split the DataFrame into 2 parts: one with no missing values and one with exactly one missing value
            # Rows with no missing values
            no_missing_df = numeric_df[missing_count_per_row == 0]

            # Rows with exactly one missing value
            one_missing_df = numeric_df[missing_count_per_row == 1]

            # Initialize KNNImputer with a small number of neighbors (e.g., 1 or 2)
            # This will find the nearest neighbor to the row with the missing value and replace it with that value.
            imputer = KNNImputer(n_neighbors=1)

            # Impute missing values in the DataFrame with one missing value
            imputed_one_missing = imputer.fit_transform(one_missing_df)

            # Convert the imputed array back to a DataFrame
            imputed_one_missing_df = pd.DataFrame(imputed_one_missing, columns=numeric_df.columns, index=one_missing_df.index)

            # Combine the DataFrames back together
            combined_df = pd.concat([no_missing_df, imputed_one_missing_df])

            # Sort the combined DataFrame to maintain the original order
            combined_df = combined_df.sort_index()

            # Add the ignored columns back to the combined DataFrame
            combined_df = pd.concat([ignored_data, combined_df], axis=1)

            print('\nDataFrame after imputing the missing values:')
            self.print_percentages_of_rows_with_missing_values(combined_df)

            return combined_df

