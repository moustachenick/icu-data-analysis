import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path


class DataPreProcessor:

      def __init__(self, file_path):
            self.file_path = file_path

      def pre_process_dataset(self):
            filtered_df = self.replace_missing_values()

            combined_df = self.known_nearest_neighbor_imputer(filtered_df)

            cleaned_df = self.delete_negative_icp_values(combined_df)

            cleaned_df = self.shift_icp_values(cleaned_df)

            # Save also the cleaned data to a CSV file
            # Construct the file path
            output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

            filepath = output_dir / 'cleaned_df.csv'

            cleaned_df.to_csv(filepath)

            return cleaned_df
           
      
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
            file_path = self.file_path
            
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

      def delete_negative_icp_values(self, combined_df): 

            # Ensure 'icp' is a numeric column
            combined_df['icp'] = pd.to_numeric(combined_df['icp'], errors='coerce')

            # Count the number of negative 'icp' values
            negative_icp_count = (combined_df['icp'] < 0).sum()
            print(f"Number of negative ICP values: {negative_icp_count}")
            print("Total Rows before deletion =", len(combined_df))

            # Delete rows where 'icp' is negative
            cleaned_df = combined_df[combined_df['icp'] >= 0]

            # Print the updated DataFrame details
            print("Total Rows after deleting negative ICP values =", len(cleaned_df))

            return cleaned_df
      
      def shift_icp_values(self, data):
            # A problem that we have in the dataset, is that each row has the ICP at that timestamp, 
            # and we actually want to predict the ICP in the "next" timestamp. 
            # So for example to be able to say: "OK, I see that patient with id 1001 right now has X temperature, Y blood pressure, and Z paco2, 
            # so what is his ICP going to be in an hour"? 
            # we could shift the ICP values by 1 row, so that each row has the ICP value at the next timestamp.
            # A problem is that the measurements are not taken at regular intervals, so we cannot just shift the ICP values by 1 row.
            # Instead of shifting the ICP values by one row, we need to ensure that the next timestamp is within a reasonable timeframe, such as within an hour.
            # We can add a new column called 'next_icp' that contains the ICP value at the next timestamp.

            # So, the approach is to:
            # 1.  Calculate Time Differences Between Consecutive Rows:
            #     For each patient, calculate the time difference between consecutive measurements. 
            #     Only keep rows where the time difference is within a specified threshold (e.g., 1 hour)
            # 2. Shift ICP Values for Rows with Valid Gaps: 
            #     Once the time gaps are calculated, shift the ICP values for rows that meet the time threshold. 
            #     Rows where the gap exceeds the threshold can be excluded..

            # Make a copy of the DataFrame to avoid the SettingWithCopyWarning
            data = data.copy()
            # Convert the timestamp column to datetime format
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Shift the timestamp first, then calculate the difference between consecutive rows
            data['time_shifted'] = data.groupby('patient_id')['timestamp'].shift(-1)

            # Now calculate the time differences between consecutive rows for each patient
            data['time_diff'] = data['time_shifted'] - data['timestamp']

            # Drop the 'time_shifted' column as it is no longer needed
            data = data.drop(columns=['time_shifted'])

            # Define the maximum time gap (e.g., 1 hour) for considering the "next" timestamp
            max_time_gap = pd.Timedelta(hours=1)

            # Shift the ICP column where the time difference is within the acceptable limit (e.g., 1 hour)
            data['icp_next'] = data.groupby('patient_id')['icp'].shift(-1)

            # Use .loc to avoid the SettingWithCopyWarning when modifying the DataFrame
            data.loc[data['time_diff'] > max_time_gap, 'icp_next'] = None  # Exclude rows with large time gaps

            # Drop rows where the target (icp_next) is NaN, using .loc again to avoid the warning
            data = data.loc[~data['icp_next'].isna()]

            # Drop the 'time_diff' column as it is no longer needed
            data = data.drop(columns=['time_diff'])

            return data
