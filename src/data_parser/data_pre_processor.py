import os
import re
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

from data_parser.time_series_processor import TimeSeriesProcessor


class DataPreProcessor:

    PATHOLOGIES_GROUPED_AS_OTHER = {
        "cns infection",
        "assdh",
        "status epilepticus",
        "intracranial hypertension due to acute leukemia",
        "hydrocephalus",
    }

    def __init__(self, raw_data_file_path):
        self.raw_data_file_path = raw_data_file_path

    def pre_process_dataset(self, hours, mode):
        """
        Preprocess the dataset and create lagged features.
        :param hours: Number of hours to use for creating lag features.
        :return: DataFrame with lagged features.
        """

        lagged_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", f"cleaned_df_lagged_{mode}.csv")
        )

        # if the file already exists, return the dataframe
        if os.path.exists(lagged_file_path):
            print(f"File {lagged_file_path} already exists. Loading the cleaned data from the file.")
            return pd.read_csv(lagged_file_path)

        # Construct the absolute path
        raw_data_file_path = self.raw_data_file_path
        # Read the data from the CSV file
        df = pd.read_csv(raw_data_file_path, engine='python')

        df = self.add_pathology_one_hot_features(df)

        df = self.transform_all_columns_to_float(df)

        print("\n~~~~ STEP 1: Deleting rows with negative ICP values ~~~~\n")
        df = self.delete_negative_icp_values(df)

        print("\n~~~~ STEP 2: Deleting patients with 2 rows or less ~~~~\n")
        # Delete patients with 2 rows or fewer
        df = df.groupby('patient_id').filter(lambda x: len(x) > 2)

        print("\n~~~~ STEP 3: Dropping the rows that have null \"ICP next\" column ~~~~\n")
        df = self.drop_rows_with_null_target_column(df)

        print("\n~~~~ STEP 4: Standardizing missing values (converting 0 to Nan, etc) ~~~~\n")
        df = self.standardize_missing_values(df)

        if mode == "regression" or input("Do you want to drop columns with more than 1 missing values? (y/n): ").lower() == 'y':
            print("\n~~~~ STEP 5: Dropping columns with high missing values ~~~~\n")
            df = self.drop_columns_with_high_missing_values(df)
            print(f"Number of rows in the dataset after the dropping: {df.shape[0]}")

        print("\n~~~~ STEP 6: Cleaning ICP outliers ~~~~\n")
        df = self.clean_icp_outliers(df)
        print(f"Number of rows in the dataset after the cleaning: {df.shape[0]}")

        if mode == "regression" or input("Do you want to impute missing values? (y/n): ").lower() == 'y':
            print("\n~~~~ STEP 7: Imputing missing values ~~~~\n")
            df = self.known_nearest_neighbor_imputer(df)

        print("\n~~~~ STEP 8: Creating lagged features ~~~~\n")
        df = self.create_lagged_features(df, hours, mode)

        print("Preprocessing complete.")
        print(f"Number of rows in the cleaned dataset: {df.shape[0]}")
        print(f"Number of rows with NaN values: {df.isnull().sum().sum()}")

        # Save also the cleaned data to a CSV file
        # Construct the file path
        output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        print(f"\nData Preprocessing completed. Saving cleaned data to {lagged_file_path}\n")
        # Save the lagged dataset to a CSV file
        df.to_csv(lagged_file_path, index=False)

        return df

    def add_pathology_one_hot_features(self, df):
        """
        Optionally enrich the dataframe with one-hot encoded pathology columns using the pathologies_filtered.csv file.
        Pathologies listed in PATHOLOGIES_GROUPED_AS_OTHER are grouped under a single "pathology_other" column.
        """
        try:
            user_input = input("Do you want to add one-hot encoded pathology columns? (y/n): ").strip().lower()
        except Exception:
            # In non-interactive contexts, default to skipping.
            return df

        if user_input not in ("y", "yes"):
            print("Skipping pathology one-hot encoding.")
            return df

        pathologies_file = os.path.abspath(
            os.path.join(os.path.dirname(self.raw_data_file_path), "pathologies_filtered.csv")
        )

        try:
            pathologies_df = pd.read_csv(pathologies_file, dtype=str)
        except FileNotFoundError:
            print(f"Warning: pathologies_filtered.csv not found at {pathologies_file}. Skipping pathology one-hot encoding.")
            return df
        except Exception as e:
            print(f"Error reading {pathologies_file}: {e}. Skipping pathology one-hot encoding.")
            return df

        if pathologies_df.shape[1] < 2:
            print(f"Unexpected format for {pathologies_file}. Skipping pathology one-hot encoding.")
            return df

        patient_col = pathologies_df.columns[0]
        pathology_col = pathologies_df.columns[1]

        pathologies_df[patient_col] = pathologies_df[patient_col].astype(str).str.strip()
        pathologies_df[pathology_col] = pathologies_df[pathology_col].astype(str).str.strip()

        patient_to_pathology = dict(zip(pathologies_df[patient_col], pathologies_df[pathology_col]))

        # Determine all unique pathologies (case-insensitive) excluding those grouped as "other".
        unique_pathologies = {
            p for p in pathologies_df[pathology_col].dropna().unique()
            if p.strip() and p.lower() not in self.PATHOLOGIES_GROUPED_AS_OTHER
        }

        if not unique_pathologies and not self.PATHOLOGIES_GROUPED_AS_OTHER:
            print("No pathologies found to encode. Skipping pathology one-hot encoding.")
            return df

        print(f"Found {len(unique_pathologies)} pathologies to one-hot encode. Grouping the following as 'other': {', '.join(sorted(self.PATHOLOGIES_GROUPED_AS_OTHER))}")

        def sanitize(name):
            name = name.lower().strip().replace(" ", "_")
            name = re.sub(r"[^0-9a-zA-Z_]", "", name)
            return name

        enriched_df = df.copy()

        # Initialize all pathology columns to 0.
        for pathology in sorted(unique_pathologies):
            col_name = f"pathology_{sanitize(pathology)}"
            enriched_df[col_name] = 0

        other_col = "pathology_other"
        enriched_df[other_col] = 0

        # Assign one-hot values per patient based on mapping.
        for idx, patient_id in enriched_df["patient_id"].astype(str).items():
            pathology = patient_to_pathology.get(patient_id, "").strip()
            if not pathology:
                continue
            if pathology.lower() in self.PATHOLOGIES_GROUPED_AS_OTHER:
                enriched_df.at[idx, other_col] = 1
                continue
            col_name = f"pathology_{sanitize(pathology)}"
            if col_name in enriched_df.columns:
                enriched_df.at[idx, col_name] = 1
            else:
                # Pathology not anticipated when initializing; treat as other to avoid losing signal.
                enriched_df.at[idx, other_col] = 1

        return enriched_df

    def transform_all_columns_to_float(self, df):
        # Convert all columns to float, except for 'patient_id', 'date_of_birth', and 'timestamp'
        for col in df.columns:
            if col not in ['patient_id', 'date_of_birth', 'timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("\nStatistics about the raw data after the transformation:\n")
        print(df.info())
        return df

    @staticmethod
    def print_percentages_of_rows_with_missing_values(dataframe):
        # Function to print the percentage of rows that have more than 1 column with missing values
        # and the percentage of rows that have exactly 1 column with missing values.

        total_num_of_rows = dataframe.shape[0]
        number_of_rows_with_more_than_1_missing_value = dataframe[dataframe.isna().sum(axis=1) > 1].shape[0]
        print('Percentage of rows that have more than 1 column with missing values: ',
              round(number_of_rows_with_more_than_1_missing_value / total_num_of_rows * 100, 3), '%')

        number_of_rows_with_1_missing_value = dataframe[dataframe.isna().sum(axis=1) == 1].shape[0]
        print('Percentage of rows that have exactly 1 column with missing value: ',
              round(number_of_rows_with_1_missing_value / total_num_of_rows * 100, 3), '%')

    def standardize_missing_values(self, df):
        """
        The dataset contains missing values that are represented by different strings ("--", ".", etc.).
        We need to standardize these missing values by replacing them with NaN.
        Standardize missing values in the dataset. Replace missing values with NaN (np.nan).
        :param df:
        :return:
        """
        # declare an array of strings that will be converted to NaN
        missing_values_representations = [-1, '-1', '--', '-', '.', "/"]
        # Replace missing values with NaN
        df.replace(missing_values_representations, np.nan, inplace=True)
        # print the total number of rows
        total_num_of_rows = df.shape[0]
        print()
        print('Total number of rows: ', total_num_of_rows)
        print()
        print('Percentage of rows with missing values after the standardization:',
              self.print_percentages_of_rows_with_missing_values(df))
        return df

    def drop_columns_with_high_missing_values(self, df):
        """
        Drop columns with a high percentage of missing values.
        :param df: DataFrame to process
        :return: DataFrame with columns dropped
        """
        # drop the 'respiration_rate' column as it has many missing values, and is not needed for the analysis
        df.drop(columns=['respiration_rate'], inplace=True)
        print('\nDataFrame after dropping the "respiration_rate" column:')
        self.print_percentages_of_rows_with_missing_values(df)
        # Let's drop the rows with more than 1 columns with missing value
        missing_count_per_row = df.isnull().sum(axis=1)
        filtered_df = df[missing_count_per_row <= 1]
        print()
        print('DataFrame after dropping rows with more than 1 missing value:')
        self.print_percentages_of_rows_with_missing_values(filtered_df)
        return filtered_df

    def known_nearest_neighbor_imputer(self, filtered_df):
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
        imputed_one_missing_df = pd.DataFrame(imputed_one_missing, columns=numeric_df.columns,
                                              index=one_missing_df.index)

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

    def clean_icp_outliers(self, cleaned_df):
        # Detect and remove outliers in the 'icp' column using the Z-score method
        # Calculate the Z-scores for each value in the 'icp' column
        z_scores = (cleaned_df['icp'] - cleaned_df['icp'].mean()) / cleaned_df['icp'].std()

        # Define a threshold for the Z-scores (how many standard deviations away from the mean)
        # We choose a threshold of 7, which is accepted for this domain
        z_score_threshold = 7

        # Identify the outliers based on the Z-scores
        outliers = cleaned_df[abs(z_scores) > z_score_threshold]

        print(f"Number of outliers detected: {len(outliers)}")

        # Print the outliers
        print("Outliers ICP:")
        print(outliers[['patient_id', 'timestamp', 'icp']])

        # Remove the outliers from the DataFrame
        cleaned_df = cleaned_df[abs(z_scores) <= z_score_threshold]

        return cleaned_df

    def create_lagged_features(self, data, hours, mode):
        """
        Preprocess the dataset and create lagged features.
        :param data: DataFrame to process
        :param hours: Number of hours to use for creating lag features
        """

        # Create lagged features
        columns_to_lag = [
            "icp", "temperature", "mean_blood_pressure", "cpp", "glucose",
            "haemoglobin", "heart_rate", "paco2", "pao2", "peep", "ph", "spo2"
        ]
        time_series_processor = TimeSeriesProcessor()
        data = time_series_processor.process_data(
            data, hours=hours, columns_to_lag=columns_to_lag, mode=mode
        )

        print(f"\nColumns after creating lagged features: {data.columns}")

        return data

    def drop_rows_with_null_target_column(self, data):
        # Store the initial number of rows
        initial_row_count = data.shape[0]

        # Drop rows with null 'icp' values
        data = data.dropna(subset=['icp'])

        # Calculate the number of dropped rows
        dropped_row_count = initial_row_count - data.shape[0]

        print(f"Number of rows in the dataset after the dropping: {data.shape[0]}")
        print(f"Number of dropped rows: {dropped_row_count}")

        return data
