import os

import pandas as pd
from pathlib import Path


class TimeSeriesProcessor:

    def process_data(self, data, lags=2, columns_to_lag=None):
        """
        Process the full dataset by creating lag features for each row and handling missing values.
        Then, forward fill missing values per patient to ensure no missing values in lagged columns.
        Also, convert the processed data back to a DataFrame.
        :param data: data (pd.DataFrame): The full dataset containing time series data for multiple patients.
        :param lags: lags (int): The number of lag periods to use.
        :param columns_to_lag: List of columns for which to create lag features.
        :return: pd.DataFrame: A new dataframe with lagged features and missing values handled.
        """
        print("\nTime Series Processor initiated.")
        print("\nSTEP 1: Processing time series data and creating lag features. This may take a while...")

        processed_data_df = self.create_lagged_features_dataset(data, lags, columns_to_lag)

        if input("Do you want to Handle missing values and forward-fill values? (y/n): ").lower() == 'y':
            print("\nstep 2: Handling missing values in lagged columns")
            processed_data_df = processed_data_df.dropna(
                subset=[col for col in processed_data_df.columns if 'lag' in col])

            print("\nSTEP 3: Forward filling missing non-lagged values per patient")
            # Use forward fill per patient to handle missing values in non-lagged columns
            processed_data_df = processed_data_df.groupby('patient_id').apply(
                lambda group: group.fillna(method='ffill'))

        return pd.DataFrame(processed_data_df)

    def create_lagged_features_dataset(self, data, lags=2, columns_to_lag=None):
        """
        Main method to process the full dataset by creating lag features for each row.

        :param data: data (pd.DataFrame): The full dataset containing time series data for multiple patients.
        :param lags: lags (int): The number of lag periods to use.
        :param columns_to_lag: List of columns for which to create lag features.
        :return: pd.DataFrame: A new dataframe with lagged features.
        """
        if columns_to_lag is None:
            columns_to_lag = ['icp', 'heart_rate', 'temperature']  # Default columns

        # Create an empty list to store processed data for each patient
        processed_data = []

        # Process each patient independently
        for patient_id, patient_data in data.groupby('patient_id'):
            # Sort the patient's data by timestamp to ensure it's in chronological order
            patient_data = patient_data.sort_values(by='timestamp')

            # Process the patient data by generating lag features
            processed_patient_data = self.process_patient_data(patient_data, lags, columns_to_lag)

            # Append the processed patient data to the full dataset
            processed_data.extend(processed_patient_data)

        return pd.DataFrame(processed_data)

    def process_patient_data(self, patient_data, lags, columns_to_lag):
        """
        Process the time series data for a single patient by generating lag features.

        Args:
            patient_data (pd.DataFrame): DataFrame containing a single patient's data.
            lags (int): The number of lag periods to use.
            columns_to_lag (list): List of columns for which to create lag features.

        Returns:
            List[dict]: A list of dictionaries where each dictionary represents one row of processed data with lagged features.
        """
        processed_patient_data = []

        # Ensure the patient_data index is reset for each patient to avoid index issues
        patient_data = patient_data.reset_index(drop=True)

        # Iterate through each row in the patient's data
        for idx, current_row in patient_data.iterrows():
            # Get the previous rows up to the current row index (but not including it)
            previous_rows = self.get_previous_rows(patient_data, idx, lags)

            # Create lag features for the current row based on previous rows
            lagged_row = self.create_lagged_features_for_row(current_row, previous_rows, columns_to_lag, lags)

            # Append the processed row (with lag features) to the processed data list
            processed_patient_data.append(lagged_row)

        return processed_patient_data

    def get_previous_rows(self, patient_data, current_index, lags):
        """
        Retrieve the previous `lags` rows for the current row, excluding the current row itself.

        Args:
            patient_data (pd.DataFrame): DataFrame containing a single patient's data, sorted by timestamp.
            current_index (int): The index of the current row in the patient's data.
            lags (int): The number of previous rows (lags) to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the previous `lags` rows, or as many rows as available.
        """
        # Start at the current_index - lags and stop at the current_index
        start_index = max(0, current_index - lags)  # Ensure index doesn't go below 0
        return patient_data.iloc[start_index:current_index]  # Slice and return the rows

    def create_lagged_features_for_row(self, current_row, previous_rows, columns_to_lag, lags):
        """
        Create lagged features for a given row based on the previous `lags` measurements.

        Args:
            current_row (pd.Series): The current row of data.
            previous_rows (pd.DataFrame): The previous `lags` rows of data.
            columns_to_lag (list): List of columns for which to create lag features.
            lags (int): The number of lag periods to use.

        Returns:
            dict: A dictionary representing the row with lagged features added.
        """
        # Initialize the row dictionary with the current row data
        row_dict = current_row.to_dict()

        # For each column, create lag features
        for col in columns_to_lag:
            # Ensure previous rows are ordered from most recent to least recent (chronological order)
            previous_rows = previous_rows.sort_index(ascending=False)

            # Iterate over the lags and assign the correct lag values
            for lag in range(1, lags + 1):
                if len(previous_rows) >= lag:
                    # Get the lagged value from the correct previous row
                    row_dict[f"{col}_lag_{lag}"] = previous_rows.iloc[lag - 1][col]
                else:
                    # If there aren't enough previous rows, assign NaN
                    row_dict[f"{col}_lag_{lag}"] = None

        return row_dict
