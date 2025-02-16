import pandas as pd


class TimeSeriesProcessor:

    def process_data(self, data, hours=2, columns_to_lag=None):
        """
        Process the full dataset by creating lag features for each row and handling missing values.
        This method creates lag features for each row based on the previous measurements within the specified number of hours.
        Then, forward fill missing values per patient to ensure no missing values in non-lagged columns.
        :param data: data (pd.DataFrame): The full dataset containing time series data for multiple patients.
        :param hours: hours (float): The number of hours to use for creating lag features.
        :param columns_to_lag: List of columns for which to create lag features.
        :return: pd.DataFrame: A new dataframe with lagged features and missing values handled.
        """
        print("\nTime Series Processor initiated.")

        # if the data is empty, return an empty DataFrame
        if data.empty:
            print("Warning: Input DataFrame is empty. Returning the data as is.")
            return pd.DataFrame(data)

        print("\nSTEP 1: Processing time series data and creating lag features. This may take a while...☕")

        # give only the 10% of the data (to speed up the process)
        # data = data.sample(frac=0.1, random_state=42)

        print(f"\nColumns of data: {data.columns}")

        processed_data_df = self.create_lagged_features_dataset(data, hours, columns_to_lag)

        if input("Do you want to forward-fill non-lagged values? (y/n): ").lower() == 'y':
            print("\nSTEP 2: Forward filling missing non-lagged values per patient")
            # Identify non-lagged columns
            non_lagged_columns = [col for col in processed_data_df.columns if 'lag' not in col]

            if 'patient_id' not in processed_data_df.columns:
                raise KeyError("'patient_id' column is missing before groupby operation.")

            # Use forward fill per patient to handle missing values in non-lagged columns
            if processed_data_df.empty:
                print("Warning: Processed DataFrame is empty. Skipping forward fill.")
            else:
                # Forward fill only the non-lagged columns
                processed_data_df[non_lagged_columns] = (
                    processed_data_df.groupby('patient_id', group_keys=False)[non_lagged_columns]
                    .transform('ffill')
                    .infer_objects()  # Ensures correct data types
                )

            print("\nSTEP 3: Dropping rows with missing values in lagged columns")
            processed_data_df = processed_data_df.dropna(
                subset=[col for col in processed_data_df.columns if 'lag' in col])

        return pd.DataFrame(processed_data_df)

    def create_lagged_features_dataset(self, data, hours=2, columns_to_lag=None):
        """
        Main method to process the full dataset by creating lag features for each row.
        This method processes each patient's data independently and then concatenates the results.

        :param data: data (pd.DataFrame): The full dataset containing time series data for multiple patients.
        :param hours: hours (float): The number of hours to use for creating lag features.
        :param columns_to_lag: List of columns for which to create lag features.
        :return: pd.DataFrame: A new dataframe with lagged features as additional columns.
        """
        if columns_to_lag is None:
            columns_to_lag = ['icp', 'heart_rate', 'temperature']  # Default columns

        # Create an empty list to store processed data for each patient
        processed_data = []

        # Process each patient independently
        for patient_id, patient_data in data.groupby('patient_id', group_keys=False):
            # Sort the patient's data by timestamp to ensure it's in chronological order
            patient_data = patient_data.sort_values(by='timestamp')

            # Process the patient data by generating lag features
            processed_patient_data = self.process_patient_data(patient_data, hours, columns_to_lag)

            # Append the processed patient data to the full dataset
            processed_data.extend(processed_patient_data)

        return pd.DataFrame(processed_data)

    def process_patient_data(self, patient_data, hours, columns_to_lag):
        """
        Process the time series data for a single patient by generating lag features.
        This method creates lag features for each row based on the previous measurements within the specified number of hours.

        Args:
            patient_data (pd.DataFrame): DataFrame containing a single patient's data.
            hours (float): The number of hours to use for creating lag features.
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
            previous_rows = self.get_previous_rows(patient_data, idx, hours)

            # Create lag features for the current row based on previous rows
            lagged_row = self.create_lagged_features_for_row(current_row, previous_rows, columns_to_lag, hours)

            # Append the processed row (with lag features) to the processed data list
            processed_patient_data.append(lagged_row)

        return processed_patient_data

    def get_previous_rows(self, patient_data, current_index, hours):
        """
        Retrieve the previous rows within the specified number of hours for the current row, excluding the current row itself.

        Args:
            patient_data (pd.DataFrame): DataFrame containing a single patient's data, sorted by timestamp.
            current_index (int): The index of the current row in the patient's data.
            hours (float): The number of hours to look back for previous rows.

        Returns:
            pd.DataFrame: A DataFrame containing the previous rows within the specified number of hours.
        """
        # Get the timestamp of the current row
        current_time = patient_data.iloc[current_index]['timestamp']

        # Define the exact start time for retrieving past rows
        start_time = current_time - pd.Timedelta(hours=hours)

        # Retrieve all previous rows that fall within the given time window
        previous_rows = patient_data[
            (patient_data['timestamp'] >= start_time) & (patient_data['timestamp'] < current_time)
            ]

        return previous_rows

    def create_lagged_features_for_row(self, current_row, previous_rows, columns_to_lag, hours):
        """
        Create lagged features for a given row based on the previous measurements within the specified number of hours.

        Args:
            current_row (pd.Series): The current row of data.
            previous_rows (pd.DataFrame): The previous rows of data within the specified number of hours.
            columns_to_lag (list): List of columns for which to create lag features.
            hours (float): The number of hours to use for creating lag features.

        Returns:
            dict: A dictionary representing the row with lagged features added.

        Logic:
        1. Initialize a dictionary with the current row's data.
        2. Sort the previous rows in descending order of their index to ensure the most recent rows come first.
        3. Determine the maximum number of lags to create, which is the minimum of the number of available previous rows
        and the specified number of hours.
        4. For each column in `columns_to_lag`:
            a. Create lag features for the current row by iterating through the previous rows up to the `max_lag` value.
            b. If fewer previous rows are available than the specified number of hours, fill the remaining lag features with None.
        5. Return the dictionary containing the current row's data along with the newly created lagged features.
        """
        # Initialize the row dictionary with the current row data
        row_dict = current_row.to_dict()

        # Ensure previous rows are ordered from most recent to the oldest
        previous_rows = previous_rows.sort_index(ascending=False)

        # Determine the maximum number of lags based on available data (at most `hours`, but can be fewer)
        # If there are fewer previous rows than `hours`, use the number of available previous rows
        max_lag = min(len(previous_rows), int(hours))

        # For each column, create lag features
        for col in columns_to_lag:
            # Iterate through the previous rows up to the `max_lag` value
            for lag in range(1, max_lag + 1):
                # Add a new column for each lag value, e.g., 'icp_lag_1', 'icp_lag_2', etc.
                row_dict[f"{col}_lag_{lag}"] = previous_rows.iloc[lag - 1][col]

            # If fewer than `hours` lags are available, fill the rest with None
            for lag in range(max_lag + 1, int(hours) + 1):
                row_dict[f"{col}_lag_{lag}"] = None

        return row_dict
