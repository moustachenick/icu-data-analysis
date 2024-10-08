import pandas as pd
import numpy as np

class TimeSeriesProcessor:
    """
    This class processes time series data, including:
    - Filling missing 30-minute time intervals.
    - Imputing missing values using specified strategies.
    - Creating lagging features.
    - Processing entire datasets grouped by patient_id.
    """

    def fill_missing_time_intervals(self, df):
        """
        Fill missing 30-minute intervals in the dataset.
        Args:
            df (pd.DataFrame): Input dataframe with a timestamp column.

        Returns:
            pd.DataFrame: Dataframe with missing 30-minute intervals filled.
        """
        # Ensure the 'timestamp' column is in datetime format and set as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Create a complete range of timestamps at 30-minute intervals
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='30T')

        # Reindex the dataframe to this full range, filling missing timestamps with NaN
        df = df.reindex(full_range)

        # Forward fill 'patient_id' to ensure newly inserted rows have the correct value
        df['patient_id'] = df['patient_id'].ffill()

        # Rename the index back to 'timestamp'
        df.index.name = 'timestamp'

        return df



    def impute_missing_rows(self, df, method='ffill'):
        """
        Impute missing values in the dataframe using the specified method.
        Args:
            df (pd.DataFrame): Dataframe with missing rows (NaN) to impute.
            method (str): Imputation method. Options are 'ffill', 'bfill', 'mean', 'median'.

        Returns:
            pd.DataFrame: Dataframe with imputed missing values.
        """
        if method == 'ffill':
            # Forward fill missing values
            df = df.ffill()
        elif method == 'bfill':
            # Backward fill missing values
            df = df.bfill()
        elif method == 'mean':
            # Impute with column means
            df = df.fillna(df.mean())
        elif method == 'median':
            # Impute with column medians
            df = df.fillna(df.median())
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        print(f"After imputing missing values using {method}, dataframe is:")
        print(df)
        return df

    def create_lagged_features(self, df, lags, columns):
        """
        Create lagging features for specified columns.
        Args:
            df (pd.DataFrame): Input dataframe.
            lags (int): Number of lag periods to create.
            columns (list): List of columns to lag.

        Returns:
            pd.DataFrame: Dataframe with lagged features added.
        """
        for lag in range(1, lags + 1):
            for column in columns:
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)

        print(f"After creating lagged features, dataframe is:")
        print(df)
        return df

    def process_data(self, data, lags, columns_to_lag, imputation_method='ffill'):
        patients = data['patient_id'].unique()
        processed_dfs = []

        for patient in patients:
            patient_data = data[data['patient_id'] == patient].copy()

            # Fill missing time intervals
            patient_data = self.fill_missing_time_intervals(patient_data)

            # Impute missing values using the specified method
            patient_data = self.impute_missing_rows(patient_data, method=imputation_method)

            # Create lagged features for each column
            patient_data = self.create_lagged_features(patient_data, lags, columns_to_lag)

            # Drop rows with NaN in lagged columns (i.e., the first row)
            lagged_columns = [f"{col}_lag_{i}" for col in columns_to_lag for i in range(1, lags + 1)]
            patient_data = patient_data.dropna(subset=lagged_columns)

            processed_dfs.append(patient_data)

        print("Processed data from all patients:")
        print(pd.concat(processed_dfs))

        # Concatenate the processed data from all patients
        final_df = pd.concat(processed_dfs)
        return final_df

