import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from icu_data_parser.TimeSeriesProcessor import TimeSeriesProcessor


class TestTimeSeriesProcessor(unittest.TestCase):
    """
    Unit tests for the TimeSeriesProcessor class.
    """

    def setUp(self):
        """
        Set up sample data for testing.
        """
        self.processor = TimeSeriesProcessor()

        # Sample dataset with 30-minute intervals and some gaps
        self.sample_data_1 = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:30:00', '2023-09-20 14:00:00'],
            'icp': [10, 11, 12, 13],
            'heart_rate': [70, 72, 74, 76],
            'temperature': [36.5, 36.6, 36.7, 36.8]
        })
        self.sample_data_1['timestamp'] = pd.to_datetime(self.sample_data_1['timestamp'])

        # Another sample dataset with missing values and some gaps
        self.sample_data_2 = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:00:00', '2023-09-20 13:30:00'],
            'icp': [10, None, 12, None],
            'heart_rate': [70, None, 74, None],
            'temperature': [36.5, None, 36.7, None]
        })
        self.sample_data_2['timestamp'] = pd.to_datetime(self.sample_data_2['timestamp'])

    # 1. Testing fill_missing_time_intervals
    def test_fill_missing_time_intervals_with_one_timestamp_missing(self):
        """
        Test filling missing 30-minute intervals, including missing rows.
        """
        # Expected result after filling missing time intervals
        expected_data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:00:00', '2023-09-20 13:30:00',
                          '2023-09-20 14:00:00'],
            'icp': [10, 11, None, 12, 13],
            'heart_rate': [70, 72, None, 74, 76],
            'temperature': [36.5, 36.6, None, 36.7, 36.8]
        })
        expected_data['timestamp'] = pd.to_datetime(expected_data['timestamp'])
        expected_data.set_index('timestamp', inplace=True)

        # Process the data
        processed_data = self.processor.fill_missing_time_intervals(self.sample_data_1)

        # Reset indices for both dataframes to ensure alignment
        processed_data.reset_index(inplace=True)
        expected_data.reset_index(inplace=True)

        # Convert the 'patient_id' column to int64 to ensure consistency
        processed_data['patient_id'] = processed_data['patient_id'].astype('int64')
        expected_data['patient_id'] = expected_data['patient_id'].astype('int64')

        # Ensure that missing time intervals are filled correctly
        assert_frame_equal(processed_data, expected_data)

    def test_fill_missing_time_intervals_with_many_timestamps_missing(self):
        """
        Test filling missing 30-minute intervals with a larger gap.
        """
        sample_data = pd.DataFrame({
            'patient_id': [1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 14:00:00'],
            'icp': [10, 11, 13],
            'heart_rate': [70, 72, 76],
            'temperature': [36.5, 36.6, 36.8]
        })
        sample_data['timestamp'] = pd.to_datetime(sample_data['timestamp'])

        expected_data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:00:00', '2023-09-20 13:30:00',
                          '2023-09-20 14:00:00'],
            'icp': [10, 11, None, None, 13],
            'heart_rate': [70, 72, None, None, 76],
            'temperature': [36.5, 36.6, None, None, 36.8]
        })
        expected_data['timestamp'] = pd.to_datetime(expected_data['timestamp'])
        expected_data.set_index('timestamp', inplace=True)

        # Process the data
        processed_data = self.processor.fill_missing_time_intervals(sample_data)

        # Reset indices for both dataframes to ensure alignment
        processed_data.reset_index(inplace=True)
        expected_data.reset_index(inplace=True)

        # Convert the 'patient_id' column to int64 to ensure consistency
        processed_data['patient_id'] = processed_data['patient_id'].astype('int64')
        expected_data['patient_id'] = expected_data['patient_id'].astype('int64')

        # Ensure that missing time intervals are filled correctly
        assert_frame_equal(processed_data, expected_data)

    # 2. Testing impute_missing_rows
    def test_impute_missing_rows_scenario_with_forward_fill_strategy(self):
        """
        Test forward fill (ffill) imputation for missing values.
        """
        expected_data_ffill = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:00:00', '2023-09-20 13:30:00'],
            'icp': [10, 10, 12, 12],
            'heart_rate': [70, 70, 74, 74],
            'temperature': [36.5, 36.5, 36.7, 36.7]
        })
        expected_data_ffill['timestamp'] = pd.to_datetime(expected_data_ffill['timestamp'])

        # Convert expected data columns to float64 (to match the imputed data type)
        expected_data_ffill['icp'] = expected_data_ffill['icp'].astype('float64')
        expected_data_ffill['heart_rate'] = expected_data_ffill['heart_rate'].astype('float64')
        expected_data_ffill['temperature'] = expected_data_ffill['temperature'].astype('float64')

        imputed_data = self.processor.impute_missing_rows(self.sample_data_2, method='ffill')

        # Reset indices for both dataframes to ensure alignment
        imputed_data.reset_index(inplace=True)
        expected_data_ffill.reset_index(inplace=True)

        # Convert the 'patient_id' column to int64 to ensure consistency
        imputed_data['patient_id'] = imputed_data['patient_id'].astype('int64')
        expected_data_ffill['patient_id'] = expected_data_ffill['patient_id'].astype('int64')

        # Ensure that forward fill imputation has worked as expected
        assert_frame_equal(imputed_data, expected_data_ffill)

    def test_impute_missing_rows_scenario_with_mean_strategy(self):
        """
        Test mean imputation for missing values.
        """
        expected_data_mean = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:00:00', '2023-09-20 13:30:00'],
            'icp': [10, 11, 12, 11],  # Expecting mean of 10 and 12 for the missing value
            'heart_rate': [70, 72, 74, 72],  # Expecting mean of 70 and 74 for the missing value
            'temperature': [36.5, 36.6, 36.7, 36.6]  # Expecting mean of 36.5 and 36.7 for the missing value
        })
        expected_data_mean['timestamp'] = pd.to_datetime(expected_data_mean['timestamp'])

        # Convert expected data columns to float64 (to match the imputed data type)
        expected_data_mean['icp'] = expected_data_mean['icp'].astype('float64')
        expected_data_mean['heart_rate'] = expected_data_mean['heart_rate'].astype('float64')
        expected_data_mean['temperature'] = expected_data_mean['temperature'].astype('float64')

        # Use mean imputation instead of forward fill
        imputed_data = self.processor.impute_missing_rows(self.sample_data_2, method='mean')

        # Reset indices for both dataframes to ensure alignment
        imputed_data.reset_index(inplace=True)
        expected_data_mean.reset_index(inplace=True)

        # Convert the 'patient_id' column to int64 to ensure consistency
        imputed_data['patient_id'] = imputed_data['patient_id'].astype('int64')
        expected_data_mean['patient_id'] = expected_data_mean['patient_id'].astype('int64')

        # Ensure that mean imputation has worked as expected
        assert_frame_equal(imputed_data, expected_data_mean)

    # 3. Testing create_lagged_features
    def test_create_lagged_features(self):
        """
        Test creation of lagged features for 1 lag period.
        """
        expected_data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1],
            'timestamp': ['2023-09-20 12:00:00', '2023-09-20 12:30:00', '2023-09-20 13:30:00', '2023-09-20 14:00:00'],
            'icp': [10, 11, 12, 13],
            'heart_rate': [70, 72, 74, 76],
            'temperature': [36.5, 36.6, 36.7, 36.8],
            'icp_lag_1': [None, 10, 11, 12],
            'heart_rate_lag_1': [None, 70, 72, 74],
            'temperature_lag_1': [None, 36.5, 36.6, 36.7]
        })
        expected_data['timestamp'] = pd.to_datetime(expected_data['timestamp'])

        lagged_data = self.processor.create_lagged_features(self.sample_data_1, lags=1,
                                                            columns=['icp', 'heart_rate', 'temperature'])

        # Ensure that lagged features have been created correctly
        assert_frame_equal(lagged_data.reset_index(drop=True), expected_data.reset_index(drop=True))
