import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from icu_data_parser.TimeSeriesProcessor import TimeSeriesProcessor  # Make sure to replace with the actual module name


class TestTimeSeriesProcessor(unittest.TestCase):

    def setUp(self):
        """
        Set up sample data for testing.
        """
        self.processor = TimeSeriesProcessor()
        self.data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 2, 2],
            'timestamp': pd.to_datetime([
                '2023-01-01 00:00', '2023-01-01 00:30', '2023-01-01 01:00', '2023-01-01 01:30', '2023-01-01 02:00',
                '2023-01-01 00:00', '2023-01-01 00:30'
            ]),
            'icp': [10, 12, 14, 13, 15, 9, 10],
            'heart_rate': [70, 75, 72, 73, 71, 66, 68],
            'temperature': [36.5, 36.6, 36.7, 36.8, 36.7, 37.0, 37.1]
        })

    def test_process_data(self):
        """
        Test the entire process_data method to ensure lag features are generated correctly,
        by using input and expected output data from external CSV files.
        """
        # Load the input data and expected output from CSV files
        input_data = pd.read_csv('datasets/input_data.csv', parse_dates=['timestamp'], index_col=False)
        expected_output = pd.read_csv('datasets/expected_output.csv', parse_dates=['timestamp'], index_col=False)

        # Process the input data and create the lag features
        result = self.processor.process_data(input_data, lags=2, columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)
        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result, expected_output)

    def test_process_patient_data(self):
        """
        Test processing patient data and creating lag features for a single patient.
        """
        patient_data = self.data[self.data['patient_id'] == 1]
        expected_output = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 00:00', '2023-01-01 00:30', '2023-01-01 01:00', '2023-01-01 01:30', '2023-01-01 02:00'
            ]),
            'icp': [10, 12, 14, 13, 15],
            'heart_rate': [70, 75, 72, 73, 71],
            'temperature': [36.5, 36.6, 36.7, 36.8, 36.7],
            'icp_lag_1': [None, 10, 12, 14, 13],
            'icp_lag_2': [None, None, 10, 12, 14],
            'heart_rate_lag_1': [None, 70, 75, 72, 73],
            'heart_rate_lag_2': [None, None, 70, 75, 72],
            'temperature_lag_1': [None, 36.5, 36.6, 36.7, 36.8],
            'temperature_lag_2': [None, None, 36.5, 36.6, 36.7]
        })

        result = self.processor.process_patient_data(patient_data, lags=2,
                                                     columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Convert result from list of dicts to DataFrame for comparison
        result_df = pd.DataFrame(result)

        assert_frame_equal(result_df, expected_output)

    def test_get_previous_rows(self):
        """
        Test the get_previous_rows method to ensure the correct number of previous rows are retrieved.
        """
        patient_data = self.data[self.data['patient_id'] == 1]

        # Test retrieving the last 2 rows before index 3
        result = self.processor.get_previous_rows(patient_data, 3, 2)
        expected_output = pd.DataFrame({
            'patient_id': [1, 1],
            'timestamp': pd.to_datetime(['2023-01-01 00:30', '2023-01-01 01:00']),
            'icp': [12, 14],
            'heart_rate': [75, 72],
            'temperature': [36.6, 36.7]
        }).set_index(patient_data.index[1:3])

        assert_frame_equal(result, expected_output)

    def test_create_lagged_features(self):
        """
        Test the create_lagged_features method to ensure the correct lagged features are created.
        """
        current_row = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 02:00:00'),
            'icp': 15,
            'heart_rate': 71,
            'temperature': 36.7
        })

        previous_rows = pd.DataFrame({
            'icp': [10, 12, 14, 13],
            'heart_rate': [70, 75, 72, 73],
            'temperature': [36.5, 36.6, 36.7, 36.8]
        })

        expected_output = {
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 02:00:00'),
            'icp': 15,
            'heart_rate': 71,
            'temperature': 36.7,
            'icp_lag_1': 13,
            'icp_lag_2': 14,
            'heart_rate_lag_1': 73,
            'heart_rate_lag_2': 72,
            'temperature_lag_1': 36.8,
            'temperature_lag_2': 36.7
        }

        result = self.processor.create_lagged_features(current_row, previous_rows, ['icp', 'heart_rate', 'temperature'],
                                                       lags=2)

        self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()
