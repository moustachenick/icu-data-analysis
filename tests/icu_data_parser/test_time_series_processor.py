import os
import unittest
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

# Adjust the Python path to include the src directory
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_parser.time_series_processor import TimeSeriesProcessor


class TestTimeSeriesProcessor(unittest.TestCase):

    def setUp(self):
        """
        Set up sample data for testing.
        """
        self.processor = TimeSeriesProcessor()

        # Get the directory of the current test file
        test_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the datasets directory
        self.datasets_dir = os.path.join(test_dir, 'datasets')

    # Add the @patch decorator to mock the input() function
    # in order to avoid user input during testing
    @patch('builtins.input', side_effect=['y', 'y'])
    def test_process_data(self, mock_input):
        """
        Test the process_data method to ensure the correct output is generated.
        """
        # Load the input data and expected output from CSV files
        input_data_path = os.path.join(self.datasets_dir, 'input_data.csv')
        expected_output_path = os.path.join(self.datasets_dir, 'expected_output_for_process_data.csv')

        input_data = pd.read_csv(input_data_path, parse_dates=['timestamp'], index_col=False)
        expected_output = pd.read_csv(expected_output_path, parse_dates=['timestamp'], index_col=False)

        # Process the input data and create the lag features
        result = self.processor.process_data(input_data, hours=2, columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Convert lagged columns to float64 in expected_output to match the result DataFrame
        lagged_columns = [col for col in expected_output.columns if 'lag' in col]
        expected_output[lagged_columns] = expected_output[lagged_columns].astype('float64')

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)

        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result, expected_output)

    def test_create_lagged_features_for_dataset(self):
        """
        Test the entire process_data method to ensure lag features are generated correctly,
        by using input and expected output data from external CSV files.
        """
        # Load the input data and expected output from CSV files
        input_data_path = os.path.join(self.datasets_dir, 'input_data.csv')
        expected_output_path = os.path.join(self.datasets_dir, 'expected_output_for_create_lagged_features.csv')

        input_data = pd.read_csv(input_data_path, parse_dates=['timestamp'], index_col=False)
        expected_output = pd.read_csv(expected_output_path, parse_dates=['timestamp'], index_col=False)

        # Process the input data and create the lag features
        result = self.processor.create_lagged_features_dataset(input_data, hours=2,
                                                               columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)

        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result, expected_output)

    def test_process_patient_data(self):
        """
        Test processing patient data and creating lag features for a single patient.
        """
        input_data_path = os.path.join(self.datasets_dir, 'input_data.csv')
        input_data = pd.read_csv(input_data_path, parse_dates=['timestamp'], index_col=False)
        patient_data = input_data[input_data['patient_id'] == 1]
        expected_output_path = os.path.join(self.datasets_dir, 'expected_output_for_process_patient_data.csv')
        expected_output = pd.read_csv(expected_output_path, parse_dates=['timestamp'], index_col=False)

        result = self.processor.process_patient_data(patient_data, hours=2,
                                                     columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # the result is a list of dictionaries, convert it to a DataFrame
        result = pd.DataFrame(result)

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)

        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result, expected_output)

    def test_create_lagged_features_for_row(self):
        # Define the current row
        current_row = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 04:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9
        })

        # Define the previous rows with multiple measurements within the same hour
        previous_rows = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 03:45:00',  # Within hour 1
                '2023-01-01 03:15:00',  # Within hour 1
                '2023-01-01 02:50:00',  # Within hour 2
                '2023-01-01 02:10:00',  # Within hour 2
                '2023-01-01 02:00:00',  # Within hour 2
                '2023-01-01 01:30:00',  # Within hour 3 (should not be used since we only go back 2 hours)
                '2023-01-01 01:10:00',  # Within hour 3 (should not be used)
                '2023-01-01 00:50:00'   # Within hour 4 (should not be used)
            ]),
            'icp': [14, 13, 15, 16, 14, 12, 11, 10],
            'heart_rate': [87, 86, 85, 88, 83, 82, 81, 80],
            'temperature': [36.8, 36.7, 36.7, 36.9, 36.6, 36.6, 36.5, 36.5]
        })

        expected_output = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 04:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9,
            'icp_lag_1': (14 + 13) / 2,  # Mean of measurements in 03:00-04:00
            'icp_lag_2': (15 + 16 + 14) / 3,  # Mean of measurements in 02:00-03:00
            'heart_rate_lag_1': (87 + 86) / 2,
            'heart_rate_lag_2': (85 + 88 + 83) / 3,
            'temperature_lag_1': (36.8 + 36.7) / 2,
            'temperature_lag_2': (36.7 + 36.9 + 36.6) / 3
        })

        # Create lagged features
        result = self.processor.create_lagged_features_for_row(current_row, previous_rows, ['icp', 'heart_rate', 'temperature'], hours=2)

        # Convert result to Series for comparison
        result_series = pd.Series(result)

        # Assert that the resulting Series matches the expected output
        assert_series_equal(result_series, expected_output)

    def test_create_lagged_features_for_row2(self):
        # Define the current row
        current_row = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 04:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9
        })

        # Define the previous rows with multiple measurements within the same hour
        previous_rows = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 03:45:00',  # Within hour 1
                '2023-01-01 03:15:00',  # Within hour 1
                '2023-01-01 02:50:00',  # Within hour 2
                '2023-01-01 02:10:00',  # Within hour 2
                '2023-01-01 02:00:00',  # Within hour 2
                '2023-01-01 01:30:00',  # Within hour 3 (should not be used since we only go back 2 hours)
                '2023-01-01 01:10:00',  # Within hour 3 (should not be used)
                '2023-01-01 00:50:00'   # Within hour 4 (should not be used)
            ]),
            'icp': [14, 13, 15, 16, 14, 12, 11, 10],
            'heart_rate': [87, 86, 85, 88, 83, 82, 81, 80],
            'temperature': [36.8, 36.7, 36.7, 36.9, 36.6, 36.6, 36.5, 36.5]
        })

        expected_output = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 04:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9,
            'icp_lag_1': (14 + 13) / 2,  # Mean of measurements in 03:00-04:00
            'icp_lag_2': (15 + 16 + 14) / 3,  # Mean of measurements in 02:00-03:00
            'heart_rate_lag_1': (87 + 86) / 2,
            'heart_rate_lag_2': (85 + 88 + 83) / 3,
            'temperature_lag_1': (36.8 + 36.7) / 2,
            'temperature_lag_2': (36.7 + 36.9 + 36.6) / 3
        })

        # Create lagged features
        result = self.processor.create_lagged_features_for_row(current_row, previous_rows, ['icp', 'heart_rate', 'temperature'], hours=2)

        # Convert result to Series for comparison
        result_series = pd.Series(result)

        # Assert that the resulting Series matches the expected output
        assert_series_equal(result_series, expected_output)

    def test_create_lagged_features_for_row_5_hours(self):
        # Define the current row
        current_row = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 06:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9
        })

        # Define the previous rows with multiple measurements within the same hour
        previous_rows = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 05:45:00',  # Within hour 1
                '2023-01-01 05:15:00',  # Within hour 1
                '2023-01-01 04:50:00',  # Within hour 2
                '2023-01-01 04:10:00',  # Within hour 2
                '2023-01-01 03:50:00',  # Within hour 3
                '2023-01-01 03:10:00',  # Within hour 3
                '2023-01-01 02:50:00',  # Within hour 4
                '2023-01-01 02:10:00',  # Within hour 4
                '2023-01-01 01:50:00',  # Within hour 5
                '2023-01-01 01:10:00',  # Within hour 5
                '2023-01-01 00:50:00',  # Outside the 5-hour window
                '2023-01-01 00:10:00'   # Outside the 5-hour window
            ]),
            'icp': [14, 13, 15, 16, 14, 12, 11, 10, 9, 8, 7, 6],
            'heart_rate': [87, 86, 85, 88, 83, 82, 81, 80, 79, 78, 77, 76],
            'temperature': [36.8, 36.7, 36.7, 36.9, 36.6, 36.6, 36.5, 36.5, 36.4, 36.3, 36.2, 36.1]
        })

        expected_output = pd.Series({
            'patient_id': 1,
            'timestamp': pd.Timestamp('2023-01-01 06:00:00'),
            'icp': 14,
            'heart_rate': 84,
            'temperature': 36.9,
            'icp_lag_1': (14 + 13) / 2,  # Mean of measurements in 05:00-06:00
            'icp_lag_2': (15 + 16) / 2,  # Mean of measurements in 04:00-05:00
            'icp_lag_3': (14 + 12) / 2,  # Mean of measurements in 03:00-04:00
            'icp_lag_4': (11 + 10) / 2,  # Mean of measurements in 02:00-03:00
            'icp_lag_5': (9 + 8) / 2,    # Mean of measurements in 01:00-02:00
            'heart_rate_lag_1': (87 + 86) / 2,
            'heart_rate_lag_2': (85 + 88) / 2,
            'heart_rate_lag_3': (83 + 82) / 2,
            'heart_rate_lag_4': (81 + 80) / 2,
            'heart_rate_lag_5': (79 + 78) / 2,
            'temperature_lag_1': (36.8 + 36.7) / 2,
            'temperature_lag_2': (36.7 + 36.9) / 2,
            'temperature_lag_3': (36.6 + 36.6) / 2,
            'temperature_lag_4': (36.5 + 36.5) / 2,
            'temperature_lag_5': (36.4 + 36.3) / 2
        })

        # Create lagged features
        result = self.processor.create_lagged_features_for_row(current_row, previous_rows, ['icp', 'heart_rate', 'temperature'], hours=5)

        # Convert result to Series for comparison
        result_series = pd.Series(result)

        # Assert that the resulting Series matches the expected output
        assert_series_equal(result_series, expected_output)


if __name__ == '__main__':
    unittest.main()
