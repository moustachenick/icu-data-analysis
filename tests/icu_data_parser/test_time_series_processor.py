import os
import unittest
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal

# Adjust the Python path to include the src directory
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from icu_data_parser.TimeSeriesProcessor import TimeSeriesProcessor


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

    def test_lag_feature_creation(self):
        """
        Test the create_lagged_features_dataset method with a small and easily verifiable dataset.
        """
        # Define small test input dataset
        small_test_data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 00:00:00',
                '2023-01-01 01:00:00',
                '2023-01-01 02:00:00',
                '2023-01-01 03:00:00',
                '2023-01-01 04:00:00'
            ]),
            'icp': [10, 12, 15, 14, 13],
            'heart_rate': [80, 82, 85, 87, 84],
            'temperature': [36.5, 36.6, 36.7, 36.8, 36.9]
        })

        # Generate corrected expected output based on available past rows
        expected_output = small_test_data.copy()

        # Manually create lagged values based on actual past rows available within hours=2
        expected_output['icp_lag_1'] = [None, 10, 12, 15, 14]
        expected_output['icp_lag_2'] = [None, None, 10, 12, 15]
        expected_output['heart_rate_lag_1'] = [None, 80, 82, 85, 87]
        expected_output['heart_rate_lag_2'] = [None, None, 80, 82, 85]
        expected_output['temperature_lag_1'] = [None, 36.5, 36.6, 36.7, 36.8]
        expected_output['temperature_lag_2'] = [None, None, 36.5, 36.6, 36.7]

        # Process the input data and create the lag features
        result = self.processor.create_lagged_features_dataset(small_test_data, hours=2,
                                                               columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)

        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result[expected_output.columns], expected_output, check_index_type=False)

    def test_lag_feature_creation_mixed_intervals(self):
        """
        Test the create_lagged_features_dataset method with a small and easily verifiable dataset.
        """
        # Define small test input dataset with mixed intervals (1-hour and 30-minutes)
        mixed_interval_data = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1],
            'timestamp': pd.to_datetime([
                '2023-01-01 00:00:00',
                '2023-01-01 01:00:00',
                '2023-01-01 01:30:00',
                '2023-01-01 02:00:00',
                '2023-01-01 03:00:00',
                '2023-01-01 03:30:00',
                '2023-01-01 04:00:00'
            ]),
            'icp': [10, 12, 14, 15, 13, 16, 14],
            'heart_rate': [80, 82, 83, 85, 87, 88, 84],
            'temperature': [36.5, 36.6, 36.55, 36.7, 36.8, 36.75, 36.9]
        })

        # Expected output with appropriate lags
        expected_output = mixed_interval_data.copy()
        expected_output['icp_lag_1'] = [None, 10, 12, 14, 15, 13, 16]
        expected_output['icp_lag_2'] = [None, None, 10, 12, 14, 15, 13]
        expected_output['heart_rate_lag_1'] = [None, 80, 82, 83, 85, 87, 88]
        expected_output['heart_rate_lag_2'] = [None, None, 80, 82, 83, 85, 87]
        expected_output['temperature_lag_1'] = [None, 36.5, 36.6, 36.55, 36.7, 36.8, 36.75]
        expected_output['temperature_lag_2'] = [None, None, 36.5, 36.6, 36.55, 36.7, 36.8]

        # Process the input data and create lag features using create_lagged_features_dataset
        result = self.processor.create_lagged_features_dataset(mixed_interval_data, hours=2,
                                                               columns_to_lag=['icp', 'heart_rate', 'temperature'])

        # Print debug information about generated columns
        print("Generated Columns:", result.columns.tolist())

        # Check the max lag created
        max_lag = max([int(col.split('_')[-1]) for col in result.columns if 'lag' in col], default=0)

        # Reset the index for both result and expected_output to ignore any index differences
        result.reset_index(drop=True, inplace=True)
        expected_output.reset_index(drop=True, inplace=True)

        # Assert that the resulting DataFrame matches the expected output
        assert_frame_equal(result[expected_output.columns], expected_output, check_index_type=False)

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

    @patch('builtins.input', side_effect=['y', 'y'])
    def test_process_data_bigger_subset(self, mock_input):
        """
        Test the process_data method with multiple input/expected output datasets.
        """
        test_cases = [
            ('input_bigger_subset.csv', 'expected_output_bigger_subset.csv'),
        ]

        print(f"Test cases: {test_cases}")

        for input_file, expected_output_file in test_cases:

            # Load input and expected output data
            input_data_path = os.path.join(self.datasets_dir, input_file)
            expected_output_path = os.path.join(self.datasets_dir, expected_output_file)

            input_data = pd.read_csv(input_data_path, parse_dates=['timestamp'], index_col=False)
            expected_output = pd.read_csv(expected_output_path, parse_dates=['timestamp'], index_col=False)

            # Process the input data and create lag features
            result = self.processor.process_data(input_data, hours=2,
                                                 columns_to_lag=['icp', 'heart_rate', 'temperature'])

            # print the result columns
            print(f"Result columns: {result.columns}")

            # Reset the index for both result and expected_output to ignore any index differences
            result.reset_index(drop=True, inplace=True)

            expected_output.reset_index(drop=True, inplace=True)

            # Assert that the resulting DataFrame matches the expected output
            assert_frame_equal(result, expected_output, check_index_type=False)


if __name__ == '__main__':
    unittest.main()
