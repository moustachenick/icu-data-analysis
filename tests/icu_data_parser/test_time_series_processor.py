import os
import unittest
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

        # Create sample data for inline tests
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
        Test the process_data method to ensure the correct output is generated.
        """
        # Load the input data and expected output from CSV files
        input_data_path = os.path.join(self.datasets_dir, 'input_data.csv')
        expected_output_path = os.path.join(self.datasets_dir, 'expected_output_for_process_data.csv')

        input_data = pd.read_csv(input_data_path, parse_dates=['timestamp'], index_col=False)
        expected_output = pd.read_csv(expected_output_path, parse_dates=['timestamp'], index_col=False)

        # Process the input data and create the lag features
        result = self.processor.process_data(input_data, lags=2, columns_to_lag=['icp', 'heart_rate', 'temperature'])

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
        result = self.processor.create_lagged_features_dataset(input_data, lags=2, columns_to_lag=['icp', 'heart_rate', 'temperature'])

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

    # Additional test methods can follow the same pattern for handling file paths


if __name__ == '__main__':
    unittest.main()
