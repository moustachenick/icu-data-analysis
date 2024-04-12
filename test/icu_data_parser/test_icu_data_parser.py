import unittest
from unittest.mock import Mock

from src.icu_data_parser.icu_data_parser import ICUDataParser


class TestICUDataParser(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.parser = ICUDataParser(self.logger)

    def test_add_data_value_for_new_patient(self):
        self.parser.add_data_value_for_patient_at_timestamp("patient_1", "timestamp_1", "cpp", 10)
        self.assertEqual(self.parser.final_data[1],
                         ["patient_1", "timestamp_1", 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    def test_add_data_value_for_existing_patient(self):
        self.parser.final_data.append(["patient_1", "timestamp_1", -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.parser.add_data_value_for_patient_at_timestamp("patient_1", "timestamp_1", "cpp", 10)
        self.assertEqual(self.parser.final_data[1],
                         ["patient_1", "timestamp_1", 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    def test_parse_file_with_valid_data(self):
        with open('test_file.txt', 'w') as f:
            f.write('"patient_1","timestamp_1","cpp","10"\n')
        self.parser.parse_file('test_file.txt', 'cpp')
        self.assertListEqual(self.parser.final_data[1],
                         ["patient_1", "timestamp_1", "10", -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    def test_save_final_data(self):
        self.parser.final_data.append(["patient_1", "timestamp_1", 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.parser.save_final_data('test_output.csv')
        with open('test_output.csv', 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[1].strip(),
                         'patient_1,timestamp_1,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1')


if __name__ == '__main__':
    unittest.main()
