import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import generate_descriptive_stats as stats


class TestAddExcludedVariables(unittest.TestCase):
    """PEEP is excluded during cleaning, but the descriptive statistics still report it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw_path = os.path.join(self.tmp.name, "final_data.csv")
        pd.DataFrame({
            "patient_id": [1, 1, 1, 1, 2],
            "timestamp": ["2020-01-01 10:00:00", "2020-01-01 10:30:00", "2020-01-01 11:00:00",
                          "2020-01-01 11:30:00", "2020-01-01 10:00:00"],
            "peep": [5.0, -1.0, 0.0, 45.0, 8.0],
        }).to_csv(self.raw_path, index=False)
        self.original_path = stats.RAW_DATA_PATH
        stats.RAW_DATA_PATH = self.raw_path

    def tearDown(self):
        stats.RAW_DATA_PATH = self.original_path
        self.tmp.cleanup()

    def test_peep_is_added_for_cleaned_rows_with_pipeline_cleaning(self):
        cleaned = pd.DataFrame({
            "patient_id": [1, 1, 1, 1],
            "timestamp": ["2020-01-01 10:00:00", "2020-01-01 10:30:00", "2020-01-01 11:00:00",
                          "2020-01-01 11:30:00"],
            "icp": [10.0, 11.0, 12.0, 13.0],
        })

        result = stats.add_excluded_variables(cleaned)

        self.assertEqual(len(result), 4)  # only the cleaned rows; patient 2 is not added
        self.assertEqual(result.loc[0, "peep"], 5.0)
        self.assertTrue(np.isnan(result.loc[1, "peep"]))  # -1 sentinel
        self.assertEqual(result.loc[2, "peep"], 0.0)  # PEEP 0 is a valid value
        self.assertTrue(np.isnan(result.loc[3, "peep"]))  # outside the physiologic range

    def test_existing_column_is_not_replaced(self):
        cleaned = pd.DataFrame({"patient_id": [1], "timestamp": ["2020-01-01 10:00:00"], "peep": [7.0]})

        result = stats.add_excluded_variables(cleaned)

        self.assertEqual(result.loc[0, "peep"], 7.0)


if __name__ == "__main__":
    unittest.main()
