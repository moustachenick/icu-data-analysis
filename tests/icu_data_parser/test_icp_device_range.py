import os
import sys
import unittest

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_parser.data_pre_processor import DataPreProcessor


class _StubConfig:
    """Minimal stand-in; delete_icp_values_above_device_range does not read any config fields."""


class TestIcpDeviceRange(unittest.TestCase):
    """ICP above the validated device range (100 mmHg, AAMI) is out of scope and removed."""

    def setUp(self):
        self.pre = DataPreProcessor("dummy_path.csv", _StubConfig())

    def test_rows_above_100_are_removed_and_100_is_kept(self):
        df = pd.DataFrame({
            "patient_id": [1, 1, 2, 2],
            "icp": [0.0, 100.0, 100.5, 160.0],
        })

        result = self.pre.delete_icp_values_above_device_range(df)

        self.assertEqual(list(result["icp"]), [0.0, 100.0])


if __name__ == "__main__":
    unittest.main()
