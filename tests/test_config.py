import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from helper.config import load_config


class TestPathologyOverride(unittest.TestCase):
    """--pathology overrides parsing.filter_by_pathology; '' means all pathologies."""

    def test_cli_pathology_overrides_the_config_file(self):
        self.assertEqual(load_config(cli_pathology="TBI").filter_by_pathology, "TBI")
        self.assertEqual(load_config(cli_pathology="").filter_by_pathology, "")


if __name__ == "__main__":
    unittest.main()
