import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_parser.data_pre_processor import DataPreProcessor


class _StubConfig:
    """Minimal stand-in; clean_physiologic_outliers does not read any config fields."""
    drop_high_missing_columns = True


class TestCleanPhysiologicOutliers(unittest.TestCase):
    """
    Physiologically impossible measurements are transcription errors and must be blanked to
    NaN so the existing missing-value handling (row drop / causal KNN impute) deals with them.
    The motivating case is pH recorded as 743 instead of 7.43.
    """

    def setUp(self):
        self.pre = DataPreProcessor("dummy_path.csv", _StubConfig())

    @staticmethod
    def _row(pid, ts, **values):
        row = {"patient_id": pid, "timestamp": ts}
        row.update(values)
        return row

    def test_dropped_decimal_point_ph_is_blanked(self):
        """The real bug: pH 743 (i.e. 7.43 with a lost decimal) becomes NaN."""
        df = pd.DataFrame([
            self._row(388, "2020-01-01 10:00:00", ph=743.0),
            self._row(388, "2020-01-01 11:00:00", ph=7.43),
        ])

        result = self.pre.clean_physiologic_outliers(df)

        self.assertTrue(np.isnan(result.loc[0, "ph"]))
        self.assertEqual(result.loc[1, "ph"], 7.43)

    def test_in_range_values_are_untouched(self):
        """Clinically extreme but plausible values must survive — the bounds catch typos only."""
        df = pd.DataFrame([
            self._row(1, "2020-01-01 10:00:00",
                      ph=6.9, temperature=34.5, pao2=520.0, paco2=144.0,
                      mean_blood_pressure=45.0, heart_rate=180.0, cpp=-40.0),
        ])
        original = df.copy()

        result = self.pre.clean_physiologic_outliers(df)

        pd.testing.assert_frame_equal(result, original)

    def test_peep_zero_survives_as_valid_clinical_value(self):
        """PEEP uses 0 as a valid value, not a missing sentinel (see CLAUDE.md)."""
        df = pd.DataFrame([
            self._row(1, "2020-01-01 10:00:00", peep=0.0),
            self._row(1, "2020-01-01 11:00:00", peep=-19.0),
        ])

        result = self.pre.clean_physiologic_outliers(df)

        self.assertEqual(result.loc[0, "peep"], 0.0)
        self.assertTrue(np.isnan(result.loc[1, "peep"]))

    def test_existing_nans_are_preserved_and_not_double_counted(self):
        """Already-missing values stay missing; NaN comparisons must not flag them."""
        df = pd.DataFrame([
            self._row(1, "2020-01-01 10:00:00", ph=np.nan),
        ])

        result = self.pre.clean_physiologic_outliers(df)

        self.assertTrue(np.isnan(result.loc[0, "ph"]))

    def test_missing_columns_are_skipped(self):
        """Frames without a bounded column (e.g. respiration_rate already dropped) are fine."""
        df = pd.DataFrame([self._row(1, "2020-01-01 10:00:00", icp=12.0)])

        result = self.pre.clean_physiologic_outliers(df)

        self.assertEqual(result.loc[0, "icp"], 12.0)

    def test_bounds_are_inclusive(self):
        """Values exactly on a bound are legitimate and must be kept."""
        df = pd.DataFrame([
            self._row(1, "2020-01-01 10:00:00", ph=6.5, temperature=45.0),
        ])

        result = self.pre.clean_physiologic_outliers(df)

        self.assertEqual(result.loc[0, "ph"], 6.5)
        self.assertEqual(result.loc[0, "temperature"], 45.0)


if __name__ == "__main__":
    unittest.main()
