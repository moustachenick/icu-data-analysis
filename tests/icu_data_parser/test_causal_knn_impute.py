import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_parser.data_pre_processor import DataPreProcessor


class _StubConfig:
    """Minimal stand-in; causal_knn_impute does not read any config fields."""
    drop_lagged_null_rows = True


def _row(pid, ts, f1, f2):
    return {
        "patient_id": pid,
        "date_of_birth": "1980-01-01",
        "timestamp": ts,
        "f1": f1,
        "f2": f2,
    }


class TestCausalKnnImpute(unittest.TestCase):
    """
    Distance is computed over the *present* feature (f2); the missing feature (f1) is
    copied from the nearest eligible donor. Donors must be (a) from the TRAIN split only
    and (b) never timestamped after the target row.
    """

    def setUp(self):
        self.pre = DataPreProcessor("dummy_path.csv", _StubConfig())

        # Train donors (complete rows) + one train target with a single missing feature.
        self.train_df = pd.DataFrame([
            _row("T1", "2020-01-01 10:00:00", 5.0, 0.0),    # D1 (past, f2=0)
            _row("T1", "2020-01-01 12:00:00", 99.0, 1.0),   # D2 (FUTURE vs target, f2=1 -> closest if allowed)
            _row("T2", "2020-01-01 09:00:00", 7.0, 10.0),   # D3 (past, f2=10)
            _row("T1", "2020-01-01 11:00:00", np.nan, 1.0),  # TT: target, missing f1, f2=1
        ])
        # Validation target: nearest past train donor is D1 (f2=0).
        self.val_df = pd.DataFrame([
            _row("V1", "2020-01-01 11:00:00", np.nan, 0.0),  # VT
        ])
        # Test target + a complete test row that would be the closest donor IF test rows
        # were allowed as donors (they must not be).
        self.test_df = pd.DataFrame([
            _row("Z1", "2020-01-01 11:00:00", np.nan, 10.0),  # ZT: target, f2=10
            _row("Z1", "2020-01-01 10:00:00", 123.0, 10.0),   # ZD: complete TEST row, same f2
        ])

    @staticmethod
    def _f1(df, pid, ts):
        return df.loc[(df["patient_id"] == pid) & (df["timestamp"] == ts), "f1"].iloc[0]

    def test_causal_and_leakfree_imputation(self):
        train, val, test = self.pre.causal_knn_impute(self.train_df, self.val_df, self.test_df)

        # (a) No future donor: target TT at 11:00 must take D1's f1 (5.0), not the
        #     future D2's f1 (99.0), even though D2 is closer in the present feature.
        self.assertEqual(self._f1(train, "T1", "2020-01-01 11:00:00"), 5.0)

        # (b) No cross-split donor: test target ZT must be filled from a TRAIN donor
        #     (D3 -> 7.0), never from the complete TEST row ZD (123.0).
        self.assertEqual(self._f1(test, "Z1", "2020-01-01 11:00:00"), 7.0)

        # Validation row imputed from the nearest past train donor (D1 -> 5.0).
        self.assertEqual(self._f1(val, "V1", "2020-01-01 11:00:00"), 5.0)

        # No NaNs should remain in the imputed targets.
        self.assertFalse(train["f1"].isna().any())
        self.assertFalse(val["f1"].isna().any())
        self.assertFalse(test["f1"].isna().any())

    def test_rows_with_multiple_missing_are_dropped(self):
        # A row missing both features cannot be imputed and must be dropped.
        train_df = pd.concat([
            self.train_df,
            pd.DataFrame([_row("T1", "2020-01-01 13:00:00", np.nan, np.nan)]),
        ], ignore_index=True)

        train, _, _ = self.pre.causal_knn_impute(train_df, self.val_df, self.test_df)
        match = train[(train["patient_id"] == "T1") & (train["timestamp"] == "2020-01-01 13:00:00")]
        self.assertTrue(match.empty)


if __name__ == "__main__":
    unittest.main()
