"""
Generate descriptive statistics for the classification dataset.

Prints a formatted console table and a markdown table for copy-paste into the manuscript.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from helper.data_frame_printer import DataFramePrinter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_df_lagged_classification.csv")

CLINICAL_COLUMNS = [
    "icp",
    "cpp",
    "glucose",
    "haemoglobin",
    "heart_rate",
    "mean_blood_pressure",
    "paco2",
    "pao2",
    "peep",
    "ph",
    "spo2",
    "temperature",
    "respiration_rate",
]

DISPLAY_NAMES = {
    "icp":                  "ICP",
    "cpp":                  "CPP",
    "glucose":              "Glucose",
    "haemoglobin":          "Haemoglobin",
    "heart_rate":           "Heart Rate",
    "mean_blood_pressure":  "Mean Blood Pressure (MAP)",
    "paco2":                "PaCO₂",
    "pao2":                 "PaO₂",
    "peep":                 "PEEP",
    "ph":                   "pH",
    "spo2":                 "SpO₂",
    "temperature":          "Temperature",
    "respiration_rate":     "Respiration Rate (dropped)",
}

UNITS = {
    "icp":                  "mmHg",
    "cpp":                  "mmHg",
    "glucose":              "mg/dL",
    "haemoglobin":          "g/dL",
    "heart_rate":           "bpm",
    "mean_blood_pressure":  "mmHg",
    "paco2":                "mmHg",
    "pao2":                 "mmHg",
    "peep":                 "cmH₂O",
    "ph":                   "—",
    "spo2":                 "%",
    "temperature":          "°C",
    "respiration_rate":     "breaths/min",
}

NORMAL_RANGES = {
    "icp":                  "0–10",
    "cpp":                  "50–70",
    "glucose":              "70–140",
    "haemoglobin":          "12–17",
    "heart_rate":           "60–100",
    "mean_blood_pressure":  "70–100",
    "paco2":                "35–45",
    "pao2":                 "75–100",
    "peep":                 "0–5",
    "ph":                   "7.35–7.45",
    "spo2":                 "95–100",
    "temperature":          "36.5–37.5",
    "respiration_rate":     "12–20",
}


def cohort_summary(df: pd.DataFrame) -> None:
    n_obs = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else "N/A"

    if "patient_id" in df.columns:
        obs_per_patient = df.groupby("patient_id").size()
        median_obs = obs_per_patient.median()
        q1 = obs_per_patient.quantile(0.25)
        q3 = obs_per_patient.quantile(0.75)
        iqr_str = f"{q1:.0f}–{q3:.0f}"
    else:
        median_obs = iqr_str = "N/A"

    print("\nCohort Summary")
    print("=" * 40)
    print(f"  Total observations : {n_obs:,}")
    print(f"  Unique patients    : {n_patients}")
    if isinstance(median_obs, float):
        print(f"  Obs/patient (median [IQR]): {median_obs:.0f} [{iqr_str}]")
    print()


def build_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    total_rows = len(df)

    rows = []
    for col in present_cols:
        series = df[col]
        missing_n = series.isna().sum()
        missing_pct = 100.0 * missing_n / total_rows
        rows.append({
            "Variable": DISPLAY_NAMES[col],
            "Unit": UNITS[col],
            "Normal Range": NORMAL_RANGES[col],
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "Min": round(series.min(), 2),
            "Max": round(series.max(), 2),
            "Missing %": f"{missing_pct:.2f}%",
        })

    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading data from: {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    cohort_summary(df)

    stats_df = build_stats_df(df)

    DataFramePrinter.print_dataframe_tabulated(stats_df, title="Descriptive Statistics — Clinical Variables")


if __name__ == "__main__":
    main()
