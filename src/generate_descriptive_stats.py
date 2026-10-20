"""
Generate descriptive statistics for the classification dataset.

Prints a formatted console table and a markdown table.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_parser.data_pre_processor import DataPreProcessor
from helper.data_frame_printer import DataFramePrinter

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_df_classification.csv")
# The cleaned data no longer contains the excluded variables (PEEP); they are read from here.
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "final_data.csv")

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
    "peep":                 "PEEP (dropped)",
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

AGE_BINS   = [0, 18, 35, 45, 55, 65, 75, 85, float("inf")]
AGE_LABELS = ["<18", "18–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]


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


def patient_summary(df: pd.DataFrame) -> None:
    if "patient_id" not in df.columns or "date_of_birth" not in df.columns or "timestamp" not in df.columns:
        return

    # One row per patient; use first measurement timestamp as admission date proxy
    first_ts = (
        pd.to_datetime(df["timestamp"])
        .groupby(df["patient_id"])
        .min()
        .rename("admission_ts")
    )
    patients = (
        df.drop_duplicates(subset="patient_id")
        .set_index("patient_id")[["date_of_birth"]]
        .join(first_ts)
    )
    patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"])
    patients["age"] = (patients["admission_ts"] - patients["date_of_birth"]).dt.days / 365.25

    n = len(patients)
    patients["age_group"] = pd.cut(patients["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    counts = patients["age_group"].value_counts().reindex(AGE_LABELS, fill_value=0)

    rows = [
        {"Age Group": label, "N": count, "%": f"{100 * count / n:.1f}%"}
        for label, count in counts.items()
    ]
    age_df = pd.DataFrame(rows)
    DataFramePrinter.print_dataframe_tabulated(age_df, title="Patient Demographics — Age at Admission")
    print()


def add_excluded_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the variables that cleaning excludes from the analysis (PEEP), for the rows of the
    cleaned dataset only, so that the descriptive statistics describe the analysed rows.
    The values get the cleaning the pipeline applied before the exclusion: the -1 sentinel
    becomes NaN, and values outside PHYSIOLOGIC_RANGES become NaN.
    """
    excluded = [c for c in DataPreProcessor.EXCLUDED_VARIABLES if c not in df.columns]
    if not excluded:
        return df

    raw = pd.read_csv(RAW_DATA_PATH, usecols=["patient_id", "timestamp"] + excluded)
    for col in excluded:
        values = pd.to_numeric(raw[col], errors="coerce").replace(-1, np.nan)
        if col in DataPreProcessor.PHYSIOLOGIC_RANGES:
            low, high = DataPreProcessor.PHYSIOLOGIC_RANGES[col]
            values = values.where(values.between(low, high))
        raw[col] = values

    merged = df.merge(raw, on=["patient_id", "timestamp"], how="left", validate="one_to_one")
    print(f"Added excluded variables {excluded} from {os.path.basename(RAW_DATA_PATH)} "
          f"for {len(merged):,} cleaned rows.")
    return merged


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
            "Mean": round(series.mean(), 2),
            "Std": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "Missing %": f"{missing_pct:.2f}%",
        })

    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading data from: {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    cohort_summary(df)
    patient_summary(df)

    stats_df = build_stats_df(add_excluded_variables(df))

    DataFramePrinter.print_dataframe_tabulated(stats_df, title="Descriptive Statistics — Clinical Variables")


if __name__ == "__main__":
    main()
