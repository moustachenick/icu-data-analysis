import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate

from data_parser.time_series_processor import TimeSeriesProcessor


class DataPreProcessor:

    PATHOLOGIES_GROUPED_AS_OTHER = {
        "cns infection",
        "assdh",
        "status epilepticus",
        "intracranial hypertension due to acute leukemia",
        "hydrocephalus",
    }

    def __init__(self, raw_data_file_path, config):
        self.raw_data_file_path = raw_data_file_path
        # Run configuration (see helper.config.AppConfig) drives the preprocessing steps
        # that used to be interactive prompts.
        self.config = config

    def pre_process_dataset(self, hours, mode):
        """
        Clean the dataset (STEPS 1-6 only). Imputation and lagged-feature creation are
        deliberately NOT done here: they run AFTER the patient-level train/val/test split
        (see main.py) so that imputation can be fit on the training split only and remain
        causal. This eliminates cross-split and temporal leakage.

        :param hours: Kept for signature compatibility; lagging now happens post-split.
        :param mode: "regression" or "classification" (drives the cache file name).
        :return: The cleaned (pre-imputation, pre-lag) DataFrame.
        """

        cleaned_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", f"cleaned_df_{mode}.csv")
        )

        # if the file already exists, return the dataframe
        if os.path.exists(cleaned_file_path):
            print(f"File {cleaned_file_path} already exists. Loading the cleaned data from the file.")
            return pd.read_csv(cleaned_file_path)

        # Construct the absolute path
        raw_data_file_path = self.raw_data_file_path
        # Read the data from the CSV file
        df = pd.read_csv(raw_data_file_path, engine='python')

        df = self.add_pathology_one_hot_features(df)

        df = self.transform_all_columns_to_float(df)

        print("\n~~~~ STEP 1: Deleting rows with negative ICP values ~~~~\n")
        df = self.delete_negative_icp_values(df)

        print("\n~~~~ STEP 2: Deleting patients with 2 rows or less ~~~~\n")
        # Delete patients with 2 rows or fewer
        df = df.groupby('patient_id').filter(lambda x: len(x) > 2)

        print("\n~~~~ STEP 3: Dropping the rows that have null \"ICP next\" column ~~~~\n")
        df = self.drop_rows_with_null_target_column(df)

        print("\n~~~~ STEP 4: Standardizing missing values (converting 0 to Nan, etc) ~~~~\n")
        df = self.standardize_missing_values(df)

        if self.config.drop_high_missing_columns:
            print("\n~~~~ STEP 5: Dropping columns with high missing values ~~~~\n")
            df = self.drop_columns_with_high_missing_values(df)
            print(f"Number of rows in the dataset after the dropping: {df.shape[0]}")

        print("\n~~~~ STEP 6: Cleaning ICP outliers ~~~~\n")
        df = self.clean_icp_outliers(df)
        print(f"Number of rows in the dataset after the cleaning: {df.shape[0]}")

        print("\nCleaning complete (imputation + lagging happen after the split).")
        print(f"Number of rows in the cleaned dataset: {df.shape[0]}")

        # Save the cleaned (pre-imputation, pre-lag) data to a CSV file
        output_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')))
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        print(f"\nData cleaning completed. Saving cleaned data to {cleaned_file_path}\n")
        df.to_csv(cleaned_file_path, index=False)

        return df

    def add_pathology_one_hot_features(self, df):
        """
        Optionally enrich the dataframe with one-hot encoded pathology columns using the pathologies_filtered.csv file.
        Pathologies listed in PATHOLOGIES_GROUPED_AS_OTHER are grouped under a single "pathology_other" column.
        """
        if not self.config.add_pathology_one_hot:
            print("Skipping pathology one-hot encoding (preprocessing.add_pathology_one_hot = false).")
            return df

        pathologies_file = os.path.abspath(
            os.path.join(os.path.dirname(self.raw_data_file_path), "pathologies_filtered.csv")
        )

        try:
            pathologies_df = pd.read_csv(pathologies_file, dtype=str)
        except FileNotFoundError:
            print(f"Warning: pathologies_filtered.csv not found at {pathologies_file}. Skipping pathology one-hot encoding.")
            return df
        except Exception as e:
            print(f"Error reading {pathologies_file}: {e}. Skipping pathology one-hot encoding.")
            return df

        if pathologies_df.shape[1] < 2:
            print(f"Unexpected format for {pathologies_file}. Skipping pathology one-hot encoding.")
            return df

        patient_col = pathologies_df.columns[0]
        pathology_col = pathologies_df.columns[1]

        pathologies_df[patient_col] = pathologies_df[patient_col].astype(str).str.strip()
        pathologies_df[pathology_col] = pathologies_df[pathology_col].astype(str).str.strip()

        patient_to_pathology = dict(zip(pathologies_df[patient_col], pathologies_df[pathology_col]))

        # Determine all unique pathologies (case-insensitive) excluding those grouped as "other".
        unique_pathologies = {
            p for p in pathologies_df[pathology_col].dropna().unique()
            if p.strip() and p.lower() not in self.PATHOLOGIES_GROUPED_AS_OTHER
        }

        if not unique_pathologies and not self.PATHOLOGIES_GROUPED_AS_OTHER:
            print("No pathologies found to encode. Skipping pathology one-hot encoding.")
            return df

        print(f"Found {len(unique_pathologies)} pathologies to one-hot encode. Grouping the following as 'other': {', '.join(sorted(self.PATHOLOGIES_GROUPED_AS_OTHER))}")

        def sanitize(name):
            name = name.lower().strip().replace(" ", "_")
            name = re.sub(r"[^0-9a-zA-Z_]", "", name)
            return name

        enriched_df = df.copy()

        # Initialize all pathology columns to 0.
        for pathology in sorted(unique_pathologies):
            col_name = f"pathology_{sanitize(pathology)}"
            enriched_df[col_name] = 0

        other_col = "pathology_other"
        enriched_df[other_col] = 0

        # Assign one-hot values per patient based on mapping.
        for idx, patient_id in enriched_df["patient_id"].astype(str).items():
            pathology = patient_to_pathology.get(patient_id, "").strip()
            if not pathology:
                continue
            if pathology.lower() in self.PATHOLOGIES_GROUPED_AS_OTHER:
                enriched_df.at[idx, other_col] = 1
                continue
            col_name = f"pathology_{sanitize(pathology)}"
            if col_name in enriched_df.columns:
                enriched_df.at[idx, col_name] = 1
            else:
                # Pathology not anticipated when initializing; treat as other to avoid losing signal.
                enriched_df.at[idx, other_col] = 1

        return enriched_df

    def transform_all_columns_to_float(self, df):
        # Convert all columns to float, except for 'patient_id', 'date_of_birth', and 'timestamp'
        for col in df.columns:
            if col not in ['patient_id', 'date_of_birth', 'timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print("\nStatistics about the raw data after the transformation:\n")
        print(df.info())
        return df

    @staticmethod
    def print_missing_value_table(df: pd.DataFrame, title: str) -> None:
        """Print a per-column missing value count and percentage, skipping non-clinical columns."""
        skip_prefixes = ("pathology_",)
        skip_cols = {"patient_id", "date_of_birth", "timestamp"}
        total = len(df)
        rows = []
        for col in df.columns:
            if col in skip_cols or any(col.startswith(p) for p in skip_prefixes):
                continue
            n_missing = df[col].isna().sum()
            rows.append({
                "Variable": col,
                "Missing N": int(n_missing),
                "Missing %": f"{100.0 * n_missing / total:.2f}%",
            })
        print(f"\n{title}\n" + "=" * len(title))
        print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))
        print()

    @staticmethod
    def print_percentages_of_rows_with_missing_values(dataframe):
        # Function to print the percentage of rows that have more than 1 column with missing values
        # and the percentage of rows that have exactly 1 column with missing values.

        total_num_of_rows = dataframe.shape[0]
        number_of_rows_with_more_than_1_missing_value = dataframe[dataframe.isna().sum(axis=1) > 1].shape[0]
        print('Percentage of rows that have more than 1 column with missing values: ',
              round(number_of_rows_with_more_than_1_missing_value / total_num_of_rows * 100, 3), '%')

        number_of_rows_with_1_missing_value = dataframe[dataframe.isna().sum(axis=1) == 1].shape[0]
        print('Percentage of rows that have exactly 1 column with missing value: ',
              round(number_of_rows_with_1_missing_value / total_num_of_rows * 100, 3), '%')

    def standardize_missing_values(self, df):
        """
        The dataset contains missing values that are represented by different strings ("--", ".", etc.).
        We need to standardize these missing values by replacing them with NaN.
        Standardize missing values in the dataset. Replace missing values with NaN (np.nan).
        :param df:
        :return:
        """
        # declare an array of strings that will be converted to NaN
        missing_values_representations = [-1, '-1', '--', '-', '.', "/"]
        # Replace missing values with NaN
        df.replace(missing_values_representations, np.nan, inplace=True)
        # print the total number of rows
        total_num_of_rows = df.shape[0]
        print()
        print('Total number of rows: ', total_num_of_rows)
        print()
        print('Percentage of rows with missing values after the standardization:',
              self.print_percentages_of_rows_with_missing_values(df))
        self.print_missing_value_table(df, "Missing Values per Column — After Standardization")
        return df

    def drop_columns_with_high_missing_values(self, df):
        """
        Drop columns with a high percentage of missing values.
        :param df: DataFrame to process
        :return: DataFrame with columns dropped
        """
        self.print_missing_value_table(df, "Missing Values per Column — Before Dropping")
        rr_missing = df['respiration_rate'].isna().sum()
        rr_pct = 100.0 * rr_missing / len(df)
        print(f"Dropping 'respiration_rate': {rr_missing:,} missing rows out of {len(df):,} ({rr_pct:.2f}% null)")
        # drop the 'respiration_rate' column as it has many missing values, and is not needed for the analysis
        df.drop(columns=['respiration_rate'], inplace=True)
        print('\nDataFrame after dropping the "respiration_rate" column:')
        self.print_percentages_of_rows_with_missing_values(df)
        # Let's drop the rows with more than 1 columns with missing value
        missing_count_per_row = df.isnull().sum(axis=1)
        filtered_df = df[missing_count_per_row <= 1]
        print()
        print('DataFrame after dropping rows with more than 1 missing value:')
        self.print_percentages_of_rows_with_missing_values(filtered_df)
        self.print_missing_value_table(filtered_df, "Missing Values per Column — After Dropping")
        return filtered_df

    # Columns that are never used as imputation features (identifiers / timestamps).
    IGNORED_IMPUTE_COLS = ['patient_id', 'date_of_birth', 'timestamp']

    def causal_knn_impute(self, train_df, val_df, test_df):
        """
        Leak-free, causal 1-NN imputation for rows with exactly one missing feature.

        The donor pool is built from the TRAINING split only (rows with no missing
        features), so validation/test data never influences imputation (no cross-split
        leakage). For each target row at time ``t``, only donors with ``timestamp <= t``
        are eligible, so a future measurement can never fill a past row (no temporal
        leakage / "cheating"). Cross-patient donors remain allowed (Missing-At-Random
        assumption); a row's own future is excluded by the timestamp rule.

        Rows with more than one missing feature are dropped (matching the previous
        imputer, which only kept complete + single-missing rows).

        Returns ``(train, val, test)`` with the same columns, imputed in place.
        """
        feature_cols = [c for c in train_df.columns if c not in self.IGNORED_IMPUTE_COLS]

        # Build the donor pool from complete TRAIN rows only.
        donor_pool = train_df[train_df[feature_cols].notna().all(axis=1)]
        donor_matrix = donor_pool[feature_cols].to_numpy(dtype=float)
        donor_ts = pd.to_datetime(donor_pool['timestamp']).to_numpy()
        donor_pids = donor_pool['patient_id'].to_numpy()

        print(f"\nCausal KNN imputation: donor pool = {len(donor_pool):,} complete training rows.")

        if len(donor_pool) == 0:
            print("Warning: no complete training rows available as donors. Skipping imputation.")
            return train_df, val_df, test_df

        results = {}
        for split_name, df in (("train", train_df), ("validation", val_df), ("test", test_df)):
            imputed, stats = self._impute_split_causally(
                df, feature_cols, donor_matrix, donor_ts, donor_pids
            )
            results[split_name] = imputed
            self._print_impute_diagnostics(split_name, stats)

        return results["train"], results["validation"], results["test"]

    @staticmethod
    def _impute_split_causally(df, feature_cols, donor_matrix, donor_ts, donor_pids):
        """Impute one split against the (train-derived) donor pool. See causal_knn_impute."""
        df = df.copy()
        missing_counts = df[feature_cols].isna().sum(axis=1)

        # Drop rows with more than one missing feature (matches the previous imputer).
        keep_mask = missing_counts <= 1
        stats = {"n_imputed": 0, "same_patient": 0, "future_donor": 0,
                 "no_donor": 0, "dropped_multi": int((~keep_mask).sum())}
        df = df[keep_mask].copy()

        feat_values = df[feature_cols].to_numpy(dtype=float)
        target_ts = pd.to_datetime(df['timestamp']).to_numpy()
        target_pids = df['patient_id'].to_numpy()
        missing_per_row = np.isnan(feat_values).sum(axis=1)
        target_positions = np.where(missing_per_row == 1)[0]

        for pos in target_positions:
            row = feat_values[pos]
            missing_col = int(np.where(np.isnan(row))[0][0])
            t = target_ts[pos]

            # Eligible donors: timestamp <= t (strictly no future information).
            eligible = donor_ts <= t
            if not eligible.any():
                stats["no_donor"] += 1
                continue

            # Partial Euclidean distance over the features present in the target row.
            present_cols = [j for j in range(len(feature_cols)) if j != missing_col]
            target_vec = row[present_cols]
            donor_sub = donor_matrix[np.ix_(eligible, present_cols)]
            diff = donor_sub - target_vec
            sq_dist = np.einsum('ij,ij->i', diff, diff)

            eligible_idx = np.where(eligible)[0]
            donor_idx = int(eligible_idx[int(np.argmin(sq_dist))])

            feat_values[pos, missing_col] = donor_matrix[donor_idx, missing_col]

            stats["n_imputed"] += 1
            if donor_pids[donor_idx] == target_pids[pos]:
                stats["same_patient"] += 1
            if donor_ts[donor_idx] > t:
                stats["future_donor"] += 1  # must remain 0 by construction

        df[feature_cols] = feat_values
        return df, stats

    @staticmethod
    def _print_impute_diagnostics(split_name, stats):
        n = stats["n_imputed"]
        print(f"\nKNN Imputation Diagnostics — {split_name} split")
        print(f"  Rows dropped (>1 missing feature)         : {stats['dropped_multi']:,}")
        print(f"  Rows imputed (exactly 1 missing feature)  : {n:,}")
        if n:
            sp, fu = stats["same_patient"], stats["future_donor"]
            print(f"  Same-patient donor                        : {sp:,} / {n:,} ({100 * sp / n:.1f}%)")
            print(f"  Future-timestamp donor                    : {fu:,} / {n:,} ({100 * fu / n:.1f}%)  <- must be 0 (causal)")
        print(f"  Rows left unimputed (no eligible past donor): {stats['no_donor']:,}")

    def delete_negative_icp_values(self, combined_df):
        # Ensure 'icp' is a numeric column
        combined_df['icp'] = pd.to_numeric(combined_df['icp'], errors='coerce')

        # Count the number of negative 'icp' values
        negative_icp_count = (combined_df['icp'] < 0).sum()
        print(f"Number of negative ICP values: {negative_icp_count}")
        print("Total Rows before deletion =", len(combined_df))

        # Delete rows where 'icp' is negative
        cleaned_df = combined_df[combined_df['icp'] >= 0]

        # Print the updated DataFrame details
        print("Total Rows after deleting negative ICP values =", len(cleaned_df))

        return cleaned_df

    def clean_icp_outliers(self, cleaned_df):
        # Detect and remove outliers in the 'icp' column using the Z-score method
        # Calculate the Z-scores for each value in the 'icp' column
        z_scores = (cleaned_df['icp'] - cleaned_df['icp'].mean()) / cleaned_df['icp'].std()

        # Define a threshold for the Z-scores (how many standard deviations away from the mean)
        # We choose a threshold of 7, which is accepted for this domain
        z_score_threshold = 7

        # Identify the outliers based on the Z-scores
        outliers = cleaned_df[abs(z_scores) > z_score_threshold]

        print(f"Number of outliers detected: {len(outliers)}")

        # Print the outliers
        print("Outliers ICP:")
        print(outliers[['patient_id', 'timestamp', 'icp']])

        # Remove the outliers from the DataFrame
        cleaned_df = cleaned_df[abs(z_scores) <= z_score_threshold]

        return cleaned_df

    def add_lagged_features(self, data, hours):
        """
        Create lagged time-series features for a single split.

        Called once per train/validation/test split (after the patient-level split and
        causal imputation). Lags are computed within each patient, so applying this
        per-split is equivalent to doing it on the whole frame — patients are disjoint
        across splits.

        :param data: DataFrame (one split) to process.
        :param hours: Number of hours to use for creating lag features.
        """

        # Create lagged features
        columns_to_lag = [
            "icp", "temperature", "mean_blood_pressure", "cpp", "glucose",
            "haemoglobin", "heart_rate", "paco2", "pao2", "peep", "ph", "spo2"
        ]
        time_series_processor = TimeSeriesProcessor()
        data = time_series_processor.process_data(
            data, hours=hours, columns_to_lag=columns_to_lag,
            drop_lagged_null_rows=self.config.drop_lagged_null_rows,
        )

        print(f"\nColumns after creating lagged features: {data.columns}")

        return data

    def drop_rows_with_null_target_column(self, data):
        # Store the initial number of rows
        initial_row_count = data.shape[0]

        # Drop rows with null 'icp' values
        data = data.dropna(subset=['icp'])

        # Calculate the number of dropped rows
        dropped_row_count = initial_row_count - data.shape[0]

        print(f"Number of rows in the dataset after the dropping: {data.shape[0]}")
        print(f"Number of dropped rows: {dropped_row_count}")

        return data
