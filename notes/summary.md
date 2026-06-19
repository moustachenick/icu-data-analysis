# ICU ICP Prediction Pipeline — Technical Summary

## Project Purpose

This project builds machine learning pipelines that predict **Intracranial Pressure (ICP)** in ICU patients using multivariate physiological time-series data. It supports two tasks:

- **Regression** — predict the next continuous ICP value (mmHg)
- **Classification** — predict whether ICP will be elevated (≥ 22 mmHg), a clinically significant threshold for intervention

Raw data comes from ICU monitoring devices stored as one `.txt` file per physiological variable. The pipeline merges these into a single dataset, cleans and imputes it, engineers lagged time-series features, and trains and evaluates multiple models.

---

## Repository Structure

```text
icu-data-analysis/
├── data/                              # All data files (source + generated)
│   ├── ICP.txt, CPP.txt, ...          # Raw ICU measurements (one file per variable)
│   ├── episodes with high icp.txt     # Source of date_of_birth per patient
│   ├── pathologies_filtered.csv       # Patient ID → pathology mapping
│   ├── final_data.csv                 # Generated: merged raw data
│   ├── cleaned_df_*.csv               # Generated: cleaned dataset (pre-impute, pre-lag)
│   ├── train_data_*.csv               # Generated: training split (imputed + lagged)
│   ├── validation_data_*.csv          # Generated: validation split (selection)
│   └── test_data_*.csv                # Generated: test split (final held-out)
├── src/
│   ├── main.py                        # Entry point and pipeline orchestrator
│   ├── generate_descriptive_stats.py  # Standalone script for cohort statistics
│   ├── data_parser/
│   │   ├── data_parser.py             # Parses raw .txt files into final_data.csv
│   │   ├── data_pre_processor.py      # 8-step cleaning + feature engineering
│   │   ├── binary_data_processor.py   # Converts ICP to binary label (≥22 mmHg)
│   │   └── time_series_processor.py   # Creates lagged features per patient
│   ├── classification/
│   │   ├── classification_pipeline.py        # Orchestrates all classifiers + evaluation
│   │   ├── xgboost_classification_predictor.py
│   │   ├── lagged_icp_baseline_predictor.py
│   │   └── latest_icp_baseline_predictor.py
│   ├── regression/
│   │   ├── regression_pipeline.py            # Orchestrates all regressors + evaluation
│   │   ├── regression.py                     # Abstract base class
│   │   ├── baseline_history_regression.py
│   │   ├── baseline_mean_regression.py
│   │   ├── baseline_time_window_mean_regression.py
│   │   └── regression_model_plotter.py
│   ├── helper/
│   │   └── data_frame_printer.py      # Pretty-print DataFrames via tabulate
│   ├── test_debugging_scripts/        # Developer scratch scripts (not production)
│   ├── icu_data_parser/               # Legacy — empty, superseded by data_parser/
│   └── icu_data_regression_classes/   # Legacy — empty, superseded by regression/
└── tests/                             # Pytest test suite
```

---

## End-to-End Data Flow

```
Raw ICU .txt files (one per variable: ICP, CPP, Glucose, ...)
        │
        ▼
 DataParser.run()                           [data_parser/data_parser.py]
   ├─ parse()            — reads each file, rounds timestamps to 30-min grid,
   │                        deduplicates via running average (Welford's method)
   ├─ create_final_data() — pivots nested map to one row per (patient, timestamp)
   ├─ filter_instances() — interactive: user can filter to a single pathology
   ├─ fix_missing_values()— forward-fill within 30-min window (certain variables
   │                        forward-fill within same calendar day)
   └─ → data/final_data.csv
        │
        ▼
 DataPreProcessor.pre_process_dataset()     [data_parser/data_pre_processor.py]
   Step 1: add_pathology_one_hot_features() — optional; reads pathologies_filtered.csv
   Step 2: transform_all_columns_to_float()
   Step 3: delete_negative_icp_values()
   Step 4: drop patients with ≤2 rows
   Step 5: drop_rows_with_null_target_column() — drop null ICP rows
   Step 6: standardize_missing_values()     — [-1, '--', '.', '/'] → NaN
   Step 7: drop_columns_with_high_missing_values() — drop respiration_rate,
           then drop rows with >1 missing value
   Step 8: clean_icp_outliers()             — Z-score > 7
   (cleaning stops here — impute & lag now happen AFTER the split)
   └─ → data/cleaned_df_{mode}.csv
        │
        ▼
 main.py (build_split_datasets) — patient-level 60/20/20 train/val/test split (random_state=42)
        │
        ├─ causal_knn_impute(train, val, test)  — 1-NN, donors = complete TRAIN rows only,
        │        donor timestamp ≤ target (no future / cross-split leakage)
        ├─ add_lagged_features() per split      — N hourly lag windows per variable
        │
        ├─ [classification] BinaryDataProcessor.create_binary_data()
        │        — adds icp_binary (1 if icp ≥ 22, else 0), drops raw icp
        │
        ├─ → data/train_data_{mode}.csv
        ├─ → data/validation_data_{mode}.csv
        ├─ → data/test_data_{mode}.csv
        │
        ├─ [regression]  RegressionPipeline.run_pipeline()   — select on val, report on test
        └─ [classification] ClassificationPipeline.run_pipeline()  — select on val, report on test
                │
                ▼
          Model evaluation results printed to console
          xgboost_feature_importance.png saved to root
```

---

## Module Reference

### `data_parser/data_parser.py` — `DataParser`

Merges 13 raw ICU variable files into a single time-aligned CSV.

**Key methods:**

| Method | What it does |
|--------|-------------|
| `run()` | Top-level entry; returns path to `final_data.csv` (cached) |
| `parse(files, data_dir)` | Dispatches each file to `parse_file()` or `parse_episodes_with_high_icp()` |
| `parse_file(file, col)` | Reads CSV rows; rounds timestamps; merges duplicates via running average |
| `parse_episodes_with_high_icp()` | Extracts `date_of_birth` from the episodes file |
| `create_final_data()` | Flattens nested `patient_id → timestamp → column → value` map |
| `filter_instances()` | Interactive pathology filter (reads `pathologies_filtered.csv`) |
| `fix_missing_values()` | Forward-fill per patient; general: 30-min window; special vars: same-day |

**Time rounding logic** (`custom_round_time`):

- Minutes :16–:45 → rounded to :30
- Minutes :46–:15 → rounded to :00 (next or current hour)

**Special forward-fill variables** (same-day, not just 30-min):
`glucose`, `haemoglobin`, `paco2`, `pao2`, `ph`

**Default missing value sentinel:** `-1` (except `peep` which defaults to `0`)

---

### `data_parser/data_pre_processor.py` — `DataPreProcessor`

Multi-step cleaning, then (post-split) causal imputation and feature engineering. Each step logs detailed diagnostics.

**Constructor:** `DataPreProcessor(raw_data_file_path, config)`

**Main method:** `pre_process_dataset(hours, mode)` — cleaning STEPS 1-6 only, cached to `cleaned_df_{mode}.csv`. Imputation and lagging are exposed as separate methods (`causal_knn_impute`, `add_lagged_features`) that `main.py` calls **after** the patient-level split.

**Cleaning step details (in `pre_process_dataset`):**

| Step | Method | Notes |
|------|--------|-------|
| — | `add_pathology_one_hot_features()` | Config-gated (`preprocessing.add_pathology_one_hot`). Reads `pathologies_filtered.csv`. Groups rare pathologies under `pathology_other`. |
| — | `transform_all_columns_to_float()` | Skips `patient_id`, `date_of_birth`, `timestamp`. |
| 1 | `delete_negative_icp_values()` | Removes rows where ICP < 0. |
| 2 | _(inline)_ | `groupby('patient_id').filter(len > 2)` — drops patients with ≤2 measurements. |
| 3 | `drop_rows_with_null_target_column()` | Drops rows with null `icp`. |
| 4 | `standardize_missing_values()` | Replaces `[-1, '-1', '--', '-', '.', '/']` with `NaN`. Prints per-column missing table. |
| 5 | `drop_columns_with_high_missing_values()` | Config-gated. Drops `respiration_rate` (always), then rows with >1 null. |
| 6 | `clean_icp_outliers()` | Z-score threshold = 7 (domain-appropriate for medical data). |

**Post-split methods (called from `main.py`):**

| Method | Notes |
|--------|-------|
| `causal_knn_impute(train, val, test)` | 1-NN imputation of rows with exactly 1 null. Donor pool = complete **TRAIN** rows; donor `timestamp ≤ target` (no future/cross-split leakage). Prints a same-patient / future-donor diagnostic. |
| `add_lagged_features(df, hours)` | Delegates to `TimeSeriesProcessor`, called once per split. |

**Pathologies grouped as "other":**
`cns infection`, `assdh`, `status epilepticus`, `intracranial hypertension due to acute leukemia`, `hydrocephalus`

**Diagnostic helpers:**

- `print_missing_value_table(df, title)` — per-column null count + %, skips non-clinical columns
- `print_percentages_of_rows_with_missing_values(df)` — row-level null distribution

---

### `data_parser/time_series_processor.py` — `TimeSeriesProcessor`

Creates lagged features: for each row at time T and lag L, the feature value is the **mean** of all readings in the window `[T − L hours, T − (L−1) hours)`.

**Main method:** `process_data(data, hours=5, columns_to_lag=None, mode="regression")`

**Variables lagged:**
`icp`, `temperature`, `mean_blood_pressure`, `cpp`, `glucose`, `haemoglobin`, `heart_rate`, `paco2`, `pao2`, `peep`, `ph`, `spo2`

**Output columns:** `{variable}_lag_1` through `{variable}_lag_{hours}` (e.g. `icp_lag_1` … `icp_lag_5`)

**Null handling in lags:** if no readings exist in a window, the lag is `NaN`. Mode-dependent handling after creation:

- `regression`: auto-drops rows with null lags
- `classification`: asks the user

Each patient is processed in isolation (no cross-patient lag contamination).

---

### `data_parser/binary_data_processor.py` — `BinaryDataProcessor`

Converts continuous ICP into a binary classification target.

**Threshold:** `ICP_BINARY_THRESHOLD = 22` mmHg

**`create_binary_data(data)`:** adds `icp_binary` column (`1` if `icp ≥ 22`, else `0`), then drops the original `icp` column.

---

### `classification/classification_pipeline.py` — `ClassificationPipeline`

Trains and evaluates six classifier variants, runs pairwise statistical tests.

**`run_pipeline(X_train, X_test, y_train, y_test)`** — evaluates:

| Variant | Model | Features |
|---------|-------|----------|
| LaggedICP Baseline | Mean of icp_lag_1…5 ≥ 15 | icp lags only |
| LatestICP Baseline | icp_lag_1 > 15 | icp_lag_1 only |
| XGBoost (default) | XGBoost | all lag features |
| XGBoost (tuned) | XGBoost + custom params | all lag features |
| XGBoost (lag-only) | XGBoost | icp_lag_* only |
| XGBoost (lag-only, tuned) | XGBoost + custom params | icp_lag_* only |

**XGBoost tuned parameters:**

```python
{
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'scale_pos_weight': 6,   # reflects ~6:1 class imbalance
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**Evaluation:**

- Confusion matrix, classification report (precision/recall/F1 for positive class)
- McNemar's exact test for every model pair (15 comparisons)
- Case analysis: rows where XGBoost is correct and baseline is wrong

**`run_cross_validation_pipeline(X, y, n_splits=10)`:**

- `StratifiedGroupKFold` grouped by `patient_id` — prevents patient data leakage
- Stratified on `icp_binary`

**Feature importance:** saves `xgboost_feature_importance.png` to project root.

---

### `classification/lagged_icp_baseline_predictor.py` — `LaggedICPBaselinePredictor`

Rule-based baseline. Predicts `1` if the row-wise mean of `icp_lag_1` … `icp_lag_5` ≥ `decision_threshold` (default: 15 mmHg).

---

### `classification/latest_icp_baseline_predictor.py` — `LatestICPBaselinePredictor`

Rule-based baseline. Predicts `1` if `icp_lag_1 > decision_threshold` (default: 15 mmHg).

---

### `classification/xgboost_classification_predictor.py` — `XGBoostClassificationPredictor`

Wraps XGBoost with configurable model parameters and a pluggable feature selector.

- **Default feature selector:** all columns whose name contains `'lag'`
- **`lag_only_feature_selector`:** only `icp_lag_*` columns
- **Scaling:** `MinMaxScaler` fitted on training set, applied to test set
- **Excluded from scaling/training:** `patient_id`, `date_of_birth`, `timestamp`

---

### `regression/regression_pipeline.py` — `RegressionPipeline`

Trains and evaluates four regressors across four feature configurations.

**Models:**

| Model | Class | Scaling |
|-------|-------|---------|
| Linear Regression | `LinearRegression()` | StandardScaler |
| Ridge Regression | `Ridge()` | StandardScaler |
| Lasso Regression | `Lasso(alpha=0.1)` | StandardScaler |
| Baseline History | `BaselineHistoryRegression` | None |

**Feature configurations evaluated:**

| Config | Description |
|--------|-------------|
| Full Dataset | All lag columns |
| Lags 1-3-5 | Only lag indices 1, 3, 5 per variable |
| Drop PEEP/PH/SpO₂ | Excludes those three variables and their lags |
| Extensive Drop | Drops 11 variables: spo2, heart_rate, paco2, pao2, temperature, ph, peep, glucose, haemoglobin, cpp, mean_blood_pressure (and all their lags) |

**Evaluation metrics:** MSE, MAE, RMSE, model score (R²-like accuracy %)

**Cross-validation:** `KFold(n_splits=10, shuffle=True, random_state=42)`, scored with RMSE.

**Outputs returned:**

```python
{
    "evaluation_results":       pd.DataFrame,  # per-model MSE/MAE/RMSE/Accuracy
    "cross_validation_results": pd.DataFrame,  # per-model mean CV RMSE
    "feature_config_results":   pd.DataFrame   # RMSE per model × feature config
}
```

---

### `regression/baseline_history_regression.py` — `BaselineHistoryRegression`

For each test row, finds the most recent ICP reading for that patient **before** the test timestamp in the training set. Falls back to training-set global mean ICP when no prior data exists. sklearn-compatible.

---

### `regression/baseline_mean_regression.py` — `BaselineMeanRegression`

For each test row, computes the mean of all ICP readings for that patient **before** the test timestamp. Falls back to global mean.

---

### `generate_descriptive_stats.py`

Standalone script (not part of the main pipeline). Reads `cleaned_df_classification.csv` and prints:

1. **Cohort summary** — total observations, unique patients, median observations per patient
2. **Patient demographics** — age at admission by age band: `<18`, `18–34`, `35–44`, `45–54`, `55–64`, `65–74`, `75–84`, `85+`
3. **Clinical variable statistics** — mean, std, median, missing % for each of the 13 clinical variables

Age at admission is computed as `first_measurement_timestamp − date_of_birth`.

---

### `helper/data_frame_printer.py` — `DataFramePrinter`

Single static method used throughout the codebase for consistent table output:

```python
DataFramePrinter.print_dataframe_tabulated(df, title="My Title")
# → prints a fancy_grid tabulate table with the given title
```

---

## Key Constants & Thresholds

| Location | Constant | Value | Purpose |
|----------|----------|-------|---------|
| `binary_data_processor.py` | `ICP_BINARY_THRESHOLD` | 22 mmHg | Clinically significant ICP elevation |
| `data_pre_processor.py` | Z-score threshold | 7 | ICP outlier removal (domain-specific) |
| `data_pre_processor.py` | Missing value sentinels | `[-1, '--', '-', '.', '/']` | Normalized to NaN |
| `data_pre_processor.py` | `PATHOLOGIES_GROUPED_AS_OTHER` | 5 pathologies | Rare diagnoses consolidated |
| `time_series_processor.py` | `hours` (default) | 5 | Lag feature lookback window |
| `data_pre_processor.py` | causal 1-NN imputation | 1 | Single nearest *past* train donor; no future/cross-split leakage |
| `classification_pipeline.py` | `scale_pos_weight` | 6 | Approximate class imbalance ratio |
| `classification_pipeline.py` | Baseline decision threshold | 15 mmHg | Rule-based classifier cutoff |
| `regression_pipeline.py` | Lasso alpha | 0.1 | L1 regularisation strength |
| `regression_pipeline.py` | `GroupKFold` splits | 10 | Patient-grouped CV folds (train only) |
| `main.py` / `config.toml` | Train/val/test split | 60/20/20 | Patient-level; select on val, report on test |
| `main.py` | `random_state` | 42 | Reproducibility |
| `data_parser.py` | Peep default | 0 | Non-sentinel missing value for PEEP |
| `data_parser.py` | Forward-fill window (general) | 30 min | Maximum gap to forward-fill |
| `data_parser.py` | Forward-fill window (special vars) | same day | For glucose, haemoglobin, paco2, pao2, ph |

---

## Caching Behaviour

The pipeline caches intermediate results to avoid reprocessing. Each cached file is checked for existence before running its generating step:

| File | Generated by | Reused by |
|------|-------------|-----------|
| `data/final_data.csv` | `DataParser.run()` | `DataPreProcessor` |
| `data/cleaned_df_{mode}.csv` | `DataPreProcessor.pre_process_dataset()` | `main()` |
| `data/train_data_{mode}.csv` | `main()` `build_split_datasets` | `main()` pipeline dispatch |
| `data/validation_data_{mode}.csv` | `main()` `build_split_datasets` | `main()` pipeline dispatch |
| `data/test_data_{mode}.csv` | `main()` `build_split_datasets` | `main()` pipeline dispatch |

**To force a re-run of any step, delete the corresponding CSV file.** The `clean_generated_files.sh` script at the project root removes all generated files.

---

## Configuration (no interactive prompts)

The pipeline is fully **non-interactive** — all decisions come from `config.toml` (created
from `config.example.toml` on first run, read by `helper/config.py`). The former `input()`
calls have been removed. CLI `--mode`/`--hours` override the `[run]` section.

| Config key | Replaces prompt | Default |
|------------|-----------------|---------|
| `parsing.filter_by_pathology` | Filter by pathology? | `""` (all) |
| `parsing.apply_instance_filtering` | Proceed with instance filtering? | `true` |
| `preprocessing.add_pathology_one_hot` | Add one-hot pathology columns? | `false` |
| `preprocessing.drop_high_missing_columns` | Drop columns with >1 missing? | `true` |
| `preprocessing.impute_missing_values` | Impute missing values? | `true` |
| `preprocessing.drop_lagged_null_rows` | Drop rows with null lags? | `true` |
| `classification.run_cross_validation` | Run 10-fold cross-validation? | `false` |
| `split.test_size` / `split.val_size` | — (new) three-way split fractions | `0.2` / `0.2` |
