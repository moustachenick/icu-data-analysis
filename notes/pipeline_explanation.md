# Data Processing and Modelling Pipeline — Detailed Description

This document describes the complete data processing and modelling pipeline used to predict
intracranial pressure (ICP) in intensive care unit (ICU) patients. Each processing stage is described
in turn, together with the software class that implements it and the rationale for the order
in which the stages are applied. The companion file `notes/summary.md` provides the corresponding
file-by-file technical reference.

Two prediction tasks are supported and share the same data preparation:

- **Regression** — the continuous ICP value (in mmHg) is predicted.
- **Classification** — the occurrence of elevated ICP (ICP ≥ 22 mmHg, a clinically
  significant threshold for intervention) is predicted as a binary outcome.

The pipeline proceeds through seven stages: (1) data acquisition and harmonisation,
(2) cleaning, (3) patient-level partitioning into training, validation, and test sets,
(4) missing-value imputation, (5) time-series feature engineering, (6) label construction
(classification only), and (7) model training, selection, and evaluation. Partitioning is
performed **before** imputation, feature engineering, and any form of model or feature
selection, so that information from the validation and test partitions cannot influence the
fitted models. The order of the stages is summarised in the diagram below.

---

## Overview of the data flow

```text
Raw per-variable .txt files (ICP, CPP, glucose, ...)
        │  DataParser
        ▼
final_data.csv            one row per (patient, timestamp); harmonised and forward-filled
        │  DataPreProcessor.pre_process_dataset  (cleaning, steps 1–6)
        ▼
cleaned_df_{mode}.csv      cleaned table; missing values retained; no lag features yet
        │  main.build_split_datasets
        ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  Patient-level split  →  train / validation / test  (disjoint patients) │
 │  Imputation           →  fitted on the training partition only          │
 │  Lagged features      →  constructed within each partition              │
 │  Binary label         →  added for the classification task              │
 └───────────────────────────────────────────────────────────────────────┘
        ▼
train_data_{mode}.csv   validation_data_{mode}.csv   test_data_{mode}.csv
        │  RegressionPipeline / ClassificationPipeline
        ▼
 Model fitting on training; selection on validation; single evaluation on test
```

---

## 1. Data acquisition and harmonisation (`DataParser`)

The raw data are provided as one plain-text file per physiological variable
(intracranial pressure, cerebral perfusion pressure, glucose, haemoglobin, heart rate, mean
blood pressure, arterial carbon dioxide and oxygen partial pressures, positive
end-expiratory pressure, pH, respiration rate, peripheral oxygen saturation, and
temperature), accompanied by a separate file from which each patient's date of birth is
obtained. These files are merged into a single time-aligned table by the `DataParser` class
(`src/data_parser/data_parser.py`).

Each measurement is read together with its patient identifier and timestamp. Timestamps are
rounded onto a regular 30-minute grid by `custom_round_time`: values in the interval
:16–:45 are rounded to the half hour, and values in the interval :46–:15 are rounded to the
full hour. When several measurements of the same variable fall on the same grid point for
the same patient, they are combined into a single value using an incremental (running) mean,
so that no individual reading is given disproportionate weight.

The nested representation (patient → timestamp → variable → value) is then flattened by
`create_final_data` into a rectangular table containing one row per (patient, timestamp)
pair and one column per variable. Missing entries are encoded with a sentinel value of -1,
with the exception of positive end-expiratory pressure (PEEP), for which 0 is used as the
sentinel because a true PEEP of 0 is clinically meaningful.

The cohort is then restricted by `filter_instances` to the patients listed in the pathology
reference file; an optional single-pathology filter may also be applied. Finally,
`fix_missing_values` performs within-patient forward-filling: a missing value is carried
forward from the most recent earlier reading of the same variable when that reading lies
within the preceding 30 minutes. For a defined subset of slowly varying laboratory
variables (glucose, haemoglobin, arterial carbon dioxide partial pressure, arterial oxygen
partial pressure, and pH), the most recent value from the same calendar day is carried
forward instead. The harmonised table is written to `data/final_data.csv` and is cached;
subsequent runs reuse it without re-parsing.

## 2. Data cleaning (`DataPreProcessor.pre_process_dataset`)

Cleaning is performed by the `DataPreProcessor` class
(`src/data_parser/data_pre_processor.py`). Only operations that treat each row independently,
or that remove rows and columns, are applied at this stage; no operation that borrows
information across rows (such as imputation) is performed here, because cleaning precedes the
data partitioning. Two preliminary operations are carried out first: optional one-hot
encoding of pathology categories, and coercion of all measurement columns to floating-point
type. Six numbered cleaning steps then follow:

1. Rows with a negative ICP value are removed.
2. Patients with two or fewer measurements are removed, as too few observations are
   available to construct lagged features for them.
3. Rows whose ICP target is missing are removed.
4. The heterogeneous missing-value markers (-1, the string "-1", "--", "-", ".", and "/")
   are replaced with a uniform missing indicator.
5. Columns with a high proportion of missing values are removed (respiration rate is removed
   in all cases), after which rows containing more than one remaining missing value are
   removed.
6. ICP outliers are removed using a Z-score criterion with a permissive threshold of 7
   standard deviations, chosen to retain physiologically extreme but valid measurements.

The cleaned table is written to `data/cleaned_df_{mode}.csv`. This table still contains the
small number of rows with a single missing value and does not yet contain any lagged
features; imputation and feature construction are deferred until after partitioning.

## 3. Dataset partitioning (`main.build_split_datasets`)

Partitioning is performed at the level of the patient by `build_split_datasets`
(`src/main.py`). The set of unique patient identifiers is divided into three disjoint groups
— training, validation, and test — in default proportions of 60%, 20%, and 20%
respectively. All rows belonging to a given patient are assigned to a single partition, so
that no patient contributes observations to more than one partition. This patient-level
partitioning prevents the model from being evaluated on a patient whose other measurements
were seen during training, which would otherwise yield optimistic performance estimates.

The split is produced by two successive applications of a random patient-level partition
with a fixed random seed (42), ensuring that runs are reproducible. The partition
proportions are configurable through the `[split]` section of `config.toml` (`test_size` and
`val_size`). For the dataset used in the present work (469 patients), the default proportions
yield approximately 279, 93, and 93 patients in the training, validation, and test
partitions.

## 4. Missing-value imputation (`DataPreProcessor.causal_knn_impute`)

Imputation is applied after partitioning and is fitted on the training partition alone, by
`causal_knn_impute`. Only rows that contain exactly one missing value are imputed; rows with
more than one missing value are removed, consistent with the cleaning policy. For each such
row, the missing entry is filled using a single-nearest-neighbour rule: the distance to
candidate donor rows is computed over the features that are present in the target row
(partial Euclidean distance), and the value of the nearest donor is copied into the missing
entry.

Two constraints are imposed on the donor pool to avoid information leakage. First, donors are
restricted to complete rows drawn from the **training partition only**, so that no
information from the validation or test partitions can enter the imputation. Second, a donor
is eligible only if its timestamp does not exceed that of the target row, so that a future
measurement is never used to reconstruct a past one; cross-patient donors are permitted under
the missing-at-random assumption, while a patient's own future observations are excluded by
the temporal constraint. Diagnostic statistics are reported for each partition, including the
proportion of donors originating from the same patient and a verification that no donor with
a later timestamp than the target is ever selected.

## 5. Time-series feature engineering (`TimeSeriesProcessor`)

Lagged features are constructed within each partition by `add_lagged_features`, which
delegates to the `TimeSeriesProcessor` class (`src/data_parser/time_series_processor.py`).
For each patient, observations are processed in chronological order. For each target
observation and each lag h from 1 to N hours, the measurements of every monitored variable
that fall within the one-hour window ending h hours before the target are aggregated by their
mean and recorded as the corresponding lagged feature (for example `icp_lag_1` to
`icp_lag_N`). When no measurement falls within a given lag window, the corresponding lagged
feature is left missing. The lookback horizon N is set to five hours by default and is
configurable. Because lagged features are derived strictly from earlier observations of the
same patient, their construction within each partition is equivalent to construction on the
full table while keeping the partitions independent. Rows that retain missing lagged values
are then removed.

## 6. Label construction (`BinaryDataProcessor`)

For the classification task, a binary outcome is created by the `BinaryDataProcessor` class
(`src/data_parser/binary_data_processor.py`): the label `icp_binary` is set to 1 when ICP is
greater than or equal to 22 mmHg and to 0 otherwise, and the continuous ICP column is then
removed. This step is applied to each partition after the lagged features have been
constructed. The resulting training, validation, and test tables are written to
`data/train_data_{mode}.csv`, `data/validation_data_{mode}.csv`, and
`data/test_data_{mode}.csv`, and are cached for reuse.

## 7. Predictive models

### 7.1 Regression (`RegressionPipeline`)

The regression pipeline (`src/regression/regression_pipeline.py`) evaluates four models: an
ordinary least-squares linear regression, a ridge regression, a lasso regression (with an L1
penalty of 0.1), and a history-based baseline. The three linear models are fitted on
standardised features (zero mean, unit variance), with the patient identifier and timestamp
excluded from the predictor set; the lagged features serve as the predictors. The
history-based baseline (`BaselineHistoryRegression`) predicts, for each observation, the most
recent earlier ICP value recorded for the same patient, falling back to the overall mean ICP
when no earlier value exists. Two further baselines are implemented in the codebase — a
per-patient historical mean (`BaselineMeanRegression`) and a time-windowed historical mean
(`BaselineTimeWindowMeanICPRegression`) — but these are not part of the default evaluation
set.

### 7.2 Classification (`ClassificationPipeline`)

The classification pipeline (`src/classification/classification_pipeline.py`) evaluates six
predictors. Two are rule-based baselines that emulate bedside decision making using only past
ICP values: `LatestICPBaselinePredictor` classifies an observation as elevated when the most
recent lagged ICP value (`icp_lag_1`) exceeds a decision threshold of 15 mmHg, and
`LaggedICPBaselinePredictor` does so when the mean of the lagged ICP values (`icp_lag_1` to
`icp_lag_5`) reaches the same threshold. The remaining four predictors are gradient-boosted
decision-tree classifiers implemented with XGBoost (`XGBoostClassificationPredictor`),
obtained by combining two parameter settings (default parameters versus a configured setting
with maximum tree depth 6, learning rate 0.05, 200 estimators, subsample and column-sample
fractions of 0.8, and a positive-class weight of 6 to reflect the approximately six-to-one
class imbalance) with two feature sets (all lagged features versus the ICP-specific lagged
features only). Features for the gradient-boosted models are scaled to the unit interval
before fitting.

## 8. Model selection and final evaluation

A strict separation is maintained between the data used to compare and select models and the
data used to report final performance. Models are fitted on the training partition. Model and
feature selection are then performed on the **validation** partition: for regression, four
feature configurations (the full feature set, a reduced set retaining lags 1, 3, and 5, a set
excluding PEEP, pH, and oxygen saturation, and a more extensively reduced set) are compared
on the validation partition; for classification, the six predictors are compared on the
validation partition. The **test** partition is read exactly once, at the end, to produce the
final performance report for the selected configuration.

As an additional robustness assessment, patient-grouped cross-validation is performed on the
training partition only. For regression this uses grouped k-fold cross-validation
(`GroupKFold`, ten folds) with folds grouped by patient identifier, and for classification it
uses grouped, stratified k-fold cross-validation (`StratifiedGroupKFold`, ten folds). No part
of the cross-validation procedure uses the validation or test partitions.

The models currently use fixed hyperparameters and no automated hyperparameter search is
performed. Should hyperparameter tuning be introduced, the present design determines its
placement unambiguously: tuning is to be carried out on the validation partition, or by means
of the patient-grouped cross-validation on the training partition, and never on the test
partition, which is reserved for the single final evaluation.

## 9. Statistical comparison of classifiers

Pairwise differences between classifiers are assessed on a common evaluation partition using
McNemar's test (exact form), applied to the discordant predictions of each pair of models.
For each pair, the number of cases correctly classified by one model and misclassified by the
other is tabulated, and the resulting p-value quantifies whether the difference in error
patterns is statistically significant. The number of cases in which a gradient-boosted model
classifies an elevated-ICP episode correctly while a baseline does not is additionally
reported.

## 10. Evaluation metrics

Regression performance is reported using the mean squared error, the mean absolute error, the
root mean squared error, and the coefficient of determination (R², expressed as a
percentage). Classification performance is reported using accuracy, and the precision,
recall, and F1-score for the elevated-ICP class, together with the confusion matrix.
Gradient-boosted feature importances (by information gain) are also produced for inspection.

## 11. Reproducibility and configuration

All run-time decisions are read from a configuration file (`config.toml`); the pipeline is
non-interactive. A fixed random seed (42) governs the data partitioning, so that the same
patients are assigned to the same partitions on every run. Intermediate artefacts (the merged
table, the cleaned table, and the three partition tables) are cached as CSV files and reused
when present, so that only the affected stages are recomputed after a change. The complete
console output of each run is additionally mirrored to a timestamped file under `output/`,
which allows the output of successive runs to be compared directly.

---

## Safeguards against information leakage (summary)

| Potential source of leakage | Safeguard | Implemented in |
|-----------------------------|-----------|----------------|
| The same patient appearing in more than one partition | Partitioning is performed at the patient level | `build_split_datasets` |
| Validation or test data influencing imputation | The donor pool is restricted to complete training rows | `causal_knn_impute` |
| A future measurement reconstructing a past one | Donors are required to have a timestamp no later than the target | `causal_knn_impute` |
| Selecting models or features by inspecting the test partition | Selection is performed on the validation partition; cross-validation uses the training partition | regression and classification pipelines |
| Hyperparameter tuning on the test partition | The three-way split reserves the test partition for the single final evaluation | `build_split_datasets`, pipelines |

---

## Appendix — mapping of stages to source code

| Stage | Class · method | File |
|-------|----------------|------|
| Data acquisition and harmonisation | `DataParser.run` | `src/data_parser/data_parser.py` |
| Cleaning (steps 1–6) | `DataPreProcessor.pre_process_dataset` | `src/data_parser/data_pre_processor.py` |
| Patient-level partitioning | `build_split_datasets` | `src/main.py` |
| Imputation | `DataPreProcessor.causal_knn_impute` | `src/data_parser/data_pre_processor.py` |
| Lagged features | `DataPreProcessor.add_lagged_features` → `TimeSeriesProcessor.process_data` | `src/data_parser/time_series_processor.py` |
| Binary label (classification) | `BinaryDataProcessor.create_binary_data` | `src/data_parser/binary_data_processor.py` |
| Regression models | `RegressionPipeline.run_pipeline` | `src/regression/regression_pipeline.py` |
| Classification models | `ClassificationPipeline.run_pipeline` | `src/classification/classification_pipeline.py` |
| Selection (validation) and evaluation (test) | `__evaluate_with_feature_configs`, `__evaluate_models`, `_evaluate_on_holdout` | regression and classification pipelines |
| Cross-validation (training partition) | `__perform_cross_validation`, `run_cross_validation_pipeline` | regression and classification pipelines |

## Changing the partition proportions

The partition proportions are set in `config.toml`:

```toml
[split]
test_size = 0.2   # fraction of all patients reserved for the final evaluation
val_size  = 0.2   # fraction used for selection; the remainder forms the training partition
```

After the proportions are changed, the cached partition files
(`data/train_data_*.csv`, `data/validation_data_*.csv`, `data/test_data_*.csv`, and
`data/cleaned_df_*.csv`) are deleted — or `clean_generated_files.sh` is run — so that the
partitions are regenerated on the next run.
