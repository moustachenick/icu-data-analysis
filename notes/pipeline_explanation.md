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
        │  DataPreProcessor.pre_process_dataset  (cleaning, steps 0–8)
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
sentinel because a true PEEP of 0 is clinically meaningful. PEEP is parsed for the
descriptive statistics, but it is excluded from the analysis during cleaning (step 0).

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
type. Nine numbered cleaning steps then follow:

0. Variables excluded from the analysis (`EXCLUDED_VARIABLES`: PEEP, at the reviewers'
   request) are removed. This is done first, so that an excluded variable cannot cause a row
   to be dropped or influence the imputation distances.
1. Rows with a negative ICP value are removed.
2. Patients with two or fewer measurements are removed, as too few observations are
   available to construct lagged features for them.
3. Rows whose ICP target is missing are removed.
4. The heterogeneous missing-value markers (-1, the string "-1", "--", "-", ".", and "/")
   are replaced with a uniform missing indicator.
5. Physiologically impossible values (outside the ranges in `PHYSIOLOGIC_RANGES`, e.g. pH
   6.5–8.0, temperature 17–45 °C, CPP −20–200 mmHg) are set to missing, so that the
   missing-value handling below deals with them.
6. Columns with a high proportion of missing values are removed (respiration rate is removed
   in all cases), after which rows containing more than one remaining missing value are
   removed.
7. ICP outliers are removed using a Z-score criterion with a permissive threshold of 7
   standard deviations, chosen to retain physiologically extreme but valid measurements.
8. Rows with ICP above 100 mmHg, the upper limit of the measurement range recommended by the
   Association for the Advancement of Medical Instrumentation (AAMI), are removed. These
   values occur mostly in sustained terminal episodes with cerebral perfusion pressure close
   to zero, where prediction has no clinical use. This step runs after step 7, so that the
   Z-score statistics are not affected.

The Z-score statistics in step 7 (mean and standard deviation of ICP) are computed on the whole
cohort before partitioning; this is the only cleaning operation that uses information across
rows. With a threshold of 7 standard deviations, it removes only extreme values.

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
`val_size`). For the dataset used in the present work (469 patients after parsing, 465 after
cleaning), the default proportions yield 279, 93, and 93 patients in the training,
validation, and test partitions. Patients whose records are too short for the full lag window
contribute no rows after the lagged features are built (454 patients remain in the full
cohort).

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
`icp_lag_N`); PEEP is not lagged. When no measurement falls within a given lag window, the corresponding lagged
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

The regression pipeline (`src/regression/regression_pipeline.py`) evaluates seven models:
ordinary least-squares linear regression, ridge regression and lasso regression; two
tree-based ensembles, Random Forest and XGBoost, which can capture non-linear relationships and
interactions; and two rule-based baselines. The lagged features serve as the predictors, and
the patient identifier and timestamp are excluded. The three linear models are fitted on
standardised features (zero mean, unit variance); the scaler is part of each model's pipeline,
so it is always fitted on the data that the model is fitted on. The tree-based models use
unscaled features. The history baseline (`BaselineHistoryRegression`) predicts the most recent
past ICP value (`icp_lag_1`), i.e. persistence; the mean baseline (`BaselineMeanRegression`)
predicts the mean of the lagged ICP values over the previous N hours.

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
feature selection are then performed on the **validation** partition. For regression, the
hyperparameters of the ridge, lasso, Random Forest and XGBoost models are selected by a grid
search that fits each combination on the training partition and keeps the one with the lowest
validation RMSE; five feature configurations (the full feature set, a reduced set retaining
lags 1, 3, and 5, a set excluding pH and oxygen saturation, the lagged ICP values only, and the
full set plus the pathology indicators) are then compared on the validation partition. For
classification, the six predictors are compared on the validation partition. The **test**
partition is read exactly once, at the end, to produce the final performance report.

As an additional robustness assessment, patient-grouped cross-validation is performed on the
training partition only. For regression this uses grouped k-fold cross-validation
(`GroupKFold`, ten folds) with folds grouped by patient identifier, and for classification it
uses grouped, stratified k-fold cross-validation (`StratifiedGroupKFold`, ten folds). No part
of the cross-validation procedure uses the validation or test partitions. In the regression
cross-validation, the scaler of the linear models is refitted within each training fold.

The classification models use fixed hyperparameters (a default and one configured setting)
and no automated search is performed for them. If a search is introduced, it is to be carried
out on the validation partition, or by means of the patient-grouped cross-validation on the
training partition, as for regression, and never on the test partition, which is reserved for
the single final evaluation.

## 9. Statistical comparison of models

Pairwise differences between classifiers are assessed on a common evaluation partition using
McNemar's test (exact form), applied to the discordant predictions of each pair of models.
For each pair, the number of cases correctly classified by one model and misclassified by the
other is tabulated, and the resulting p-value quantifies whether the difference in error
patterns is statistically significant. The number of cases in which a gradient-boosted model
classifies an elevated-ICP episode correctly while a baseline does not is additionally
reported. McNemar's test treats each observation as independent; because consecutive
observations of the same patient are correlated, its p-values are optimistic.

For regression, pairs of models are compared with the Wilcoxon signed-rank test on the
per-patient mean absolute error in the test partition, with the patient as the unit of
analysis: Random Forest and XGBoost versus linear regression, and linear regression and Random
Forest versus the history baseline.

## 10. Evaluation metrics

Regression performance is reported using the mean squared error, the mean absolute error, the
root mean squared error, and the coefficient of determination (R²), overall and stratified by
the true ICP value (≤ 20, 20–30, and > 30 mmHg). Linear-model coefficients and the
permutation importance of the tree-based models on the validation partition are reported for
inspection. Classification performance is reported using accuracy, and the precision,
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
| Hyperparameter tuning on the test partition | Regression tuning uses the validation partition; the test partition is reserved for the single final evaluation | `build_split_datasets`, regression pipeline |
| The held-out fold influencing feature scaling in cross-validation | The scaler is part of each linear model's pipeline and is refitted in each training fold | regression pipeline |

---

## Appendix — mapping of stages to source code

| Stage | Class · method | File |
|-------|----------------|------|
| Data acquisition and harmonisation | `DataParser.run` | `src/data_parser/data_parser.py` |
| Cleaning (steps 0–8) | `DataPreProcessor.pre_process_dataset` | `src/data_parser/data_pre_processor.py` |
| Patient-level partitioning | `build_split_datasets` | `src/main.py` |
| Imputation | `DataPreProcessor.causal_knn_impute` | `src/data_parser/data_pre_processor.py` |
| Lagged features | `DataPreProcessor.add_lagged_features` → `TimeSeriesProcessor.process_data` | `src/data_parser/time_series_processor.py` |
| Binary label (classification) | `BinaryDataProcessor.create_binary_data` | `src/data_parser/binary_data_processor.py` |
| Regression models | `RegressionPipeline.run_pipeline` | `src/regression/regression_pipeline.py` |
| Classification models | `ClassificationPipeline.run_pipeline` | `src/classification/classification_pipeline.py` |
| Hyperparameter tuning (validation) | `__tune_hyperparameters` | `src/regression/regression_pipeline.py` |
| Selection (validation) and evaluation (test) | `__evaluate_with_feature_configs`, `__evaluate_models`, `__evaluate_by_icp_range`, `_evaluate_on_holdout` | regression and classification pipelines |
| Per-patient Wilcoxon test (test) | `__compare_models_wilcoxon` | `src/regression/regression_pipeline.py` |
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
