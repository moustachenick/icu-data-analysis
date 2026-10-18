# Manuscript corrections — regression section

Target file: `ICP_manuscript .md` (line numbers refer to that file).
Scope: regression (reviewer comment "Να τρέξουμε κι άλλα μοντέλα regression (e.g. random forest)"),
the PEEP removal (reviewer request), and the ICP > 100 mmHg removal. The last two change the
shared preprocessing, so they also affect the classification results.
Approach: minimal changes. Each item quotes the current text and gives the replacement.
`[TBD]` marks a value that comes from the final runs (see `tasks/todo.md`, Phase 4).

## NOTE FOR THE CO-AUTHOR — tables are images

Tables 3–8 in the manuscript are embedded images (`image5`–`image10`). They cannot be edited as
text. Regenerate them from the output files, plus the new stratified table (section 2.6) and
the supplementary hyperparameter table (section 1.5):

| Table | Source file | Source table in the output |
|---|---|---|
| 3 (full, test) | `output/regression_20261018-190934.txt` | "Regression Predictions Results (Test)" |
| 4 (full, CV) | same | "Cross-Validation Results (Train, grouped)" |
| 5 (full, configurations) | same | "Feature Configuration RMSE Results (Validation)" |
| new (full, by ICP range) | same | "Regression Results by True ICP Range (Test)" |
| 6 (TBI, test) | `output/regression_20261018-195204.txt` | "Regression Predictions Results (Test)" |
| 7 (TBI, CV) | same | "Cross-Validation Results (Train, grouped)" |
| 8 (TBI, configurations) | same | "Feature Configuration RMSE Results (Validation)" |
| Suppl. (hyperparameters) | both files | "Hyperparameter Tuning Results (Validation)" |

Table 4 / 7: show mean ± SD. Use R² only if the authors want it; the old tables do not have it.

## 0. Why the numbers change

The current regression numbers in the manuscript come from an older code version. Three
groups of changes affect them:

| Source | Change | Effect on the manuscript |
|---|---|---|
| Code before today | Three-way patient-level split (60/20/20) instead of train/test | Sample counts, all metrics |
| Code before today | Causal, train-fit KNN imputation | All metrics |
| Code before today | Physiologic-range cleaning (`PHYSIOLOGIC_RANGES`) | All metrics, Methods §3 |
| Code before today | Baseline History bug fixed. It now predicts the last ICP value (`icp_lag_1`). The old version fell back to the training mean, which gave RMSE 10.63 | Large change in the baseline result and in the claims about it |
| Code before today | New Baseline Mean model (mean of `icp_lag_1..5`) | New row in all tables |
| Code before today | Feature configurations are evaluated on the validation split, not on the test split | Table 5 / 8 captions |
| Today | Scaling fitted inside each CV fold | CV numbers (small change) |
| Today | "Drop PEEP, PH, SPO2" configuration fixed (it was identical to "Full Dataset") and renamed "Drop PH, SPO2" after the PEEP removal | Table 5 / 8 column |
| Today | Random Forest and XGBoost added; all models tuned on the validation split | New rows, Methods §5 |
| Today | CV reported as mean ± SD | Tables 4 / 7 |
| Today | PEEP excluded from the analysis (reviewer request) | Methods §2, §3; all metrics; classification too |
| Today | Rows with ICP > 100 mmHg removed (device range) | Methods §3; all metrics; classification too |
| Today | Test metrics stratified by true ICP (≤ 20, 20–30, > 30 mmHg) | New table / sentence in Results |

The "Accuracy (%)" label (R² × 100) does not appear in the manuscript tables. No correction is needed for it.

Response-letter note: state openly that the baseline result changed because of a corrected
implementation, and that the old Table 5 used the test split for feature selection.

---

## 1. Methods

### 1.0 §2 Data Acquisition and Parsing (after line 47) — PEEP removal

Keep the list of recorded variables in line 47 (PEEP was recorded). Add one sentence at the
end of that paragraph:

> Positive end-expiratory pressure was recorded but was not used as a model input, because of its limited relevance to the prediction of ICP.

Table 2 (descriptive statistics) can keep PEEP; mark it with a footnote: "Not used as a model input."

### 1.1 §3 Preprocessing Pipeline — add the physiologic-range step (after line 69)

Add a new item 7 after "6. **Outlier Removal**":

> 7. **Physiologic Range Filtering**: Measurements outside physiologically plausible ranges
> (pH 6.5–8.0, temperature 17–45 °C, PaO₂ 0–700 mmHg, PaCO₂ 0–150 mmHg, MAP 20–200 mmHg,
> heart rate 15–280 bpm, CPP −20–200 mmHg) were set to missing and then
> handled by the missing-value steps above.

Note: the code runs this step before imputation. If the order of the list matters to the
authors, insert it between items 4 and 5 instead.

### 1.1b §3 item 6 Outlier Removal (line 69) — factual error, ICP > 100 mmHg removed

The current text says that the Z-score threshold of 7 "corresponds well to the clinically
accepted physiological range of intracranial pressure (0–100 mmHg)". This was not true: values
up to 160 mmHg remained after the Z-score filter. The code now removes rows with ICP > 100 mmHg
after the Z-score step.

Evidence for the justification (analysis of the retained rows with ICP > 100 mmHg, before the
change): the episodes were mostly sustained (only 5 of 136 rows were isolated spikes), CPP was
near or below 0 mmHg (median per patient −19 to +47 mmHg), and 6 of the 15 patients reached these
values within 18 h of the end of their record. So these values are mostly real terminal
intracranial hypertension, not artifacts. Do not describe them as artifacts.

Current:
> 6. **Outlier Removal**: Z-score filtering with a threshold of 7 (defines how many standard deviations away from the mean) was applied to remove extreme ICP values which corresponds well to the clinically accepted physiological range of intracranial pressure (0–100 mmHg), as recommended by the Association for the Advancement of Medical Instrumentation (AAMI) [(23)](https://www.zotero.org/google-docs/?1cndLj). Values beyond this range were considered highly unlikely and likely reflected technical artifacts.

Replace with:
> 6. **Outlier Removal**: Z-score filtering with a threshold of 7 (defines how many standard deviations away from the mean) was applied to remove extreme ICP values, which likely reflected technical artifacts. In addition, observations with ICP above 100 mmHg, the upper limit of the measurement range recommended by the Association for the Advancement of Medical Instrumentation (AAMI) [(23)](https://www.zotero.org/google-docs/?1cndLj), were excluded (163 observations from 17 patients). These values mostly occurred during sustained terminal episodes with cerebral perfusion pressure close to or below 0 mmHg, in which ICP prediction has no clinical utility.

Note: keep the Zotero link exactly as it is in the manuscript.

### 1.2 §3 item 5 Imputation (line 67) — resolves the Greek note on that line

Current:
> a K-Nearest Neighbors (KNN) imputer (`k=1`) filled single missing values by identifying the most similar complete instance, regardless of patient ID. (να βαλουμε οτι ειναι από παρελθοντικό timestamp και από το ιδιο set)

Replace with:
> a K-Nearest Neighbors (KNN) imputer (`k=1`) filled single missing values by identifying the most similar complete instance, regardless of patient ID. Candidate donors were restricted to complete rows of the training set with a timestamp not later than the imputed row, so that no information from the validation or test sets, and no future measurement, was used.

### 1.3 §5 Regression, second paragraph (line 86)

Current:
> Three linear regression models were implemented: ordinary least squares linear regression, Ridge regression, and Lasso regression. In addition, a baseline history regression model was used, which predicts the current ICP as the mean of recent historical ICP values. All regression models were trained on lagged physiological features generated as described above. Model performance was evaluated using mean squared error (MSE), mean absolute error (MAE) and root mean squared error (RMSE).

Replace with:
> Three linear regression models were implemented: ordinary least squares linear regression, Ridge regression, and Lasso regression. To capture non-linear relationships and interactions between variables, two tree-based ensemble models were also implemented: Random Forest and XGBoost. Two rule-based baselines were used for comparison: a baseline history model, which predicts the most recent ICP value (persistence), and a baseline mean model, which predicts the mean of the ICP values over the previous 5 hours. All regression models were trained on lagged physiological features generated as described above. Features were standardized for the linear models only; the scaler was fitted on the training data (within each training fold during cross-validation). Hyperparameters of all machine learning models were selected on the validation set by minimizing RMSE (Supplementary Table S[TBD], see 1.5). Model performance was evaluated on the independent test set using mean squared error (MSE), mean absolute error (MAE) and root mean squared error (RMSE), overall and stratified by the true ICP value (≤ 20, 20–30 and > 30 mmHg). In addition, 10-fold patient-grouped cross-validation was performed on the training set, and RMSE was reported as mean ± standard deviation across folds.

### 1.4 §5 Regression — statistical comparison (append to the paragraph in 1.3)

> Pairwise differences between models were assessed with the Wilcoxon signed-rank test on per-patient MAE in the test set, with the patient as the unit of analysis.

Decision (2026-10-18): report the MAE-based test only.

### 1.5 Supplementary table — selected hyperparameters (replaces "Supplementary Table [TBD]" in 1.3)

| Model | Full cohort | TBI subgroup |
|---|---|---|
| Ridge | α = 0.01 | α = 100 |
| Lasso | α = 0.001 | α = 0.01 |
| Random Forest | 300 trees, max depth 10, min samples per leaf 20, all features | same |
| XGBoost | 500 trees, max depth 3, learning rate 0.03, subsample 0.8 | 500 trees, max depth 3, learning rate 0.03, subsample 1.0 |

Grids: Ridge α ∈ {0.01, 0.1, 1, 10, 100}; Lasso α ∈ {0.001, 0.01, 0.1, 1}; RF max depth ∈ {None, 10, 20}, min samples per leaf ∈ {1, 5, 20}, max features ∈ {all, √p}; XGBoost trees ∈ {200, 500}, max depth ∈ {3, 6, 9}, learning rate ∈ {0.03, 0.1}, subsample ∈ {0.8, 1.0}.

---

## 2. Results — §2 Full Cohort Regression

Source: `output/regression_20261018-190934.txt` (same data and numbers as `output/regression_20261018-181758.txt`, which also shows the cleaning counts).

### 2.1 Line 169

Current:
> A total of 125,060 time-stamped observations from 469 patients were included in the regression analysis after data parsing and preprocessing. The dataset was split at the patient level into training and test sets, resulting in 101,544 training samples and 23,516 test samples. KNN imputation was used to fill missing values. Four regression models were evaluated: Linear Regression, Ridge Regression, Lasso Regression, and a Baseline History Regression model. Performance metrics on the independent test set are summarized in Tables 3,4 and 5.

Replace with:
> A total of 121,367 time-stamped observations from 454 patients were included in the regression analysis after data parsing, preprocessing and lagged feature generation. The dataset was split at the patient level into training, validation and test sets, resulting in 70,141 training samples (271 patients), 27,191 validation samples (92 patients) and 24,035 test samples (91 patients). KNN imputation was used to fill missing values. Seven regression models were evaluated: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, XGBoost, and the Baseline History and Baseline Mean models. Results are summarized in Tables 3, 4 and 5.

Patient counts: 469 patients after parsing → 465 after cleaning → 454 with at least one observation after the lagged features (patients with too short records have no complete 5-hour history). Check that Methods line 41 ("469 patients remained in the final cohort used for model development") is consistent; suggested: "469 patients remained after parsing; 454 contributed observations to the regression models".

### 2.2 Line 171

Replace with:
> Linear and Ridge regression models demonstrated identical performance, achieving a mean squared error (MSE) of 12.53, mean absolute error (MAE) of 2.06, and root mean squared error (RMSE) of 3.540. The Lasso regression model showed nearly identical performance, with an RMSE of 3.540 (MSE 12.53, MAE 2.06). Random Forest achieved the lowest error (RMSE 3.483, MSE 12.13, MAE 2.01), followed by XGBoost (RMSE 3.487, MSE 12.16, MAE 2.04). The Baseline History and Baseline Mean models achieved RMSE values of 3.802 (MSE 14.46, MAE 2.06) and 3.862 (MSE 14.91, MAE 2.37), respectively.

Add after it (Wilcoxon, per-patient MAE, 91 test patients):
> At the patient level, Random Forest had a lower MAE than Linear Regression (median difference −0.06 mmHg, p < 0.001), as had XGBoost (−0.02 mmHg, p = 0.009). However, neither Linear Regression (median difference +0.05 mmHg, p = 0.07) nor Random Forest (−0.01 mmHg, p = 0.36) had a significantly lower per-patient MAE than the Baseline History model.

### 2.3 Line 173 — CV

Replace with:
> Ten-fold patient-grouped cross-validation on the training set yielded mean RMSE values (± SD) of 3.36 ± 0.39 for Linear, Ridge and Lasso Regression, 3.33 ± 0.40 for Random Forest and 3.35 ± 0.38 for XGBoost, compared with 3.57 ± 0.49 and 3.71 ± 0.34 for the Baseline History and Baseline Mean models.

### 2.4 Line 175 — feature importance

Append:
> Permutation importance on the validation set showed the same pattern for Random Forest and XGBoost: icp_lag_1 and icp_lag_2 accounted for almost all of the predictive contribution, and non-ICP variables increased the validation RMSE by less than 0.06 mmHg when permuted.

### 2.5 Line 177 — feature configurations

Replace with:
> Feature configuration experiments on the validation set showed that restricting the models to selected lagged features (lags 1, 3, and 5), excluding selected physiological variables, or adding the admission diagnosis changed the RMSE by less than 0.05 mmHg, indicating stability of the regression models across different feature configurations.

### 2.6 Tables 3, 4, 5 (images `image5`, `image6`, `image7`) and new table

- Regenerate Tables 3–5 from the run output. Add rows: Random Forest, XGBoost, Baseline Mean.
- Table 4: mean ± SD.
- Table 5: rename "Non-Significant Feature Drop" to "Drop pH, SpO₂"; add "Lags + Pathology".
- New Table (stratified, test set):

| Model | RMSE ≤ 20 (n = 22,940) | RMSE 20–30 (n = 618) | RMSE > 30 (n = 477) |
|---|---|---|---|
| Linear / Ridge | 2.88 | 9.09 | 11.22 |
| Lasso | 2.88 | 9.09 | 11.22 |
| Random Forest | 2.73 | 9.63 | 11.55 |
| XGBoost | 2.72 | 9.57 | 11.71 |
| Baseline History | 3.22 | 9.08 | 11.05 |
| Baseline Mean | 3.10 | 9.81 | 12.78 |

  Sentence for §2:
  > Stratified by the true ICP value, the tree-based models had the lowest errors for ICP ≤ 20 mmHg (RMSE 2.72–2.73 vs. 2.88 for the linear models), whereas for ICP of 20–30 mmHg and above 30 mmHg no model was more accurate than the Baseline History model (RMSE 9.08 and 11.05, respectively).

- Caption, Table 4 (line 183): "…using 10-fold patient-grouped cross-validation on the training set. Values are mean ± standard deviation across folds." (Regression uses `GroupKFold`; it is not stratified.)
- Caption, Table 5 (line 186): "Validation-set RMSE of regression models across different feature configurations in the full cohort".

---

## 3. Results — §3 TBI Subgroup Regression (lines 190–207)

Source: `output/regression_20261018-195204.txt`. No TBI observation had ICP > 100 mmHg.

### 3.1 Line 190

Replace the counts and model number:
> The filtered dataset included 64,330 timestamped observations from 234 TBI patients. A patient-level split was applied, producing 40,578 training, 11,866 validation and 11,886 test samples. The same seven regression models were evaluated.

Check: 234 is the number of patients in the split (140 / 47 / 47). One test patient had no observations after lagging (the Wilcoxon test used 46 test patients). Report the patients with observations if the full-cohort paragraph does the same.

### 3.2 Line 192

Replace with:
> Linear, Ridge and Lasso regression showed nearly identical performance (RMSE 3.610, 3.608 and 3.605; MSE 13.03, 13.02 and 13.00; MAE 2.18, 2.18 and 2.17). Random Forest achieved the lowest error (RMSE 3.525, MSE 12.42, MAE 2.12), whereas XGBoost (RMSE 3.594, MSE 12.91, MAE 2.17) performed similarly to the linear models. The Baseline History and Baseline Mean models achieved RMSE values of 3.910 (MSE 15.29, MAE 2.17) and 3.861 (MSE 14.91, MAE 2.46). At the patient level (46 test patients), Random Forest had a lower MAE than Linear Regression (median difference −0.04 mmHg, p < 0.001); XGBoost did not (p = 0.07), and neither Linear Regression (p = 0.20) nor Random Forest (p = 0.91) differed significantly from the Baseline History model.

### 3.3 Line 194

Replace with:
> Ten-fold patient-grouped cross-validation yielded mean RMSE values of 3.34 ± 0.39 for Linear, Ridge and Lasso Regression, 3.29 ± 0.38 for Random Forest and 3.31 ± 0.38 for XGBoost, compared with 3.55 ± 0.47 and 3.70 ± 0.41 for the Baseline History and Baseline Mean models.

### 3.4 Lines 196–198

Line 196: append "Permutation importance of the tree-based models showed the same dominance of recent ICP history." Line 198: keep.

### 3.5 Tables 6, 7, 8

Regenerate; captions as for Tables 3–5. Optional: stratified table for TBI (test, RMSE ≤ 20 / 20–30 / > 30): Linear 2.97 / 10.75 / 10.14; Random Forest 2.80 / 11.20 / 10.34; XGBoost 2.80 / 11.13 / 11.47; Baseline History 3.35 / 10.83 / 9.89.

---

## 4. Discussion

IMPORTANT: The per-patient Wilcoxon results do not support "ML models outperformed the baselines" for regression. The RMSE gap (3.48–3.54 vs. 3.80) comes from fewer large errors, not from a better typical patient. The text below reflects this.

### 4.1 Line 249

Current:
> Linear regression models achieved good predictive accuracy for continuous ICP estimation (RMSE ≈ 3.3) and consistently outperformed baseline approaches based on simple historical averaging.

Replace with:
> Linear and tree-based regression models achieved good predictive accuracy for continuous ICP estimation (RMSE ≈ 3.5) and lower overall errors than baseline approaches based on the most recent ICP value or its recent mean, although the per-patient MAE did not differ significantly from that of the most-recent-value baseline.

### 4.2 Line 251

"an RMSE of approximately 3.3 mmHg" → "an RMSE of approximately 3.5 mmHg".

### 4.3 Line 255 — add after the first sentence

> The tree-based models, which can capture non-linear relationships and interactions between variables, improved only marginally on the linear models (median per-patient MAE difference below 0.1 mmHg), which further supports this observation.

### 4.4 Line 259

"linear models maintained similar performance (RMSE ≈ 3.25), and the dominance of recent ICP history remained evident, with baseline averaging approaches again performing substantially worse" → "linear and tree-based models maintained similar performance (RMSE ≈ 3.5–3.6), and the dominance of recent ICP history remained evident".

### 4.5 Line 263

Current: "In line with previous studies, machine learning models outperformed baseline approaches in both regression and classification settings."

Replace with: "In line with previous studies, machine learning models outperformed baseline approaches in the classification setting, and achieved lower overall errors in the regression setting."

(The classification claim must be checked against the new classification results; see section 5.)

### 4.6 Limitations (line 267) — add after "…rapid physiological changes."

> In addition, for ICP values above 20 mmHg, none of the regression models was more accurate than the most recent ICP value, and tree-based models cannot predict values outside the range observed in the training data, which limits their accuracy for rare, very high ICP values.

---

## 5. Not covered here

- Classification results (§4, §5, Tables 9–12) probably also come from the older code version (two-way split). They must be rerun, because the PEEP and ICP > 100 changes are in the shared preprocessing. The numbers in Results §4 line 211 (141,639 observations, 30,079 test samples) will change.
