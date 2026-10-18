# Manuscript corrections — regression section

Target file: `ICP_manuscript .md` (line numbers refer to that file).
Scope: regression (reviewer comment "Να τρέξουμε κι άλλα μοντέλα regression (e.g. random forest)"),
the PEEP removal (reviewer request), and the ICP > 100 mmHg removal. The last two change the
shared preprocessing, so they also affect the classification results.
Approach: minimal changes. Each item quotes the current text and gives the replacement.
`[TBD]` marks a value that comes from the final runs (see `tasks/todo.md`, Phase 4).

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
> Three linear regression models were implemented: ordinary least squares linear regression, Ridge regression, and Lasso regression. To capture non-linear relationships and interactions between variables, two tree-based ensemble models were also implemented: Random Forest and XGBoost. Two rule-based baselines were used for comparison: a baseline history model, which predicts the most recent ICP value (persistence), and a baseline mean model, which predicts the mean of the ICP values over the previous 5 hours. All regression models were trained on lagged physiological features generated as described above. Features were standardized for the linear models only; the scaler was fitted on the training data (within each training fold during cross-validation). Hyperparameters of all machine learning models were selected on the validation set by minimizing RMSE (Supplementary Table [TBD]). Model performance was evaluated on the independent test set using mean squared error (MSE), mean absolute error (MAE) and root mean squared error (RMSE), overall and stratified by the true ICP value (≤ 20, 20–30 and > 30 mmHg). In addition, 10-fold patient-grouped cross-validation was performed on the training set, and RMSE was reported as mean ± standard deviation across folds.

### 1.4 §8 Evaluation and Validation (line 136) — optional, only if the Wilcoxon test is done (todo 3.5)

Add at the end of §5 instead (it is the regression section):
> Differences between the best tree-based and the best linear model were assessed with the Wilcoxon signed-rank test on per-patient MAE in the test set.

---

## 2. Results — §2 Full Cohort Regression

### 2.1 Line 169

Current:
> A total of 125,060 time-stamped observations from 469 patients were included in the regression analysis after data parsing and preprocessing. The dataset was split at the patient level into training and test sets, resulting in 101,544 training samples and 23,516 test samples. KNN imputation was used to fill missing values. Four regression models were evaluated: Linear Regression, Ridge Regression, Lasso Regression, and a Baseline History Regression model. Performance metrics on the independent test set are summarized in Tables 3,4 and 5.

Replace with:
> A total of [TBD] time-stamped observations from 469 patients were included in the regression analysis after data parsing and preprocessing. The dataset was split at the patient level into training, validation and test sets, resulting in [TBD] training, [TBD] validation and [TBD] test samples. KNN imputation was used to fill missing values. Seven regression models were evaluated: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, XGBoost, and the Baseline History and Baseline Mean models. Results are summarized in Tables 3, 4 and 5.

### 2.2 Line 171 — rewrite with new numbers

Current:
> Linear and Ridge regression models demonstrated identical performance, achieving a mean squared error (MSE) of 11.10, mean absolute error (MAE) of 1.73, and root mean squared error (RMSE) of 3.331. The Lasso regression model showed nearly identical performance, with an RMSE of 3.334 (MSE 11.12, MAE 1.73). In contrast, the Baseline History Regression model performed substantially worse, with an RMSE of 10.63, indicating limited predictive ability when relying solely on historical averaging.

Replace with (template; adjust the wording to the final ranking):
> Linear and Ridge regression models demonstrated [TBD: identical/similar] performance, achieving an MSE of [TBD], MAE of [TBD], and RMSE of [TBD]. The Lasso regression model achieved an RMSE of [TBD] (MSE [TBD], MAE [TBD]). Random Forest and XGBoost achieved RMSE values of [TBD] and [TBD], respectively. The Baseline History and Baseline Mean models achieved RMSE values of [TBD] and [TBD].

IMPORTANT: the old claim "performed substantially worse … limited predictive ability" is no
longer supported. In the reference run the gap is ≈ 0.26 mmHg RMSE (3.58 vs. 3.84). Phrase
the comparison according to the final numbers (and the Wilcoxon test, if it is done).

### 2.3 Line 173 — CV

Current:
> Ten-fold cross-validation analysis further supported these findings. Mean cross-validated RMSE values were 3.69 for Linear Regression, 3.67 for Ridge Regression, and 3.64 for Lasso Regression, whereas the Baseline History Regression model exhibited a markedly higher mean RMSE of 4.28, confirming inferior generalization performance.

Replace with:
> Ten-fold patient-grouped cross-validation on the training set yielded mean RMSE values (± SD) of [TBD] for Linear Regression, [TBD] for Ridge Regression, [TBD] for Lasso Regression, [TBD] for Random Forest and [TBD] for XGBoost, compared with [TBD] and [TBD] for the Baseline History and Baseline Mean models.

Note: the final wording depends on the outcome of the Linear Regression CV anomaly
(todo Phase 2). If one fold stays extreme, report it and explain it in one sentence.

### 2.4 Line 175 — feature importance

Append one sentence:
> Permutation importance for the Random Forest and XGBoost models showed [TBD: the same dominance of recent ICP history / additional contribution from …].

### 2.5 Line 177 — feature configurations

Current:
> Feature configuration experiments showed that restricting the model to selected lagged ICP features (e.g., lags 1, 3, and 5) or excluding selected physiological variables resulted in only marginal changes in RMSE, indicating stability of the regression models across different feature configurations.

Replace with:
> Feature configuration experiments on the validation set showed that restricting the model to selected lagged ICP features (e.g., lags 1, 3, and 5), excluding selected physiological variables, or adding the admission diagnosis resulted in [TBD: only marginal] changes in RMSE.

### 2.6 Tables 3, 4, 5 (images `image5`, `image6`, `image7`)

- Regenerate all three tables from the final full-cohort run. Add rows: Random Forest, XGBoost, Baseline Mean.
- Table 4: add an SD column (mean ± SD).
- Table 5: rename the column "Non-Significant Feature Drop" to "Drop pH, SpO₂"; add the "Lags + Pathology" column.
- New table (main text or supplement): test RMSE / MAE by true ICP range (≤ 20, 20–30, > 30 mmHg), from "Regression Results by True ICP Range (Test)" in the run output. Add one sentence to §2: "Stratified by the true ICP value, [TBD: the tree-based models had the lowest errors for ICP ≤ 20 mmHg, whereas … for ICP > 30 mmHg]."
- Caption, Table 4 (line 183). Current: "…using 10-fold stratified group cross-validation." Replace: "…using 10-fold patient-grouped cross-validation on the training set. Values are mean ± standard deviation across folds." (Regression uses `GroupKFold`; it is not stratified.)
- Caption, Table 5 (line 186). Current: "Regression model performance across different feature configurations in the full cohort". Replace: "Validation-set RMSE of regression models across different feature configurations in the full cohort".

---

## 3. Results — §3 TBI Subgroup Regression (lines 190–207)

Apply the same changes as in section 2:

- Line 190: sample counts → [TBD] (three-way split); "The same four regression models" → "The same seven regression models".
- Line 192: new numbers for all seven models. Remove "performed worse" for the baseline unless the final numbers support it.
- Line 194: CV mean ± SD for all seven models.
- Line 196: one sentence on tree-model permutation importance.
- Tables 6, 7, 8: regenerate; captions as for Tables 3–5 (line 204: "stratified group" → "patient-grouped"; line 207: "Validation-set RMSE …").

---

## 4. Discussion

### 4.1 Line 249

Current:
> Linear regression models achieved good predictive accuracy for continuous ICP estimation (RMSE ≈ 3.3) and consistently outperformed baseline approaches based on simple historical averaging.

Replace with:
> Linear and tree-based regression models achieved good predictive accuracy for continuous ICP estimation (RMSE ≈ [TBD]) and [TBD: outperformed / performed close to] baseline approaches based on the most recent ICP value or its recent mean.

### 4.2 Line 251

"an RMSE of approximately 3.3 mmHg" → "an RMSE of approximately [TBD] mmHg".

### 4.3 Line 255 — add one sentence after the first sentence

> The tree-based models, which can capture non-linear relationships and interactions, [TBD: did not improve / improved only marginally on] the linear models, which further supports this observation.

(Use only if the final results support it. This sentence is the direct answer to the reviewer.)

### 4.4 Line 259

"linear models maintained similar performance (RMSE ≈ 3.25)" → "linear and tree-based models maintained similar performance (RMSE ≈ [TBD])"; check "baseline averaging approaches again performing substantially worse" against the final TBI numbers.

### 4.5 Line 263

"machine learning models outperformed baseline approaches in both regression and classification settings" → keep only if the final regression numbers support it. Otherwise: "machine learning models outperformed baseline approaches in the classification setting and [TBD] in the regression setting".

---

### 4.6 Limitations (line 267) — add one sentence after "…rapid physiological changes."

> In addition, tree-based models cannot predict values outside the range observed in the training data, which limits their accuracy for rare, very high ICP values.

(Use only if the stratified results confirm it, as in the run before the ICP > 100 removal.)

## 5. Not covered here

- Classification results (§4, §5, Tables 9–12) probably also come from the older code version (two-way split). They must be rerun, because the PEEP and ICP > 100 changes are in the shared preprocessing. The numbers in Results §4 line 211 (141,639 observations, 30,079 test samples) will change.
