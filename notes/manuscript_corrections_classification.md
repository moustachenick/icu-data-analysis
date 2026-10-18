# Manuscript corrections — classification section

Target file: `ICP_manuscript .md` (line numbers refer to that file).
Companion file: `notes/manuscript_corrections_regression.md` (shared preprocessing changes:
three-way split, PEEP removal, ICP > 100 mmHg removal, imputation text).
Approach: minimal changes. Each item quotes the current text and gives the replacement.

Sources:
- Full cohort: `output/classification_20261018-193100.txt`
- TBI subgroup: `output/classification_20261018-201243.txt`

## NOTE FOR THE CO-AUTHOR — tables are images

Tables 9–12 in the manuscript are embedded images (`image11`–`image14`). They cannot be edited
as text. Regenerate all four from the output files above:

| Table | Source table in the output file |
|---|---|
| 9 (full, test) | "Classification Models Comparison (TEST)" in the full-cohort file |
| 10 (full, CV) | "10-Fold Cross-Validation Results (Mean ± Std Dev)" in the full-cohort file |
| 11 (TBI, test) | "Classification Models Comparison (TEST)" in the TBI file |
| 12 (TBI, CV) | "10-Fold Cross-Validation Results (Mean ± Std Dev)" in the TBI file |

Use the same model names as in the text (Latest ICP Baseline, Lagged ICP Mean Baseline, XGBoost (default), XGBoost (configured), XGBoost lag-only, XGBoost lag-only (configured)).

## 0. Why the numbers change

| Change | Effect |
|---|---|
| Three-way patient-level split (60/20/20); the validation split is used to compare the variants, the test split is the final report | Sample counts, all test metrics |
| Cross-validation runs on the training split only | CV metrics |
| Causal, train-fit imputation; physiologic-range cleaning | All metrics |
| PEEP excluded; rows with ICP > 100 mmHg removed (163 rows, 17 patients) | All metrics; fewer positive samples |

The test F1 of the XGBoost models is lower than before (e.g. lag-only 0.816 → 0.718), mostly
because recall is lower on the new test patients. The cross-validated F1 is almost unchanged
(lag-only 0.790 ± 0.069 → 0.797 ± 0.042). So the change comes mostly from the composition of the
single test split. The CV estimate is the more reliable one.

---

## 1. Methods

### 1.1 §7.3 XGBoost Classifier (line 108)

Current:
> A gradient-boosted decision tree classifier (XGBoost) was trained using lagged ICP features as inputs. Input features were normalized using Min–Max scaling.

Replace with:
> A gradient-boosted decision tree classifier (XGBoost) was trained using the lagged physiological features as inputs. Input features were normalized using Min–Max scaling fitted on the training data.

Reason: the default and configured variants use the lags of all variables; only the lag-only variants use ICP lags alone.

### 1.2 §7.3 hyperparameters (lines 110–117) — resolves the Greek note on line 117

Current:
> Hyperparameters were tuned to mitigate class imbalance and overfitting, including:
> * scale_pos_weight = 6 to account for minority class prevalence,
> * max_depth = 6,
> * learning_rate = 0.05
> * n_estimators = 200.
>
> (να γραψουμε οτι παιζαμε με τα hyperparameters)

Replace with:
> In addition to the default configuration, a second hyperparameter configuration was defined to mitigate class imbalance and overfitting:
> * scale_pos_weight = 6 to account for minority class prevalence,
> * max_depth = 6,
> * learning_rate = 0.05,
> * n_estimators = 200,
> * subsample = 0.8 and colsample_bytree = 0.8.
>
> The default and configured variants were compared on the validation set, and the test set was used only for the final evaluation.

Reason: the code uses one fixed configuration. There is no search. "Tuned" would claim a search
that did not happen. `subsample` and `colsample_bytree` are in the code but not in the text.
Only the configured variants use `scale_pos_weight`; the default variants use no class weighting.

Note for the authors: the regression models are now tuned by grid search on the validation set.
A reviewer may ask why classification is not. Option: apply the same grid search to the
classification XGBoost (not done).

### 1.3 §7.3 variant list (lines 125–128)

Current:
> * Baseline XGBoost (default parameters, all lagged features)
> * Tuned XGBoost (hyperparameter optimization, all lagged features)
> * Feature-Reduced XGBoost (using only the most important ICP lag features)
> * Tuned Feature-Reduced XGBoost (important features only, with hyperparameter tuning)

Replace with:
> * XGBoost (default parameters, all lagged features)
> * Configured XGBoost (configured hyperparameters, all lagged features)
> * Lag-only XGBoost (default parameters, ICP lag features only)
> * Configured lag-only XGBoost (configured hyperparameters, ICP lag features only)

Line 130: "…the impact of feature selection and hyperparameter tuning…" → "…the impact of feature selection and hyperparameter configuration…".

### 1.4 §7.1 / §7.2 baselines (lines 100, 104) — minor

The Latest ICP baseline predicts elevated ICP when icp_lag_1 > 15 mmHg. The Lagged ICP Mean
baseline uses mean ≥ 15 mmHg. The text says "exceeds" for both. Either change line 104 to "is
equal to or exceeds the same threshold of 15 mmHg", or align the code (not done; the effect is
very small).

### 1.5 §8 Evaluation and Validation (lines 134–138) — resolves "(Τσεκ αυτο)" on line 138

Current:
> In addition to single train–test evaluation, a 10-fold stratified group cross-validation scheme was applied. Stratification was performed on the binary ICP label, while grouping was performed at the patient level to ensure that all observations from a given patient were contained within a single fold. Performance metrics were reported as mean ± standard deviation across folds.
>
> (Τσεκ αυτο)

Replace with:
> In addition to the evaluation on the independent test set, a 10-fold stratified group cross-validation scheme was applied to the training set. Stratification was performed on the binary ICP label, while grouping was performed at the patient level to ensure that all observations from a given patient were contained within a single fold. Performance metrics were reported as mean ± standard deviation across folds.

Checked in the code: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`, groups = patient, on the training split only.

---

## 2. Results — §4 Full Cohort Classification

### 2.1 Line 211

Current:
> After preprocessing and feature engineering, a total of 141,639 time-stamped observations from 469 patients were included in the classification analysis. The independent test set comprised 30,079 samples, of which 1,673 (5.6%) were labeled as elevated ICP (≥ 22 mmHg) and 28,406 (96.2%) as normal ICP, reflecting a markedly imbalanced class distribution.

Replace with:
> After preprocessing and feature engineering, a total of 121,367 time-stamped observations from 454 patients were included in the classification analysis (70,141 training, 27,191 validation and 24,035 test samples). The independent test set comprised 24,035 samples, of which 987 (4.1%) were labeled as elevated ICP (≥ 22 mmHg) and 23,048 (95.9%) as normal ICP, reflecting a markedly imbalanced class distribution.

Note: the current text has an arithmetic error: 28,406 / 30,079 = 94.4 %, not 96.2 %.

### 2.2 Line 213

"Six classifiers were evaluated: Latest ICP Baseline, Lagged ICP Mean Baseline, XGBoost (default), XGBoost with tuned hyperparameters, XGBoost using lag-only features, and XGBoost using lag-only features with tuned hyperparameters." → replace "tuned hyperparameters" with "configured hyperparameters" (two places).

### 2.3 Line 215

Current:
> Both baseline models achieved comparable overall accuracy (0.917–0.919) but demonstrated low precision for the positive class (0.387–0.394), despite high recall values (0.841–0.846). The default XGBoost classifier achieved an accuracy of 0.981, precision of 0.934, recall of 0.704, and F1-score of 0.803. Hyperparameter tuning resulted in a more balanced precision–recall trade-off, with an F1-score of 0.796. The lag-only XGBoost variant yielded the highest F1-score (0.816), while the tuned lag-only model showed a lower precision(0.776) with higher recall(0.794), resulting in an F1- score of 0.785.

Replace with:
> Both baseline models achieved comparable overall accuracy (0.904–0.911) but demonstrated low precision for the positive class (0.275–0.284), despite high recall values (0.769–0.809). The default XGBoost classifier achieved an accuracy of 0.981, precision of 0.935, recall of 0.571, and F1-score of 0.709. The configured hyperparameters resulted in a more balanced precision–recall trade-off (precision 0.786, recall 0.644), with an F1-score of 0.708. The lag-only XGBoost variant yielded the highest F1-score (0.718), while the configured lag-only model showed a lower precision (0.665) with higher recall (0.696), resulting in an F1-score of 0.680.

### 2.4 Line 217 — CV

Replace with:
> Ten-fold stratified group cross-validation on the training set confirmed these findings. Mean accuracy across folds ranged from 0.914–0.916 for baseline models and 0.974–0.982 for XGBoost-based models. The highest mean F1-scores were observed for the lag-only XGBoost classifier (0.797 ± 0.042) and the default XGBoost classifier (0.788 ± 0.043), whereas baseline models demonstrated substantially lower mean F1-scores (0.511–0.512).

### 2.5 Line 219 — McNemar

Current:
> Pairwise model comparisons using McNemar's test demonstrated statistically significant differences between XGBoost-based classifiers and both baseline approaches (all p < 0.001). No statistically significant difference was observed between the two baseline models (p = 0.091). Comparisons between default and tuned XGBoost variants also revealed significant differences in several cases (e.g., p = 0.000005), indicating non-identical prediction behavior despite similar aggregate performance metrics.

Replace with:
> Pairwise model comparisons using McNemar's test demonstrated statistically significant differences between XGBoost-based classifiers and both baseline approaches (all p < 0.001), and between the two baseline models (p < 0.001). Comparisons between default and configured XGBoost variants also revealed significant differences (e.g., p < 0.001 for default versus configured XGBoost), indicating non-identical prediction behavior despite similar aggregate performance metrics, whereas the default and lag-only XGBoost models did not differ (p = 1.00).

IMPORTANT: the claim about the two baselines reverses (not significant → significant).

### 2.6 Captions (lines 222, 225)

Table 10 caption: "…using 10-fold stratified group cross-validation." → "…using 10-fold stratified group cross-validation on the training set."

---

## 3. Results — §5 TBI Subgroup Classification

### 3.1 Line 231

Replace with:
> The subgroup analysis focused exclusively on patients diagnosed with traumatic brain injury (TBI). After pathology-based filtering and preprocessing, the dataset comprised 64,330 time-stamped observations from 234 patients (40,578 training, 11,866 validation and 11,886 test samples). The independent test set included 11,886 samples, of which 454 (3.8%) were labeled as elevated ICP (≥ 22 mmHg) and 11,432 (96.2%) as normal ICP.

No TBI observation had ICP > 100 mmHg.

### 3.2 Line 235

Current:
> Baseline models again demonstrated high overall accuracy (0.910–0.920) but limited precision for the positive class (0.220–0.238), with recall values ranging from 0.632 to 0.664. XGBoost-based classifiers achieved substantially higher precision compared to baseline approaches. The default XGBoost model yielded an accuracy of 0.976, precision of 0.846, recall of 0.355, and an F1-score of 0.500. Hyperparameter tuning improved recall (0.497) at the expense of precision (0.585), resulting in a higher F1-score (0.537). The lag-only XGBoost model demonstrated comparable performance, with an F1-score of 0.525, while the tuned lag-only variant achieved an F1-score of 0.522.

Replace with:
> Baseline models again demonstrated high overall accuracy (0.916–0.935) but limited precision for the positive class (0.275–0.329), with recall values ranging from 0.685 to 0.738. XGBoost-based classifiers achieved substantially higher precision compared to baseline approaches. The default XGBoost model yielded an accuracy of 0.981, precision of 0.949, recall of 0.529, and an F1-score of 0.679. The configured hyperparameters improved recall (0.566) at the expense of precision (0.840), resulting in a similar F1-score (0.676). The lag-only XGBoost model demonstrated comparable performance, with an F1-score of 0.674, while the configured lag-only variant achieved an F1-score of 0.649.

IMPORTANT: "resulting in a higher F1-score" is no longer true (0.679 → 0.676).

### 3.3 Line 237 — CV

Replace with:
> Ten-fold stratified group cross-validation on the training set corroborated these findings. Mean accuracy across folds ranged from 0.903–0.907 for baseline models and 0.970–0.978 for XGBoost-based classifiers. Mean F1-scores were highest for the configured XGBoost model (0.567 ± 0.222) and the configured lag-only XGBoost model (0.552 ± 0.191), whereas baseline models showed substantially lower mean F1-scores (0.357–0.369).

### 3.4 Line 239 — McNemar

Keep the text. It is still correct: XGBoost vs. baselines all p < 0.001; default vs. lag-only p = 0.15 and lag-only vs. configured p = 0.36 (not significant). Optionally add "(e.g., default versus lag-only, p = 0.15)".

### 3.5 Caption (line 245)

Table 12 caption: add "on the training set" as for Table 10.

---

## 4. Discussion

### 4.1 Line 253

Current:
> This was most evident in the default XGBoost model (precision 0.934, recall 0.704, F1-score 0.803) and the lag-only XGBoost model, which achieved the highest F1-score (0.816).

Replace with:
> This was most evident in the default XGBoost model (precision 0.935, recall 0.571, F1-score 0.709) and the lag-only XGBoost model, which achieved the highest F1-score (0.718).

Check the next sentence: "without generating excessive false-positive warnings" still holds (precision ≥ 0.9), but recall is now lower (≈ 0.57–0.60): about 4 of 10 elevated-ICP episodes are missed by the default models. Consider adding: "although a substantial proportion of elevated ICP episodes was not detected".

### 4.2 Line 259 — TBI

"they exhibited higher precision but reduced recall compared to the full cohort" — still holds (default precision 0.949 vs. 0.935; recall 0.529 vs. 0.571; CV recall 0.42–0.56 vs. 0.68–0.79). Keep.

### 4.3 Limitations (line 267) — recommended addition

> McNemar's test was applied to individual observations; because consecutive observations of the same patient are correlated, the resulting p-values may overstate the statistical significance of the differences between models.

Reason: McNemar counts each 30-minute row as independent. With ~24,000 correlated rows, very
small p-values are expected even for small differences. A reviewer can raise this. The
regression section avoids it by testing per-patient errors.

---

## 5. Not covered here

- No code change was made to the classification models. The numbers above come from the
  current code with the shared preprocessing changes.
- Optional work not done: grid search for the classification XGBoost; patient-level
  significance test for classification.
