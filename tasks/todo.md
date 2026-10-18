# Reviewer comment: "Να τρέξουμε κι άλλα μοντέλα regression (e.g. random forest)"

Scope: regression pipeline only (`src/regression/regression_pipeline.py`).
Decisions taken: add Random Forest + XGBoost; tune all models; fix the problems found in review.
Reference run (before any change): `output/regression_20261018-165121.txt`.

Rule: one commit per step. Rerun the full-cohort regression after each step and compare it
with the previous run, so that every change in the numbers has one known cause.

## Phase 0 — Reproducible starting point

- [x] 0.1 Run `clean_generated_files.sh`, then rerun `python src/main.py --mode regression`.
      → result: the caches dated from 2026-06-19, before the physiologic-range cleaning.
        New reference run: `output/regression_20261018-171155.txt` (≈ 22 min incl. data
        regeneration). Split: 279 / 93 / 93 patients. Test metrics changed only slightly.
        Use the conda env: `conda run -n icu-data-analysis ...`.
- [x] 0.2 Run `pytest` → 1 stale test failed (`cpp=-40` outside the new range [-20, 200]
      from commit 96500a8). Test value changed to `-15`. Now 15/15 pass.

## Phase 1 — Fixes (no new models yet)

Done. Run: `output/regression_20261018-173541.txt`. Test metrics unchanged; R² = old
"Accuracy" / 100; Lasso CV mean 3.49271 → 3.49256 (per-fold scaling); "Drop PEEP, PH, SPO2"
now differs from "Full Dataset" (LR 3.43971 → 3.43941); "Lags + Pathology" LR 3.43983.
Open question for 1.3: "Extensive Drop" kept unchanged until the authors confirm it.

- [x] 1.1 Wrap each model in a `sklearn.pipeline.Pipeline` (`StandardScaler` → model for the
      linear models, model only for the baselines). Remove the shared `self.scaler`.
      This fixes problem 1 (shared scaler) and problem 2 (CV scaler fitted outside the folds).
      → verify: test metrics are unchanged; CV numbers change only slightly.
      → test: a unit test that checks that `cross_val_score` gets a `Pipeline` and that the
        scaler is fitted per fold.
- [x] 1.2 Rename "Accuracy (%)" to "R²" and report it as a value in [−∞, 1], not as a percentage.
      → verify: same values / 100.
- [x] 1.3 Fix the feature configurations. "Drop PEEP, PH, SPO2" must drop the `peep_lag_*`,
      `ph_lag_*` and `spo2_lag_*` columns. Check "Extensive Drop" for the same defect
      (it keeps `temperature_lag_2/3` and `mean_blood_pressure_lag_4`; confirm with the
      authors if this is intended).
      → verify: the "Drop PEEP, PH, SPO2" column differs from "Full Dataset".
      → test: a unit test that checks that each configuration removes the expected columns.
- [x] 1.4 Add a fifth feature configuration "Lags + Pathology" (lag features + pathology
      one-hot columns). It is an extra experiment, not the default.
- [x] 1.5 Feature importance: stop refitting the models in `__compute_feature_importance`;
      reuse the fitted pipelines. Print signed coefficients next to the absolute ranking.

## Phase 2 — Investigate the Linear Regression CV anomaly (problem 5)

- [x] Resolved by Phase 0: stale caches without the physiologic-range cleaning. With fresh
      data, Linear Regression worst fold = 4.07, SD = 0.38. No code change needed.
      Documented in `notes/linear_regression_cv_anomaly.md`.

## Phase 3 — New models and tuning

- [ ] 3.1 Add `RandomForestRegressor` (no scaling, `n_jobs=-1`, `random_state=42`).
- [ ] 3.2 Add `XGBRegressor` (no scaling, `n_jobs=-1`, `random_state=42`).
- [ ] 3.3 Tune all ML models on the validation split (same protocol for all):
      - Ridge: `alpha` ∈ {0.01, 0.1, 1, 10, 100}
      - Lasso: `alpha` ∈ {0.001, 0.01, 0.1, 1}
      - RF: `n_estimators` ∈ {200, 500}, `max_depth` ∈ {None, 10, 20},
        `min_samples_leaf` ∈ {1, 5, 20}, `max_features` ∈ {1.0, "sqrt"}
      - XGB: `n_estimators` ∈ {200, 500}, `max_depth` ∈ {3, 6, 9},
        `learning_rate` ∈ {0.03, 0.1}, `subsample` ∈ {0.8, 1.0}
      - Linear Regression and the baselines have no hyperparameters.
      Select by validation RMSE; refit on train; report on test once.
      Print the selected hyperparameters (needed for the manuscript).
- [ ] 3.4 Permutation importance (validation split) for RF and XGB.
- [ ] 3.5 Optional (decide at review): Wilcoxon signed-rank test on per-patient MAE,
      best tree model vs. best linear model, on the test split.
- [ ] 3.6 Check the run time. Grouped 10-fold CV × RF can be slow at ~100k rows.

### Phase 3 status (run `output/regression_20261018-173849.txt`)

- [x] 3.1–3.4 implemented; 3.5 not done; 3.6 run time acceptable.
- Selected: Ridge α=10, Lasso α=0.01, RF depth 10 / leaf 20 / max_features 1.0,
  XGB depth 3 / lr 0.03 / 500 trees / subsample 0.8. RF leaf and XGB depth/lr are at the grid edge.
- RMSE: test RF 3.518, XGB 3.519, LR 3.578 | validation RF 3.61, XGB 3.84, LR 3.44 |
  CV RF 3.482, LR 3.484, XGB 3.538. The ranking is not consistent across splits.
- Cause (scratchpad `extrap.py`): trees cannot predict above the training range (max prediction
  ≈ 112 vs. true values up to 160). Rows with ICP > 50 are 2 % of validation rows but 37–45 %
  of the tree models' squared error. For ICP ≤ 20 (≈ 94 % of rows) trees are better
  (RMSE 2.14 vs. 2.32). Validation has 84 rows with ICP > 100 (6 patients), test has 2.
- The Z-score (7) filter does NOT cap ICP at 100 mmHg (values up to 160 remain). Methods line 69
  of the manuscript says it does.
- [x] DECISION: enforce ICP ≤ 100 mmHg (STEP 8, after the Z-score step): 163 rows / 17 patients removed.
- [x] DECISION: metrics stratified by true ICP (≤ 20, 20–30, > 30) added to the regression report.
- [x] Reviewer request: PEEP excluded from the analysis (STEP 0, `EXCLUDED_VARIABLES`).

### Run after ICP ≤ 100 + PEEP removal (`output/regression_20261018-181758.txt`)

- Rows: train 70,141 / validation 27,191 / test 24,035.
- RMSE: test RF 3.483, XGB 3.487, LR 3.540 | validation RF 2.905, XGB 2.931, LR 2.969 |
  CV RF 3.325, XGB 3.349, LR 3.363. RF is now best on all three; the gain is ≈ 1–2 %.
- By ICP range (test): trees best for ICP ≤ 20 (RMSE 2.72 vs. LR 2.88); for 20–30 and > 30,
  persistence and LR are better than the trees.

## Phase 4 — Final runs for the manuscript

- [ ] 4.1 Full cohort: `filter_by_pathology = ""` → run, save output.
- [ ] 4.2 TBI subgroup: `filter_by_pathology = "TBI"` → run, save output.
      WARNING: the cached CSVs do not include the cohort in their file names. Run
      `clean_generated_files.sh` before each cohort switch.
- [ ] 4.3 Fill the `[TBD]` values in `notes/manuscript_corrections_regression.md`.
- [ ] 4.4 Regenerate Tables 3–8 for the manuscript.

## Out of scope (flag only)

- The classification results in the manuscript probably also come from the older code
  (two-way split). This plan does not cover them.

## Review

(to be filled after completion)
