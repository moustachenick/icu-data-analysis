# Linear Regression CV anomaly

## Symptom

Run `output/regression_20261018-165121.txt`, grouped 10-fold CV on the training split:

| Model | Mean CV RMSE | SD | Worst fold |
|---|---|---|---|
| Linear Regression | 8.21 | 14.30 | 51.07 |
| Ridge Regression | 4.02 | 1.78 | 9.18 |
| Lasso Regression | 3.48 | 0.46 | 4.22 |

The unregularised model was the most affected. This points to extreme feature values in a few
held-out patients, on which the linear model extrapolates.

## Cause

The run used cached CSVs from 2026-06-19. `clean_physiologic_outliers()` (commits db56ce5 and
96500a8, August 2026) did not exist when those files were created. The cached data still
contained physiologically impossible values, e.g. pH = 21 for patient 2984.

The pipeline reuses a cached CSV when the file exists. A code change in preprocessing therefore
has no effect until the caches are removed.

## Resolution

After `clean_generated_files.sh` and a fresh run (`output/regression_20261018-171155.txt`):

| Model | Mean CV RMSE | SD | Worst fold |
|---|---|---|---|
| Linear Regression | 3.48 | 0.38 | 4.07 |
| Ridge Regression | 3.48 | 0.38 | 4.07 |
| Lasso Regression | 3.49 | 0.38 | 4.07 |

No code change was needed. Test-split metrics changed only in the third decimal place.

## Lesson

Remove the cached CSVs after any change in `DataParser` or `DataPreProcessor`, and before
any run whose numbers go into the manuscript.
