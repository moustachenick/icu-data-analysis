# KNN Imputation — Reviewer Comments & Analysis

## What the current code does

`DataPreProcessor.known_nearest_neighbor_imputer()` in `src/data_parser/data_pre_processor.py`:

- Only rows with **exactly 1 missing value** are imputed (`one_missing_df`)
- Rows with 0 missing values (`no_missing_df`) are **excluded from the KNN pool** — this is unusual; normally you would fit on complete cases
- `patient_id`, `date_of_birth`, and `timestamp` are dropped before distance computation, so **any row from any patient at any time** can be the nearest neighbor
- `KNNImputer(n_neighbors=1)` is fitted and transformed on `one_missing_df` only
- Imputation happens **before** the 80/20 train/test split in `main.py`

---

## Reviewer Comment 1

> "Could this be an issue? What if the imputed value is drawn from a totally unrelated patient? Do we know how the literature addresses this in time-series clinical data?"

### Response

Cross-patient imputation is widely used in clinical ML (van Buuren & Groothuis-Oudshoorn, 2011; Sterne et al., 2009). It is defensible under the **Missing At Random (MAR)** assumption — that missingness is driven by observed values (equipment schedules, care protocols) rather than the unobserved value itself. For ICU monitoring, this assumption is reasonable.

The key guarantee is that the nearest neighbor is **physiologically similar**: for a row missing `glucose`, the KNN selects the row whose *other* 11 clinical variables are closest in Euclidean space — not an arbitrary patient.

Importantly, within-patient forward-fill (`DataParser.fix_missing_values()`) has already run before this step. The KNN step is a secondary cleanup for the residual rows that still have exactly 1 missing value after forward-fill. The fraction of rows actually imputed by KNN is therefore small.

We should add a diagnostic (see "Proposed Code Changes" below) to empirically quantify how often the selected neighbor comes from a different patient.

---

## Reviewer Comment 2

> "You could do an analysis here. For example how often the nearest instance was from the same patient and how often from another patient. Do we care about the sequence of instances? For example can I use a nearest instance from a later period of time. If Yes is that a problem? μήπως κλέβουμε αν το κάνουμε αυτό;"

### Response

Two concerns need to be separated:

#### a) Cross-patient borrowing

Yes, this happens — `patient_id` is excluded from the distance metric, so the nearest neighbor may belong to a different patient. This is addressed by the MAR argument above and is quantifiable via the proposed diagnostic.

#### b) Temporal leakage — "Are we cheating?" (μήπως κλέβουμε;)

**YES — technically.** Two forms of leakage exist:

1. **Temporal leakage**: A future-timestamped row can serve as the nearest neighbor for a past-timestamped row. At inference time, future data is unavailable, so this violates causal ordering.

2. **Cross-split leakage**: Imputation runs on the *full dataset* before the train/test split. A test-set row can therefore serve as the nearest neighbor for a train-set row, inflating apparent model performance.

**How serious is it in practice?** The effect is likely small because:
- Only rows with exactly 1 missing value are imputed
- Only a single feature per row is imputed
- The imputed value is drawn from the physiologically nearest row

The methodologically correct approach is to fit the imputer on training data only and transform train and test separately. For now we document this as a known limitation and add the diagnostic to make the empirical extent visible. Fixing it architecturally is tracked as a future improvement.

---

## Proposed Code Changes

### 1. New `_diagnose_knn_neighbors()` method

Add this static method to `DataPreProcessor`. It approximates which row would be chosen as the KNN neighbor for each imputed row, using `NearestNeighbors` with mean-filled NaN (a close approximation to sklearn's partial-distance approach). Reports % same-patient and % future-timestamp neighbors.

```python
@staticmethod
def _diagnose_knn_neighbors(one_missing_df: pd.DataFrame, ignored_data: pd.DataFrame) -> None:
    """Approximate diagnostic: how often is the KNN neighbor cross-patient or from the future?"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.impute import SimpleImputer

    n = len(one_missing_df)
    if n == 0:
        return

    # Mean-fill NaN to enable Euclidean distance (approximation of sklearn's partial-distance)
    temp_filled = SimpleImputer(strategy='mean').fit_transform(one_missing_df)
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn.fit(temp_filled)
    _, indices = nn.kneighbors(temp_filled)
    # indices[:, 0] = self (distance 0); indices[:, 1] = nearest other row
    neighbor_pos = indices[:, 1]

    original_idx   = one_missing_df.index.tolist()
    source_pids    = ignored_data.loc[original_idx, 'patient_id'].values
    neighbor_pids  = np.array([ignored_data.loc[original_idx[i], 'patient_id'] for i in neighbor_pos])
    source_ts      = pd.to_datetime(ignored_data.loc[original_idx, 'timestamp']).values
    neighbor_ts    = np.array([pd.to_datetime(ignored_data.loc[original_idx[i], 'timestamp'])
                               for i in neighbor_pos])

    same_patient    = int((source_pids == neighbor_pids).sum())
    future_neighbor = int((neighbor_ts > source_ts).sum())

    print(f"\nKNN Imputation Neighbor Diagnostics ({n:,} rows imputed)")
    print(f"  Same-patient neighbor     : {same_patient:,} / {n:,} ({100 * same_patient / n:.1f}%)")
    print(f"  Future-timestamp neighbor : {future_neighbor:,} / {n:,} ({100 * future_neighbor / n:.1f}%)")
    print("  (Note: imputation runs before train/test split — cross-patient and future borrowing is possible)\n")
```

### 2. Call site in `known_nearest_neighbor_imputer()`

Call `_diagnose_knn_neighbors(one_missing_df, ignored_data)` after constructing `one_missing_df` and before calling `imputer.fit_transform()`.

### 3. Explanatory comment

Add above the `KNNImputer` instantiation:

```python
# NOTE: The KNN pool is one_missing_df only (complete rows are excluded — atypical).
# patient_id and timestamp are excluded from the distance metric, so cross-patient
# and future-row neighbors are possible. Imputation runs before the train/test split,
# introducing potential cross-split leakage. The diagnostic below quantifies the extent.
```

---

## Future improvement (not yet implemented)

Move imputation to after the train/test split in `main.py`:
- Fit `KNNImputer` on training rows only
- Transform training rows and test rows separately

This eliminates cross-split leakage entirely.
