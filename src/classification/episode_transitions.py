import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from data_parser.binary_data_processor import ICP_BINARY_THRESHOLD

# Row types, named by (state in the preceding hour -> state at the row's own time point).
# "Low/high before" is read from icp_lag_1.
ONSET = "onset"                # low -> high
CONTINUATION = "continuation"  # high -> high
STABLE = "stable"              # low -> low
RESOLUTION = "resolution"      # high -> low

def assign_transition_categories(icp_lag_1, y, threshold=ICP_BINARY_THRESHOLD):
    """
    Label every row by the transition from the preceding hour to the row's own outcome.

    The preceding hour is "high" when icp_lag_1 >= threshold. The outcome is the true binary label y (1 = ICP >= threshold).

    """
    lag = np.asarray(icp_lag_1, dtype=float)
    if np.isnan(lag).any():
        raise ValueError("icp_lag_1 contains missing values.")

    high_before = lag >= threshold
    high_now = np.asarray(y).astype(int) == 1

    categories = np.empty(len(lag), dtype=object)
    categories[~high_before & high_now] = ONSET
    categories[high_before & high_now] = CONTINUATION
    categories[~high_before & ~high_now] = STABLE
    categories[high_before & ~high_now] = RESOLUTION
    return categories

def low_before_mask(categories):
    """
    True for the rows whose preceding hour was low: onset (low -> high) and stable (low -> low).
    These are the rows where an episode could start.
    """
    categories = np.asarray(categories, dtype=object)
    return (categories == ONSET) | (categories == STABLE)

def restrict_result_to_rows(result, row_mask):
    """
    Recompute a predictor's evaluation on a subset of the holdout rows.

    """
    row_mask = np.asarray(row_mask, dtype=bool)
    y_true = np.asarray(result["y_true"])[row_mask]
    y_pred = np.asarray(result["y_pred"])[row_mask]
    return {
        "name": result["name"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": np.asarray(result["y_score"])[row_mask],
    }


def onset_summary_table(subset_results):
    """
    One row per predictor, computed on the rows whose preceding hour was low.

    Positive = onset (low -> high), negative = stable (low -> low).
    """
    rows = []
    for res in subset_results:
        tn, fp, fn, tp = res["confusion_matrix"].ravel()
        report = res["classification_report"]["1"]
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        rows.append({
            "Predictor": res["name"],
            "Onsets caught (TP)": int(tp),
            "Onsets missed (FN)": int(fn),
            "False alarms (FP)": int(fp),
            "Recall (1)": f"{report['recall']:.3f}",
            "Precision (1)": f"{report['precision']:.3f}",
            "F1-score (1)": f"{report['f1-score']:.3f}",
            "False alarm rate": f"{false_alarm_rate:.4f}",
            "AUROC": f"{roc_auc_score(res['y_true'], res['y_score']):.3f}",
            "PR-AUC": f"{average_precision_score(res['y_true'], res['y_score']):.3f}",
        })
    return pd.DataFrame(rows)