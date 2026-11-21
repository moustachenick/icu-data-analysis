import itertools
import math
import pandas as pd
from classification.latest_icp_baseline_predictor import LatestICPBaselinePredictor
from classification.lagged_icp_baseline_predictor import LaggedICPBaselinePredictor
from classification.xgboost_classification_predictor import XGBoostClassificationPredictor
from classification.episode_transitions import (
    ONSET, STABLE, assign_transition_categories, low_before_mask, onset_summary_table, restrict_result_to_rows,
)
from helper.data_frame_printer import DataFramePrinter
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
import numpy as np
from tqdm import tqdm
from statsmodels.stats.contingency_tables import mcnemar
from tabulate import tabulate
import matplotlib.pyplot as plt

# Clinical baseline: at the bedside a patient is considered at risk once ICP leaves the normal
# range (7-15 mmHg). The statistical baseline (persistence) uses the outcome threshold, 22 mmHg.
CLINICAL_RULE_THRESHOLD = 15
CLINICAL_RULE_NAME = "LatestICPBaselinePredictor_15"


def threshold_for_recall(y_true, y_score, target_recall):
    """Highest score threshold whose recall on (y_true, y_score) is at least target_recall."""
    positive_scores = np.sort(np.asarray(y_score)[np.asarray(y_true) == 1])[::-1]
    k = max(1, math.ceil(target_recall * len(positive_scores)))
    return positive_scores[k - 1]


def format_p(p):
    return "< 0.0000000001" if p < 1e-10 else f"{p:.10f}"


class ClassificationPipeline:
    """
    A class that orchestrates the classification pipeline for predicting abnormal Intracranial Pressure (ICP) values.
    It includes preparation of datasets, running baseline predictors, and XGBoost classification.
    It also performs McNemar's test to compare the performance of different models.
    """

    # Fixed XGBoost settings; the tuned values from xgb_param_grid are added to them.
    xg_boost_model_params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'colsample_bytree': 0.8
        }

    # Grid searched on the VALIDATION split. Selection by PR-AUC: threshold-free, and suited to
    # the ~5% positive rate (accuracy or F1 at 0.5 would depend on the decision threshold).
    xgb_param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.03, 0.1],
        'n_estimators': [200, 500],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 6, 18],
    }

    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.tuned_params = None  # {"all": params, "lagonly": params}, set by run_pipeline()

    def run_pipeline(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # Tune on VALIDATION, compare the variants on VALIDATION, then produce a single final
        # report on the held-out TEST split.
        print("\n################ TUNING — XGBoost grid search on the VALIDATION split ################")
        lag_only = XGBoostClassificationPredictor.lag_only_feature_selector
        self.tuned_params = {
            "all": self._tune_xgboost(X_train, X_val, y_train, y_val, None),
            "lagonly": self._tune_xgboost(X_train, X_val, y_train, y_val, lag_only),
        }

        print("\n################ VALIDATION SPLIT — model/feature selection ################")
        val_results = self._evaluate_on_holdout(X_train, X_val, y_train, y_val, holdout_name="VALIDATION", plot=False)

        print("\n################ TEST SPLIT — final held-out report ################")
        test_results = self._evaluate_on_holdout(X_train, X_test, y_train, y_test, holdout_name="TEST", plot=True)
        self._report_matched_sensitivity(val_results, test_results)
        return test_results

    def _tune_xgboost(self, X_train, X_val, y_train, y_val, feature_selector):
        best_params, best_score = None, -np.inf
        for grid_params in ParameterGrid(self.xgb_param_grid):
            params = {**self.xg_boost_model_params, **grid_params}
            result = XGBoostClassificationPredictor(model_params=params, feature_selector=feature_selector) \
                .run_pipeline(X_train, X_val, y_train, y_val)
            score = average_precision_score(result["y_true"], result["y_score"])
            if score > best_score:
                best_params, best_score = params, score
        label = "ICP lags only" if feature_selector else "all lags"
        selected = {key: best_params[key] for key in self.xgb_param_grid}
        print(f"XGBoost ({label}): selected {selected} (validation PR-AUC = {best_score:.4f})")
        return best_params

    def _make_predictors(self):
        if self.tuned_params is None:
            raise ValueError("run_pipeline() must run first: it selects the tuned XGBoost parameters.")
        lag_only = XGBoostClassificationPredictor.lag_only_feature_selector
        baselines = [
            LaggedICPBaselinePredictor(),
            LatestICPBaselinePredictor(),
            LatestICPBaselinePredictor(decision_threshold=CLINICAL_RULE_THRESHOLD, name=CLINICAL_RULE_NAME),
        ]
        xgb_models = [
            XGBoostClassificationPredictor(),
            XGBoostClassificationPredictor(model_params=self.tuned_params["all"]),
            XGBoostClassificationPredictor(feature_selector=lag_only),
            XGBoostClassificationPredictor(model_params=self.tuned_params["lagonly"], feature_selector=lag_only),
        ]
        return baselines, xgb_models

    def _evaluate_on_holdout(self, X_train, X_holdout, y_train, y_holdout, holdout_name="TEST", plot=True):
        # Train each variant on X_train and evaluate it on the given holdout split.
        baselines, xgb_models = self._make_predictors()
        results = [b.run_pipeline(X_holdout, y_holdout) for b in baselines]
        results += [m.run_pipeline(X_train, X_holdout, y_train, y_holdout) for m in xgb_models]
        baseline_results, xgb_results = results[:len(baselines)], results[len(baselines):]

        print(f"\n=== CLASSIFICATION RESULTS ({holdout_name}) ===")
        print(f"Number of {holdout_name.lower()} samples: {len(X_holdout)}")
        print(f"Number of positive samples (icp_binary is 1): {sum(y_holdout == 1)}")
        print(f"Number of negative samples (icp_binary is 0): {sum(y_holdout == 0)}")

        print("\n Running McNemar's Test Between Models...\n")
        mcnemar_rows = []
        for result_a, result_b in itertools.combinations(results, 2):
            label = f"{result_a['name']} vs {result_b['name']}"
            p_value = self.run_mcnemar_test(result_a, result_b, label=label)
            mcnemar_rows.append([label, format_p(p_value), "✓" if p_value < 0.05 else "✗"])
        print(tabulate(mcnemar_rows, headers=["Comparison", "p-value", "Significant?"], tablefmt="fancy_grid"))

        # Prepare a summary table
        summary = []
        for res in results:
            report = res['classification_report']
            class_1 = report.get('1', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
            summary.append({
                'Predictor': res['name'],
                'Accuracy': f"{report.get('accuracy', 0.0):.3f}",
                'Precision (1)': f"{class_1['precision']:.3f}",
                'Recall (1)': f"{class_1['recall']:.3f}",
                'F1-score (1)': f"{class_1['f1-score']:.3f}",
                'AUROC': f"{roc_auc_score(res['y_true'], res['y_score']):.3f}",
                'PR-AUC': f"{average_precision_score(res['y_true'], res['y_score']):.3f}",
            })

        summary_df = pd.DataFrame(summary)
        DataFramePrinter.print_dataframe_tabulated(summary_df, f"Classification Models Comparison ({holdout_name})")

        print(f"\n=== CASES WHERE XGBOOST IS CORRECT AND BASELINE IS WRONG ({holdout_name}, on ICP ≥ 22mmHg) ===\n")
        for baseline_result in baseline_results:
            for xgb_result in xgb_results:
                self.count_cases_where_xgb_correct_baseline_wrong(
                    xgb_result, baseline_result, f"{xgb_result['name']} vs {baseline_result['name']}")

        # Extra evaluation on the rows whose preceding hour was low (TEST only, descriptive).
        # It reuses the predictions above: nothing is retrained and nothing is used for model selection.
        if holdout_name == "TEST":
            y_all = np.asarray(y_holdout)
            for res in results:
                if not np.array_equal(res["y_true"], y_all):
                    raise ValueError(f"{res['name']}: y_true is not aligned with the holdout rows.")

            categories = assign_transition_categories(X_holdout["icp_lag_1"], y_all)
            low_before_rows = low_before_mask(categories)
            onset_results = [restrict_result_to_rows(res, low_before_rows) for res in results]

            print("\n=== ONSET EVALUATION: rows whose preceding hour was low ===")
            print(f"Onset rows (low -> high, positive): {np.sum(categories == ONSET)}")
            print(f"Stable rows (low -> low, negative): {np.sum(categories == STABLE)}")
            DataFramePrinter.print_dataframe_tabulated(
                onset_summary_table(onset_results),
                "Onset Evaluation (TEST): rows whose preceding hour was low"
            )

            print("\n Running McNemar's Test Between Models (onset evaluation)...\n")
            onset_mcnemar_rows = []
            for result_a, result_b in itertools.combinations(onset_results, 2):
                label = f"{result_a['name']} vs {result_b['name']}"
                p_value = self.run_mcnemar_test(result_a, result_b, label=label)
                onset_mcnemar_rows.append([label, format_p(p_value), "✓" if p_value < 0.05 else "✗"])
            print(tabulate(onset_mcnemar_rows, headers=["Comparison", "p-value", "Significant?"], tablefmt="fancy_grid"))

        # Plot feature importances (only for the final test report to avoid overwriting the PNG twice).
        if plot:
            xgb_default, _, xgb_lagonly, xgb_lagonly_tuned = xgb_models
            self.plot_xgboost_feature_importances(xgb_default.model)
            self.plot_xgboost_feature_importances(xgb_lagonly.model, title="XGBoost Feature Importances (Lag Only)",
                                                  filename="xgboost_feature_importance_lag_only.png")
            self.plot_xgboost_feature_importances(xgb_lagonly_tuned.model, title="XGBoost Feature Importances (Lag Only, Tuned)",
                                                  filename="xgboost_feature_importance_lag_only_tuned.png")

        return results

    def _report_matched_sensitivity(self, val_results, test_results):
        """
        Clinical comparison at the same sensitivity: each XGBoost threshold is chosen on VALIDATION
        so that it catches the same share of cases as the clinical rule (ICP >= 15) catches on
        VALIDATION; the threshold is then applied once to TEST.
        """
        val = {r["name"]: r for r in val_results}
        test = {r["name"]: r for r in test_results}
        target_recall = val[CLINICAL_RULE_NAME]["classification_report"]["1"]["recall"]

        def operating_point(name, y_true, y_pred, threshold):
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            return {"Predictor": name, "Threshold (from validation)": threshold, "Caught (TP)": tp,
                    "Missed (FN)": fn, "False alarms (FP)": fp, "Recall (1)": f"{tp / (tp + fn):.3f}",
                    "Precision (1)": f"{tp / max(tp + fp, 1):.3f}"}

        clinical = test[CLINICAL_RULE_NAME]
        rows = [operating_point(CLINICAL_RULE_NAME, clinical["y_true"], clinical["y_pred"], f"ICP >= {CLINICAL_RULE_THRESHOLD}")]
        for name, res in test.items():
            if not name.startswith("XGBoost"):
                continue
            threshold = threshold_for_recall(val[name]["y_true"], val[name]["y_score"], target_recall)
            rows.append(operating_point(name, res["y_true"], (res["y_score"] >= threshold).astype(int), f"{threshold:.4f}"))

        table = pd.DataFrame(rows)
        clinical_fp = rows[0]["False alarms (FP)"]
        table["False alarms vs clinical rule"] = [f"{100 * (r['False alarms (FP)'] - clinical_fp) / clinical_fp:+.1f}%" for r in rows]
        DataFramePrinter.print_dataframe_tabulated(
            table, f"Matched Sensitivity (TEST): thresholds set on VALIDATION to the clinical rule's recall ({target_recall:.3f})"
        )
        return table

    def run_mcnemar_test(self, model_a_result, model_b_result, label=""):
        y_true = model_a_result['y_true']
        y_pred_a = model_a_result['y_pred']
        y_pred_b = model_b_result['y_pred']

        b = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        c = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))

        print(f"\n🔍 McNemar Test: {label}")
        print(f"   ➤ b (A wrong, B right): {b}")
        print(f"   ➤ c (A right, B wrong): {c}")

        if b + c == 0:
            print("McNemar's test not applicable (b + c = 0). Returning p = 1.0.")
            return 1.0

        table = [[0, b], [c, 0]]
        result = mcnemar(table, exact=True)
        p = result.pvalue

        if p < 1e-10:
            print("   ➤ p-value: < 0.0000000001")
        else:
            print(f"   ➤ p-value: {p:.10f}")  # 10 δεκαδικά ψηφία

        return p
    
    def count_cases_where_xgb_correct_baseline_wrong(self, xgb_result, baseline_result, label=""):
        y_true = xgb_result["y_true"]
        xgb_pred = xgb_result["y_pred"]
        baseline_pred = baseline_result["y_pred"]

        mask = y_true == 1

        correct_by_xgb_wrong_by_baseline = (xgb_pred == 1) & (baseline_pred != 1) & mask
        count = np.sum(correct_by_xgb_wrong_by_baseline)
        total_positive = np.sum(mask)

        percentage = (count / total_positive) * 100 if total_positive > 0 else 0

        print(f"🧠 {label}")
        print(f"   ➤ True Positives (ICP ≥ 22mmHg): {total_positive}")
        print(f"   ➤ XGBoost correct & Baseline wrong: {count} ({percentage:.2f}%)\n")



    def run_cross_validation_pipeline(self, X, y, n_splits=10):
        print("\nRunning 10-fold cross-validation...\n")
        baselines, xgb_models = self._make_predictors()
        predictors = baselines + xgb_models
        fold_results = [[] for _ in predictors]

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in tqdm(sgkf.split(X, y.astype(int), groups=X["patient_id"]),
                                        total=n_splits, desc="Cross-validation folds"):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            for i, predictor in enumerate(predictors):
                if isinstance(predictor, XGBoostClassificationPredictor):
                    fold_results[i].append(predictor.run_pipeline(X_train, X_test, y_train, y_test))
                else:
                    fold_results[i].append(predictor.run_pipeline(X_test, y_test))

        def mean_sd(values):
            return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

        summary = []
        for folds in fold_results:
            reports = [r["classification_report"] for r in folds]
            summary.append({
                "Predictor": folds[0]["name"],
                "Accuracy": mean_sd([r["accuracy"] for r in reports]),
                "Precision (1)": mean_sd([r["1"]["precision"] for r in reports]),
                "Recall (1)": mean_sd([r["1"]["recall"] for r in reports]),
                "F1-score": mean_sd([r["1"]["f1-score"] for r in reports]),
                "AUROC": mean_sd([roc_auc_score(r["y_true"], r["y_score"]) for r in folds]),
                "PR-AUC": mean_sd([average_precision_score(r["y_true"], r["y_score"]) for r in folds]),
            })

        summary_df = pd.DataFrame(summary)
        DataFramePrinter.print_dataframe_tabulated(summary_df, "10-Fold Cross-Validation Results (Mean ± Std Dev)")

    def plot_xgboost_feature_importances(self, model, title="XGBoost Feature Importances",
                                         filename="xgboost_feature_importance.png"):
        importances = model.get_booster().get_score(importance_type='gain')
        importance_df = pd.DataFrame(importances.items(), columns=["Feature", "Importance"])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title(title)

        for bar, score in zip(bars, importance_df["Importance"]):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', ha='left', fontsize=9)
        plt.yticks(fontsize=8)
        plt.gca().invert_yaxis()  
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved clean feature importance plot as '{filename}'")
        plt.close()
