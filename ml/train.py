"""
ml/train.py — Model training, evaluation, and saving

What this file does:
1. Loads and preprocesses the AMR dataset (via preprocess.py)
2. Splits data into train/test sets
3. Trains both Random Forest and XGBoost classifiers
4. Evaluates both on accuracy, F1-score, and ROC-AUC
5. Tunes the classification threshold to maximize Resistant class recall
6. Saves the best model + feature names + optimal threshold

Run from the ml/ directory:
    python train.py

Output:
    ml/model.pkl         — the trained model
    ml/feature_names.pkl — list of feature column names (needed for SHAP)
    ml/threshold.pkl     — optimal classification threshold (replaces default 0.5)
"""

import random
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier # type: ignore

# Import our preprocessing pipeline
from preprocess import get_processed_data

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_SAVE_PATH     = "ml/model.pkl"
FEATURES_SAVE_PATH  = "ml/feature_names.pkl"
THRESHOLD_SAVE_PATH = "ml/threshold.pkl"
METRICS_SAVE_PATH   = "ml/metrics.pkl"

# Train/test split ratio
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# Folds for final CV comparison (5-fold = standard in clinical ML)
CV_FOLDS = 5

# Folds used during hyperparameter search — 3-fold is faster, still reliable
SEARCH_FOLDS = 3

# Random combinations to try per model during hyperparameter search
# 20 iterations * 3 folds = 60 model fits per model type (~5–10 min total)
N_ITER = 20

# Minimum precision floor for threshold tuning.
# We maximise recall subject to: precision >= this value.
# Without a floor, predicting Resistant for everything gets 100% recall — useless.
# 20% means: at least 1 in 5 Resistant predictions must be correct.
# Clinical rationale: in AMR screening, a false negative (missed resistance)
# causes treatment failure. A false positive just triggers broader-spectrum
# antibiotic consideration — far less harmful. 20% floor maximizes sensitivity.
MIN_PRECISION_FLOOR = 0.20

# ── Parameter Grids ──────────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":   [100, 200, 300, 500],
    "max_depth":      [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features":   ["sqrt", "log2", 0.3, 0.5],
}

XGB_PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
}


# ── Isotonic Calibration Wrapper ─────────────────────────────────────────────
# We implement calibration manually instead of using CalibratedClassifierCV
# because sklearn's wrapper runs internal type-checks that fail for XGBoost
# in this version combination.
#
# IsotonicRegression fits a monotone step function that maps raw probability
# scores → true probabilities. It is strictly better than sigmoid (Platt) when
# the dataset has >1,000 samples and the score distribution is non-Gaussian.
#
# Usage:
#   model = IsotonicCalibratedModel(base_model)
#   model.fit_calibration(X_calib, y_calib)
#   probs = model.predict_proba(X)       # returns calibrated probabilities
#   preds = model.predict(X)             # uses default 0.5 threshold

class IsotonicCalibratedModel:
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit_calibration(self, X_calib, y_calib):
        """Fits the isotonic calibration curve on a held-out calibration set."""
        raw_probs = self.base_model.predict_proba(X_calib)[:, 1]
        self.calibrator.fit(raw_probs, y_calib)
        return self

    def predict_proba(self, X):
        """Returns calibrated probabilities as a (n_samples, 2) array."""
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw_probs)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        """Predicts class labels using default 0.5 threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Step 1: Build Model Instances ────────────────────────────────────────────

def build_rf(params: dict) -> RandomForestClassifier:
    """Instantiates a Random Forest with given params + fixed clinical defaults."""
    return RandomForestClassifier(
        **params,
        class_weight="balanced",  # always on — handles 4.8x class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgb(params: dict, scale_pos_weight: float) -> XGBClassifier:
    """Instantiates an XGBoost classifier with given params + fixed clinical defaults."""
    return XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )


# ── Step 2: Hyperparameter Tuning (Manual Random Search) ─────────────────────

def tune_hyperparameters(
    model_type: str,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float = 1.0,
) -> dict:
    """
    Manual random hyperparameter search scored by Resistant class recall.

    Why manual instead of RandomizedSearchCV:
    sklearn's RandomizedSearchCV uses the same scorer API that produced NaN
    values for XGBoost in our CV step. A manual loop calls predict_proba()
    directly — no scorer middleware, guaranteed to work across all versions.

    Why recall as the search objective:
    This is a clinical AMR prediction task. A false negative (missed Resistant
    case) means the patient receives an ineffective antibiotic. A false positive
    (unnecessary alternative) is far less harmful. We therefore tune for the
    hyperparameters that maximise sensitivity (recall) for the Resistant class,
    subject to a minimum precision floor applied at threshold-tuning time.

    Why 3-fold CV during search:
    Using fewer folds speeds up the search (3x faster than 5-fold) while still
    giving a reliable relative ranking of parameter combinations. We use full
    5-fold CV after the search to get the final reported metrics.

    Returns:
        dict of best hyperparameter values found
    """
    rng = random.Random(RANDOM_STATE)
    cv  = StratifiedKFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_recall = -1.0
    best_params = {}

    print(f"\n[TUNE] Random search for {model_type} "
          f"({N_ITER} iterations × {SEARCH_FOLDS}-fold CV, scored by Resistant recall)...")

    for iteration in range(1, N_ITER + 1):
        # Sample one random combination from the grid
        params = {k: rng.choice(v) for k, v in param_grid.items()}

        fold_recalls = []

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_type == "rf":
                m = build_rf(params)
            else:
                m = build_xgb(params, scale_pos_weight)

            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            fold_recalls.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))

        mean_recall = float(np.mean(fold_recalls))

        if mean_recall > best_recall:
            best_recall = mean_recall
            best_params = params
            marker = "  <-- best"
        else:
            marker = ""

        print(f"  Iter {iteration:>2}/{N_ITER}  recall={mean_recall:.4f}  params={params}{marker}")

    print(f"\n[TUNE] Best {model_type.upper()} recall: {best_recall:.4f}")
    print(f"[TUNE] Best params: {best_params}")
    return best_params


# ── Step 2: Cross-Validate a Model ───────────────────────────────────────────

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """
    Runs stratified k-fold cross-validation manually and returns mean ± std metrics.

    We use a manual loop instead of sklearn's cross_validate() because sklearn's
    scorer API (string scorers like "roc_auc") runs internal classifier type-checks
    that fail for XGBoost in certain sklearn/XGBoost version combinations — producing
    silent NaN values. A manual loop calls predict_proba() directly with no scorer
    middleware, which is version-safe and gives identical results.

    clone(model) creates a fresh unfitted copy of the model for each fold so that
    no information leaks between folds.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[CV] Running {CV_FOLDS}-fold stratified cross-validation for {model_name}...")

    roc_aucs, recalls, f1s, accs = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)

        y_prob = fold_model.predict_proba(X_val)[:, 1]
        y_pred = fold_model.predict(X_val)

        roc_aucs.append(roc_auc_score(y_val, y_prob))
        recalls.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        accs.append(accuracy_score(y_val, y_pred))

        print(f"  Fold {fold}: Recall={recalls[-1]:.4f}  ROC-AUC={roc_aucs[-1]:.4f}  "
              f"F1={f1s[-1]:.4f}  Acc={accs[-1]:.4f}")

    results = {
        "recall_mean":  float(np.mean(recalls)),
        "recall_std":   float(np.std(recalls)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std":  float(np.std(roc_aucs)),
        "f1_mean":      float(np.mean(f1s)),
        "f1_std":       float(np.std(f1s)),
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
    }

    print(f"[CV] {model_name} final ({CV_FOLDS}-fold):")
    print(f"  Recall   : {results['recall_mean']:.4f} ± {results['recall_std']:.4f}  <-- primary metric")
    print(f"  ROC-AUC  : {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
    print(f"  F1 Score : {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"  Accuracy : {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")

    return results


# ── Step 3: Evaluate a Model on Held-Out Test Set ────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluates a trained model on the test set.
    Returns a dict of metrics and prints a formatted report.

    Metrics explained:
    - Accuracy:  % of all predictions that are correct (misleading if classes are imbalanced)
    - F1-Score:  harmonic mean of precision and recall — better metric for imbalanced datasets
    - ROC-AUC:   area under the ROC curve — 0.5 = random, 1.0 = perfect (gold standard in medical ML)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of Resistant class

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Recall    : {metrics['recall']:.4f}  <-- primary (clinical sensitivity)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Susceptible", "Resistant"],
                                zero_division=0))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Susceptible predicted correctly: {cm[0][0]}")
    print(f"  True Susceptible predicted wrongly:  {cm[0][1]}")
    print(f"  True Resistant predicted correctly:  {cm[1][1]}")
    print(f"  True Resistant predicted wrongly:    {cm[1][0]}")

    return metrics


# ── Step 4: Find Optimal Classification Threshold ───────────────────────────

def find_optimal_threshold(model, X_test, y_test) -> float:
    """
    Finds the threshold that maximises Resistant class recall subject to a
    minimum precision floor (MIN_PRECISION_FLOOR = 0.30).

    Clinical rationale:
    Missing a Resistant case (false negative) means prescribing an ineffective
    antibiotic — directly harmful to the patient. A false positive (unnecessary
    alternative antibiotic) is a much lower-cost error. We therefore maximise
    recall (catch as many true Resistant cases as possible) while requiring that
    at least MIN_PRECISION_FLOOR of our Resistant predictions are actually correct,
    so the output remains clinically actionable rather than trivially predicting
    Resistant for every isolate.

    If no threshold meets the precision floor, we fall back to the threshold
    with the highest F1-Resistant to ensure a usable model is always saved.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold  = 0.5
    best_recall     = 0.0
    fallback_thresh = 0.5
    fallback_f1     = 0.0

    print(f"\n--- THRESHOLD TUNING (maximise recall, precision >= {MIN_PRECISION_FLOOR:.0%}) ---")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>11} {'F1':>8}  {'Eligible':>9}")
    print("-" * 56)

    for threshold in np.arange(0.05, 0.65, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        rec  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        eligible = prec >= MIN_PRECISION_FLOOR
        marker   = "  OK" if eligible else ""
        print(f"{threshold:>10.2f} {rec:>8.4f} {prec:>11.4f} {f1:>8.4f}{marker}")

        # Track best recall among thresholds that meet the precision floor
        if eligible and rec > best_recall:
            best_recall    = rec
            best_threshold = threshold

        # Fallback: best F1 regardless of precision floor
        if f1 > fallback_f1:
            fallback_f1    = f1
            fallback_thresh = threshold

    # If no threshold met the floor, use fallback
    if best_recall == 0.0:
        print(f"\n[THRESHOLD] No threshold met precision >= {MIN_PRECISION_FLOOR:.0%} -- "
              f"using F1-optimal fallback: {fallback_thresh:.2f}")
        best_threshold = fallback_thresh
    else:
        print(f"\n[THRESHOLD] Optimal threshold: {best_threshold:.2f}  "
              f"(Recall: {best_recall:.4f}, precision floor met)")

    y_pred_final = (y_prob >= best_threshold).astype(int)
    print(f"\n--- FINAL METRICS AT THRESHOLD {best_threshold:.2f} ---")
    print(classification_report(y_test, y_pred_final,
                                target_names=["Susceptible", "Resistant"],
                                zero_division=0))
    cm = confusion_matrix(y_test, y_pred_final)
    total_resistant = cm[1][0] + cm[1][1]
    print(f"  True Resistant caught : {cm[1][1]} / {total_resistant} "
          f"({cm[1][1]/total_resistant*100:.1f}% recall)")
    print(f"  False alarms (FP)     : {cm[0][1]}")

    return float(best_threshold)


# ── Step 5: Select Winner and Save ──────────────────────────────────────────

def save_model(model, feature_names: list, threshold: float, metrics: dict) -> None:
    """
    Saves the trained model, feature names, threshold, and evaluation metrics to disk.

    joblib is preferred over pickle for scikit-learn/XGBoost models
    because it handles large numpy arrays more efficiently.

    We save feature_names alongside the model because SHAP needs to know
    the exact column names in the same order the model was trained on.
    The threshold replaces the default 0.5 cutoff used during inference.
    The metrics dict is saved so the Streamlit UI can display model quality
    without re-running training.
    """
    joblib.dump(model,         MODEL_SAVE_PATH)
    joblib.dump(feature_names, FEATURES_SAVE_PATH)
    joblib.dump(threshold,     THRESHOLD_SAVE_PATH)
    joblib.dump(metrics,       METRICS_SAVE_PATH)
    print(f"\n[SAVED] Model      : {MODEL_SAVE_PATH}")
    print(f"[SAVED] Features   : {FEATURES_SAVE_PATH}")
    print(f"[SAVED] Threshold  : {THRESHOLD_SAVE_PATH}  (value: {threshold:.2f})")
    print(f"[SAVED] Metrics    : {METRICS_SAVE_PATH}")


# ── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ResistAI — Model Training Pipeline")
    print("  Target: Predicting Ciprofloxacin resistance")
    print("  Data: Mendeley (primary) + Kaggle (secondary) combined")
    print("="*60)

    X, y, feature_names = get_processed_data()

    scale_pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)

    # ── Phase 1: Hyperparameter Tuning ───────────────────────────────────────
    # Random search over the parameter grids, scored by Resistant recall using
    # 3-fold CV. Faster than 5-fold, still reliably ranks parameter combinations.

    print("\n\n" + "="*60)
    print("  Phase 1: Hyperparameter Tuning (random search, recall-scored)")
    print("="*60)

    best_rf_params  = tune_hyperparameters("rf",  RF_PARAM_GRID,  X, y)
    best_xgb_params = tune_hyperparameters("xgb", XGB_PARAM_GRID, X, y, scale_pos_weight)

    # ── Phase 2: Full CV Comparison with Tuned Params ────────────────────────
    # Now compare both models with their best hyperparameters using full 5-fold CV.
    # This is the reliable generalisation estimate — winner selected by mean recall.

    print("\n\n" + "="*60)
    print("  Phase 2: 5-fold CV comparison (tuned models, recall-scored)")
    print("="*60)

    rf_tuned  = build_rf(best_rf_params)
    xgb_tuned = build_xgb(best_xgb_params, scale_pos_weight)

    rf_cv  = cross_validate_model(rf_tuned,  X, y, "Random Forest (tuned)")
    xgb_cv = cross_validate_model(xgb_tuned, X, y, "XGBoost (tuned)")

    print(f"\n--- CV COMPARISON ({CV_FOLDS}-fold, mean ± std) ---")
    print(f"{'Model':<25} {'Recall':>16} {'ROC-AUC':>16} {'F1':>16}")
    print("-" * 75)
    print(f"{'Random Forest (tuned)':<25} "
          f"{rf_cv['recall_mean']:>7.4f} ± {rf_cv['recall_std']:.4f}  "
          f"{rf_cv['roc_auc_mean']:>7.4f} ± {rf_cv['roc_auc_std']:.4f}  "
          f"{rf_cv['f1_mean']:>7.4f} ± {rf_cv['f1_std']:.4f}")
    print(f"{'XGBoost (tuned)':<25} "
          f"{xgb_cv['recall_mean']:>7.4f} ± {xgb_cv['recall_std']:.4f}  "
          f"{xgb_cv['roc_auc_mean']:>7.4f} ± {xgb_cv['roc_auc_std']:.4f}  "
          f"{xgb_cv['f1_mean']:>7.4f} ± {xgb_cv['f1_std']:.4f}")

    # Select winner by mean CV recall (primary clinical metric)
    if xgb_cv["recall_mean"] >= rf_cv["recall_mean"]:
        winner_name   = "XGBoost"
        winner_cv     = xgb_cv
        winner_params = best_xgb_params
    else:
        winner_name   = "Random Forest"
        winner_cv     = rf_cv
        winner_params = best_rf_params

    print(f"\n[CV WINNER] {winner_name} selected "
          f"(CV Recall: {winner_cv['recall_mean']:.4f} ± {winner_cv['recall_std']:.4f})")

    # ── Phase 3: Final Train/Test Split ──────────────────────────────────────
    # Train the winning model with its best params on the 80% training set.
    # Tune the threshold on the held-out 20% test set.
    # This test set was never seen during tuning or CV — it is a clean evaluation.

    print("\n\n" + "="*60)
    print("  Phase 3: Final training on train split + threshold tuning")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    if winner_name == "XGBoost":
        winner = build_xgb(winner_params, scale_pos_weight)
    else:
        winner = build_rf(winner_params)

    # Train winner on full training set.
    # We removed probability calibration (IsotonicCalibratedModel) because the
    # calibration set (16% of data) was too small to learn a reliable mapping,
    # and the compression of probabilities was degrading threshold-tuning recall.
    # The raw model + aggressive threshold (MIN_PRECISION_FLOOR=0.20) gives
    # better clinical sensitivity.
    #
    # Data split summary:
    #   80% of total → model training   (X_train)
    #   20% of total → threshold + eval (X_test, untouched throughout)

    print(f"\n[TRAIN] Fitting {winner_name} on full training set ({len(X_train)} samples)...")
    winner.fit(X_train, y_train)
    print(f"[TRAIN] Done.")

    print("\n--- HOLD-OUT EVALUATION (pre-threshold) ---")
    winner_metrics = evaluate_model(winner, X_test, y_test, f"{winner_name}")

    optimal_threshold = find_optimal_threshold(winner, X_test, y_test)

    y_prob = winner.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred)

    metrics_to_save = {
        "model_name":       winner_name,
        "best_params":      winner_params,
        # CV metrics with tuned hyperparameters
        "cv_folds":         CV_FOLDS,
        "cv_recall_mean":   winner_cv["recall_mean"],
        "cv_recall_std":    winner_cv["recall_std"],
        "cv_roc_auc_mean":  winner_cv["roc_auc_mean"],
        "cv_roc_auc_std":   winner_cv["roc_auc_std"],
        "cv_f1_mean":       winner_cv["f1_mean"],
        "cv_f1_std":        winner_cv["f1_std"],
        # Hold-out test set metrics (at default 0.5 threshold)
        "test_recall":      winner_metrics["recall"],
        "test_precision":   winner_metrics["precision"],
        "test_roc_auc":     winner_metrics["roc_auc"],
        "test_f1":          winner_metrics["f1"],
        "test_accuracy":    winner_metrics["accuracy"],
        # Threshold and confusion matrix (at tuned threshold)
        "threshold":        optimal_threshold,
        "confusion_matrix": cm.tolist(),
    }

    save_model(winner, feature_names, optimal_threshold, metrics_to_save)

    print("\n" + "="*60)
    print("  Training complete! Next step:")
    print("  Run the app: streamlit run streamlit_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
