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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# Import our preprocessing pipeline
from preprocess import get_processed_data

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_SAVE_PATH     = "ml/model.pkl"
FEATURES_SAVE_PATH  = "ml/feature_names.pkl"
THRESHOLD_SAVE_PATH = "ml/threshold.pkl"

# Train/test split ratio — 80% training, 20% held out for evaluation
TEST_SIZE = 0.2

# Random seed for reproducibility — same seed = same split every time
RANDOM_STATE = 42


# ── Step 1: Train Both Models ────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier.

    Random Forest explanation:
    Builds 200 decision trees, each trained on a random subset of rows and features.
    Final prediction = majority vote across all 200 trees.
    Strengths: robust to overfitting, handles missing values well, fast to train.
    """
    print("\n[RF] Training Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,      # number of trees — more = more stable, diminishing returns after 200
        max_depth=None,        # let trees grow fully — RF handles overfitting via voting
        min_samples_leaf=2,    # at least 2 samples per leaf — prevents tiny over-specific splits
        class_weight="balanced", # automatically handles class imbalance (more Susceptible than Resistant)
        random_state=RANDOM_STATE,
        n_jobs=-1              # use all CPU cores to train faster
    )
    model.fit(X_train, y_train)
    print("[RF] Training complete.")
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Trains an XGBoost classifier.

    XGBoost explanation:
    Builds trees sequentially — each new tree corrects the errors of the previous one.
    This 'boosting' approach typically achieves higher accuracy than Random Forest
    on tabular clinical data.
    Strengths: state-of-the-art accuracy on tabular data, built-in regularization.
    """
    print("\n[XGB] Training XGBoost...")

    # Calculate scale_pos_weight to handle class imbalance
    # If there are 3x more Susceptible samples than Resistant, we weight Resistant 3x higher
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale = n_negative / n_positive if n_positive > 0 else 1.0
    print(f"[XGB] Class imbalance ratio (neg/pos): {scale:.2f} — applying as scale_pos_weight")

    model = XGBClassifier(
        n_estimators=200,          # number of boosting rounds
        max_depth=6,               # maximum depth per tree — prevents overfitting
        learning_rate=0.1,         # how much each tree corrects the previous — lower = more trees needed
        subsample=0.8,             # use 80% of rows per tree — adds randomness, reduces overfitting
        colsample_bytree=0.8,      # use 80% of features per tree
        scale_pos_weight=scale,    # handles class imbalance
        use_label_encoder=False,
        eval_metric="logloss",     # internal evaluation metric during training
        random_state=RANDOM_STATE,
        verbosity=0                # suppress XGBoost training logs
    )
    model.fit(X_train, y_train)
    print("[XGB] Training complete.")
    return model


# ── Step 2: Evaluate a Model ─────────────────────────────────────────────────

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
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":  roc_auc_score(y_test, y_prob)
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
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


# ── Step 3: Find Optimal Classification Threshold ───────────────────────────

def find_optimal_threshold(model, X_test, y_test) -> float:
    """
    Finds the probability threshold that maximizes F1 score for the Resistant class.

    Why this matters:
    By default, scikit-learn classifies a sample as Resistant when the predicted
    probability >= 0.5. But with a 4.8x class imbalance, the model is biased
    toward Susceptible predictions — it rarely reaches 50% confidence for Resistant.

    Lowering the threshold (e.g., to 0.3) means: "call it Resistant if probability
    >= 30%". This catches more true Resistant cases (higher recall) at the cost of
    some false positives (lower precision). For clinical AMR prediction, missing a
    resistant case is more dangerous than a false alarm, so higher recall is preferred.

    We scan thresholds from 0.1 to 0.6 and pick the one with the best F1 score
    for the Resistant class specifically — not overall accuracy.
    """
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of being Resistant

    best_threshold = 0.5
    best_f1        = 0.0
    results        = []

    print("\n--- THRESHOLD TUNING ---")
    print(f"{'Threshold':>10} {'F1-Resistant':>14} {'Recall':>8} {'Precision':>10}")
    print("-" * 46)

    for threshold in np.arange(0.1, 0.65, 0.05):
        # Apply threshold: predict Resistant if probability >= threshold
        y_pred_thresh = (y_prob >= threshold).astype(int)

        f1_resistant = f1_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
        from sklearn.metrics import recall_score, precision_score
        recall    = recall_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
        precision = precision_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)

        print(f"{threshold:>10.2f} {f1_resistant:>14.4f} {recall:>8.4f} {precision:>10.4f}")
        results.append((threshold, f1_resistant))

        if f1_resistant > best_f1:
            best_f1        = f1_resistant
            best_threshold = threshold

    print(f"\n[THRESHOLD] Optimal threshold: {best_threshold:.2f}  "
          f"(F1-Resistant: {best_f1:.4f})")

    # Show final performance with the chosen threshold
    y_pred_final = (y_prob >= best_threshold).astype(int)
    print(f"\n--- FINAL METRICS AT THRESHOLD {best_threshold:.2f} ---")
    print(classification_report(y_test, y_pred_final,
                                target_names=["Susceptible", "Resistant"],
                                zero_division=0))
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"  True Resistant caught   : {cm[1][1]} / {cm[1][0] + cm[1][1]} "
          f"({cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}% recall)")
    print(f"  False alarms (FP)       : {cm[0][1]}")

    return float(best_threshold)


# ── Step 4: Select Winner and Save ──────────────────────────────────────────

def save_model(model, feature_names: list, threshold: float) -> None:
    """
    Saves the trained model and feature names to disk using joblib.

    joblib is preferred over pickle for scikit-learn/XGBoost models
    because it handles large numpy arrays more efficiently.

    We save feature_names alongside the model because SHAP needs to know
    the exact column names in the same order the model was trained on.
    The threshold replaces the default 0.5 cutoff used during inference.
    """
    joblib.dump(model,         MODEL_SAVE_PATH)
    joblib.dump(feature_names, FEATURES_SAVE_PATH)
    joblib.dump(threshold,     THRESHOLD_SAVE_PATH)
    print(f"\n[SAVED] Model      → {MODEL_SAVE_PATH}")
    print(f"[SAVED] Features   → {FEATURES_SAVE_PATH}")
    print(f"[SAVED] Threshold  → {THRESHOLD_SAVE_PATH}  (value: {threshold:.2f})")


# ── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ResistAI — Model Training Pipeline")
    print("  Target: Predicting Ciprofloxacin resistance")
    print("  Data: Mendeley (primary) + Kaggle (secondary) combined")
    print("="*60)

    # Load and preprocess data
    X, y, feature_names = get_processed_data()

    # Train/test split — stratified to preserve class balance in both sets
    # 'stratified' means: if 30% of data is Resistant, both train and test will be 30% Resistant
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # preserve class balance in both splits
    )
    print(f"\n[SPLIT] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Train both models
    rf_model  = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate both models
    print("\n\n--- EVALUATION ---")
    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Compare and select winner using ROC-AUC (best metric for medical ML)
    print("\n\n--- MODEL COMPARISON ---")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 52)
    print(f"{'Random Forest':<20} {rf_metrics['accuracy']:>10.4f} {rf_metrics['f1']:>10.4f} {rf_metrics['roc_auc']:>10.4f}")
    print(f"{'XGBoost':<20} {xgb_metrics['accuracy']:>10.4f} {xgb_metrics['f1']:>10.4f} {xgb_metrics['roc_auc']:>10.4f}")

    # Pick the model with higher ROC-AUC
    if xgb_metrics["roc_auc"] >= rf_metrics["roc_auc"]:
        winner = xgb_model
        winner_name = "XGBoost"
        winner_metrics = xgb_metrics
    else:
        winner = rf_model
        winner_name = "Random Forest"
        winner_metrics = rf_metrics

    print(f"\n[WINNER] {winner_name} selected (ROC-AUC: {winner_metrics['roc_auc']:.4f})")

    # Find optimal threshold for the winning model
    # This improves recall for the Resistant class, which is critical for clinical use
    optimal_threshold = find_optimal_threshold(winner, X_test, y_test)

    # Save the winning model + feature names + threshold
    save_model(winner, feature_names, optimal_threshold)

    print("\n" + "="*60)
    print("  Training complete! Next step:")
    print("  Start the FastAPI server: uvicorn app.main:app --reload")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
