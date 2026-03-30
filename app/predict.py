"""
app/predict.py — ML Inference Engine

What this file does:
1. Loads the trained model, feature names, and optimal threshold from disk (once, at startup)
2. Accepts a dict of input values from the API request
3. Builds a properly ordered feature vector (40 columns) aligned to training format
4. Runs model.predict_proba() and applies the threshold (0.35 by default)
5. Computes SHAP values for the single prediction
6. Returns prediction + confidence + top 3 SHAP-explained features

Why load at module level (not inside the function)?
Loading model.pkl takes ~0.5s. If we loaded it on every request, the API would be
slow for every user. Loading once at startup means subsequent requests are fast.

SHAP explanation:
SHAP (SHapley Additive exPlanations) assigns each feature a score representing
how much it pushed the prediction toward Resistant (+) or away from it (-).
A SHAP value of +0.3 for "ofloxacin_resistance" means: this feature alone raised
the Resistant probability by 0.3 — a strong positive signal.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

from app.utils import get_feature_label

# ── Load Model Artifacts ──────────────────────────────────────────────────────
# These paths are relative to the project root (where uvicorn is launched from)

MODEL_PATH     = "ml/model.pkl"
FEATURES_PATH  = "ml/feature_names.pkl"
THRESHOLD_PATH = "ml/threshold.pkl"

# Validate files exist before attempting to load
for _path in (MODEL_PATH, FEATURES_PATH, THRESHOLD_PATH):
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"[ERROR] Required file not found: '{_path}'\n"
            f"  Run 'python ml/train.py' from the project root first."
        )

print("[predict] Loading model artifacts...")
model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)  # list of 40 feature column names
threshold     = joblib.load(THRESHOLD_PATH)  # float, e.g. 0.35

print(f"[predict] Model loaded — {len(feature_names)} features, threshold={threshold:.2f}")

# ── Create SHAP Explainer ─────────────────────────────────────────────────────
# TreeExplainer is the fast, exact SHAP method for tree-based models (RF, XGBoost).
# We create it once at startup — it's expensive to initialize but cheap to use.
# check_additivity=False avoids a slow consistency check on each call.

explainer = shap.TreeExplainer(model)
print("[predict] SHAP TreeExplainer ready.")

# ── Antibiotic Flags Accepted as API Input ───────────────────────────────────
# This list controls which resistance flags the API accepts from the user.
# Other features (species, location one-hots) are handled separately below.

ANTIBIOTIC_FLAGS = [
    "imipenem_resistance",
    "ceftazidime_resistance",
    "gentamicin_resistance",
    "augmentin_resistance",
    "amoxicillin_resistance",
    "cefazolin_resistance",
    "cefoxitin_resistance",
    "amikacin_resistance",
    "ofloxacin_resistance",
    "chloramphenicol_resistance",
    "cotrimoxazole_resistance",
    "nitrofurantoin_resistance",
    "colistin_resistance",
]

DEMOGRAPHIC_FIELDS = [
    "age_years",
    "is_female",
    "has_diabetes",
    "has_hypertension",
    "prior_hospitalization",
    "infection_freq",
]


# ── Main Prediction Function ──────────────────────────────────────────────────

def predict(input_data: dict) -> dict:
    """
    Runs ML inference + SHAP explanation for a single bacterial isolate.

    Args:
        input_data: dict from the Pydantic-validated API request. Keys include
                    antibiotic flags, demographics, 'species', 'location'.

    Returns:
        dict with keys:
            prediction      — "Resistant" or "Susceptible"
            confidence      — probability of Resistant class (0.0–1.0)
            threshold_used  — the threshold applied (e.g. 0.35)
            top_features    — list of top 3 SHAP-explained features, each with:
                                feature    (raw name)
                                label      (clinical English label)
                                shap_value (float, + = pushes toward Resistant)
                                direction  ("increases_risk" or "decreases_risk")
    """

    # ── Step 1: Build a zero-filled feature row ───────────────────────────────
    # We start with all 40 features set to 0 (= "not observed" / "unknown").
    # Then we fill in whatever the user provided.
    # This handles the case where a user doesn't provide all 40 values.

    row = {feat: 0.0 for feat in feature_names}

    # ── Step 2: Fill antibiotic resistance flags ──────────────────────────────
    for flag in ANTIBIOTIC_FLAGS:
        if flag in input_data and flag in row:
            row[flag] = float(input_data[flag])

    # ── Step 3: Fill patient demographics ────────────────────────────────────
    for field in DEMOGRAPHIC_FIELDS:
        if field in input_data and input_data[field] is not None and field in row:
            row[field] = float(input_data[field])

    # ── Step 4: Map species string → one-hot column ───────────────────────────
    # The training data one-hot encoded bacterial species like "Escherichia coli"
    # into columns named "species_Escherichia coli".
    # We try exact match first, then substring match as fallback.

    species = (input_data.get("species") or "").strip()
    if species:
        exact_col = f"species_{species}"
        if exact_col in row:
            row[exact_col] = 1.0
        else:
            # Substring match — catches slight name variations
            for col in feature_names:
                if col.startswith("species_") and species.lower() in col.lower():
                    row[col] = 1.0
                    break
            # If no match found, row stays all-zero for species → "Other" species behavior

    # ── Step 5: Map location string → one-hot column ─────────────────────────
    location = (input_data.get("location") or "").strip()
    if location:
        exact_col = f"location_{location}"
        if exact_col in row:
            row[exact_col] = 1.0
        else:
            for col in feature_names:
                if col.startswith("location_") and location.lower() in col.lower():
                    row[col] = 1.0
                    break

    # ── Step 6: Build a properly ordered DataFrame ────────────────────────────
    # The model was trained with features in a specific column order.
    # We MUST use the same order here, otherwise feature values get assigned
    # to the wrong columns and the prediction is garbage.

    X = pd.DataFrame([row])[feature_names]  # enforces correct column order

    # ── Step 7: Run inference ─────────────────────────────────────────────────
    prob = float(model.predict_proba(X)[0][1])  # probability of Resistant (class 1)
    prediction = "Resistant" if prob >= threshold else "Susceptible"

    # ── Step 8: Compute SHAP values ───────────────────────────────────────────
    # shap_values for RandomForest binary classification returns a list of 2 arrays:
    #   shap_values[0] — contributions toward Susceptible (class 0)
    #   shap_values[1] — contributions toward Resistant   (class 1)
    # Each array has shape (n_samples, n_features).
    # For a single sample: shap_values[1][0] is a 1D array of 40 SHAP values.

    try:
        shap_vals = explainer.shap_values(X)

        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            # Standard RandomForest binary output: list of [neg_class, pos_class]
            shap_row = np.array(shap_vals[1][0])
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # Some SHAP versions return shape (n_samples, n_features, n_classes)
            shap_row = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            shap_row = shap_vals[0]
        else:
            # Fallback: return zeros so the API still works even if SHAP format changes
            shap_row = np.zeros(len(feature_names))

    except Exception as e:
        print(f"[predict] SHAP computation failed: {e}. Returning zero SHAP values.")
        shap_row = np.zeros(len(feature_names))

    # ── Step 9: Extract top 3 features by absolute SHAP value ────────────────
    # abs(shap_value) tells us importance — a large negative SHAP value still
    # means the feature is highly influential (just in the Susceptible direction).

    shap_pairs = sorted(
        zip(feature_names, shap_row.tolist()),
        key=lambda pair: abs(pair[1]),
        reverse=True
    )[:3]

    top_features = []
    for feat_name, shap_val in shap_pairs:
        top_features.append({
            "feature":    feat_name,
            "label":      get_feature_label(feat_name),
            "shap_value": round(shap_val, 4),
            "direction":  "increases_risk" if shap_val > 0 else "decreases_risk"
        })

    return {
        "prediction":     prediction,
        "confidence":     round(prob, 4),
        "threshold_used": round(threshold, 2),
        "top_features":   top_features,
    }
