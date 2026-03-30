"""
ml/preprocess.py — Combined preprocessing pipeline for both AMR datasets

DATASET 1 — Primary (Mendeley AMR, 274 rows):
    Columns: Location, IMIPENEM, CEFTAZIDIME, GENTAMICIN, AUGMENTIN, CIPROFLOXACIN
    Values:  Numeric zone diameters (mm) from disk diffusion tests
    Action:  Convert to R/S/I using CLSI breakpoints, then encode to 0/1

DATASET 2 — Secondary (Kaggle Multi-Resistance, 10,710 rows):
    Columns: patient demographics, 14 antibiotic columns with R/S/I labels
    Action:  Normalize labels, extract features, encode to 0/1

COMBINATION STRATEGY:
    Both datasets share a common feature space after encoding.
    Dataset-specific features (e.g., demographics only in secondary) are set to 0
    for rows from the other dataset. Both contribute to the same target:
    Ciprofloxacin resistance (0 = Susceptible, 1 = Resistant).

Run directly to explore and verify preprocessing:
    python ml/preprocess.py
"""

import pandas as pd
import numpy as np
import re
import os

# ── File Paths ───────────────────────────────────────────────────────────────

PRIMARY_DATA_PATH   = "data/amr_data.csv"
SECONDARY_DATA_PATH = "data/amr_data_secondary.csv"

# ── CLSI Disk Diffusion Breakpoints (mm) ────────────────────────────────────
# Source: CLSI M100 standard for Enterobacteriaceae (general use)
# Format: { column_name: (susceptible_cutoff, resistant_cutoff) }
# Zone >= susceptible_cutoff → S (Susceptible)
# Zone <= resistant_cutoff   → R (Resistant)
# Zone between               → I (Intermediate) → treated as Resistant

CLSI_BREAKPOINTS = {
    "CIPROFLOXACIN": (21, 15),  # ≥21mm=S, 16-20mm=I, ≤15mm=R
    "IMIPENEM":      (23, 19),  # ≥23mm=S, 20-22mm=I, ≤19mm=R
    "CEFTAZIDIME":   (21, 17),  # ≥21mm=S, 18-20mm=I, ≤17mm=R
    "GENTAMICIN":    (15, 12),  # ≥15mm=S, 13-14mm=I, ≤12mm=R
    "AUGMENTIN":     (18, 13),  # ≥18mm=S, 14-17mm=I, ≤13mm=R
}

# ── Resistance Label Normalization (for secondary dataset) ───────────────────
# Maps all label variants → standard R / S / I
RESISTANCE_NORMALIZE = {
    "r": "R", "resistant": "R",   "Resistant": "R",
    "s": "S", "susceptible": "S", "Susceptible": "S",
    "i": "I", "intermediate": "I","Intermediate": "I",
}

# Labels that mean "no usable data" — rows with these in the TARGET column are dropped
INVALID_LABELS = {"?", "missing", "error", "unknown", "nan", ""}

# ── Secondary Dataset Column Mappings ────────────────────────────────────────
# Maps secondary dataset antibiotic column names → standardized feature names
# We align these with primary dataset antibiotic names where possible
SECONDARY_ANTIBIOTIC_MAP = {
    "AMX/AMP":        "amoxicillin_resistance",   # Amoxicillin/Ampicillin
    "AMC":            "augmentin_resistance",      # Amoxicillin-Clavulanate (≈ AUGMENTIN)
    "CZ":             "cefazolin_resistance",      # Cefazolin (first-gen cephalosporin)
    "FOX":            "cefoxitin_resistance",      # Cefoxitin (second-gen cephalosporin)
    "CTX/CRO":        "ceftazidime_resistance",    # Cefotaxime/Ceftriaxone ≈ CEFTAZIDIME class
    "IPM":            "imipenem_resistance",       # Imipenem (≈ IMIPENEM in primary)
    "GEN":            "gentamicin_resistance",     # Gentamicin (≈ GENTAMICIN in primary)
    "AN":             "amikacin_resistance",       # Amikacin (aminoglycoside)
    "ofx":            "ofloxacin_resistance",      # Ofloxacin (fluoroquinolone — same class as CIP)
    "C":              "chloramphenicol_resistance",# Chloramphenicol
    "Co-trimoxazole": "cotrimoxazole_resistance",  # Trimethoprim-Sulfamethoxazole
    "Furanes":        "nitrofurantoin_resistance", # Nitrofurantoin
    "colistine":      "colistin_resistance",       # Colistin (last-resort drug)
}


# ════════════════════════════════════════════════════════════════════════════
# PRIMARY DATASET PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def apply_clsi_breakpoint(value, susceptible_cutoff: int, resistant_cutoff: int) -> str:
    """
    Converts a numeric zone diameter (mm) to R/S/I using CLSI breakpoints.

    Disk diffusion explanation:
    An antibiotic-soaked disk is placed on a bacterial culture plate.
    The antibiotic diffuses outward and kills bacteria near the disk,
    creating a clear 'zone of inhibition'. Larger zone = bacteria more susceptible.
    CLSI publishes standard cutoffs for interpreting zone sizes.
    """
    try:
        zone = float(value)
        if zone >= susceptible_cutoff:
            return "S"
        elif zone <= resistant_cutoff:
            return "R"
        else:
            return "I"  # Intermediate — treated as Resistant (clinically unreliable)
    except (ValueError, TypeError):
        return "INVALID"


def process_primary(path: str = PRIMARY_DATA_PATH) -> pd.DataFrame:
    """
    Loads and processes the primary Mendeley dataset.

    Steps:
    1. Load CSV
    2. Convert numeric zone diameters to R/S/I using CLSI breakpoints
    3. Convert CIPROFLOXACIN → binary target (R/I=1, S=0)
    4. Convert other antibiotics → binary features (R/I=1, S=0)
    5. One-hot encode Location
    6. Return standardized DataFrame with common column names

    Returns: DataFrame with columns ready for combination with secondary data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Primary dataset not found at '{path}'. See data/README.md")

    df = pd.read_csv(path)
    print(f"[PRIMARY] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[PRIMARY] Columns: {list(df.columns)}")

    # ── Convert zone diameters to R/S/I ──
    for col, (s_cut, r_cut) in CLSI_BREAKPOINTS.items():
        if col in df.columns:
            df[col + "_label"] = df[col].apply(
                lambda x: apply_clsi_breakpoint(x, s_cut, r_cut)
            )
            counts = df[col + "_label"].value_counts().to_dict()
            print(f"[PRIMARY] {col}: {counts}")

    # ── Build target: CIPROFLOXACIN resistance ──
    cip_col = "CIPROFLOXACIN_label"
    df_valid = df[df[cip_col].isin(["R", "S", "I"])].copy()
    dropped = len(df) - len(df_valid)
    if dropped > 0:
        print(f"[PRIMARY] Dropped {dropped} rows with invalid CIP values")

    y_primary = df_valid[cip_col].map({"R": 1, "I": 1, "S": 0}).astype(int)

    # ── Build feature columns ──
    result = pd.DataFrame(index=df_valid.index)

    # Antibiotic features (shared with secondary)
    antibiotic_mapping = {
        "IMIPENEM_label":   "imipenem_resistance",
        "CEFTAZIDIME_label":"ceftazidime_resistance",
        "GENTAMICIN_label": "gentamicin_resistance",
        "AUGMENTIN_label":  "augmentin_resistance",
    }
    for src_col, dest_col in antibiotic_mapping.items():
        if src_col in df_valid.columns:
            result[dest_col] = df_valid[src_col].map({"R": 1, "I": 1, "S": 0}).fillna(0).astype(int)
        else:
            result[dest_col] = 0

    # Location encoding (primary-only feature — will be 0 for secondary rows)
    location_dummies = pd.get_dummies(df_valid["Location"], prefix="location")
    for col in location_dummies.columns:
        result[col] = location_dummies[col]

    # Demographics / clinical (not in primary — set to 0, secondary will have real values)
    for col in ["age_years", "is_female", "has_diabetes", "has_hypertension",
                "prior_hospitalization", "infection_freq"]:
        result[col] = 0

    # Source flag: mark which dataset this row came from (useful for analysis)
    result["dataset_source"] = 0  # 0 = primary

    result["target"] = y_primary.values

    # Drop rows where any core antibiotic feature conversion failed
    result = result.dropna()

    print(f"[PRIMARY] Final shape: {result.shape}")
    print(f"[PRIMARY] CIP Resistant: {result['target'].sum()} / {len(result)} "
          f"({result['target'].mean()*100:.1f}%)")
    return result


# ════════════════════════════════════════════════════════════════════════════
# SECONDARY DATASET PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def normalize_label(value) -> str:
    """Normalizes any resistance label variant to R / S / I / INVALID."""
    val = str(value).strip()
    normalized = RESISTANCE_NORMALIZE.get(val, val)
    if normalized.upper() in ("R", "S", "I"):
        return normalized.upper()
    if val.lower() in INVALID_LABELS:
        return "INVALID"
    return "INVALID"


def parse_age_gender(val) -> tuple:
    """
    Parses '37/F' → (37, 1)  where 1=female, 0=male
    Returns (NaN, NaN) on failure — filled with median/mode later.
    """
    try:
        parts = str(val).strip().split("/")
        age = float(parts[0])
        gender = 1.0 if parts[1].strip().upper() == "F" else 0.0
        return age, gender
    except Exception:
        return np.nan, np.nan


def extract_species_name(val) -> str:
    """
    Extracts species from 'S3413 Escherichia coli' → 'Escherichia coli'
    Removes the leading sample ID number (S + digits).
    """
    val = str(val).strip()
    cleaned = re.sub(r'^S\d+\s*', '', val).strip()
    if not cleaned or cleaned.lower() in INVALID_LABELS:
        return "Unknown"
    return cleaned


def process_secondary(path: str = SECONDARY_DATA_PATH) -> pd.DataFrame:
    """
    Loads and processes the secondary Kaggle dataset.

    Steps:
    1. Load CSV
    2. Build target from CIP column (R/I → 1, S → 0), drop invalid rows
    3. Parse age/gender into two numeric columns
    4. Extract species name from Souches, one-hot encode top species
    5. Encode clinical features (Diabetes, Hypertension, Hospital_before, Infection_Freq)
    6. Encode all antibiotic columns to binary resistance flags
    7. Return standardized DataFrame with common column names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Secondary dataset not found at '{path}'. See data/README.md")

    df = pd.read_csv(path, low_memory=False)
    print(f"\n[SECONDARY] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Build target: CIP (Ciprofloxacin) resistance ──
    df["_cip_norm"] = df["CIP"].apply(normalize_label)
    df_valid = df[df["_cip_norm"].isin(["R", "S", "I"])].copy()
    dropped = len(df) - len(df_valid)
    print(f"[SECONDARY] Dropped {dropped} rows with invalid CIP labels")

    y_secondary = df_valid["_cip_norm"].map({"R": 1, "I": 1, "S": 0}).astype(int)

    result = pd.DataFrame(index=df_valid.index)

    # ── Age and gender ──
    parsed = df_valid["age/gender"].apply(parse_age_gender)
    result["age_years"] = parsed.apply(lambda x: x[0])
    result["is_female"]  = parsed.apply(lambda x: x[1])
    result["age_years"]  = result["age_years"].fillna(result["age_years"].median())
    result["is_female"]  = result["is_female"].fillna(result["is_female"].mode()[0])
    print(f"[SECONDARY] Age range: {result['age_years'].min():.0f}–{result['age_years'].max():.0f} "
          f"| Median: {result['age_years'].median():.0f}")

    # ── Clinical features ──
    result["has_diabetes"] = df_valid["Diabetes"].apply(
        lambda x: 1 if str(x).strip().lower() in ("true", "yes", "1") else 0
    )
    result["has_hypertension"] = df_valid["Hypertension"].apply(
        lambda x: 1 if str(x).strip().lower() in ("true", "yes", "1") else 0
    )
    # Prior hospitalization is a strong AMR risk factor — hospital strains carry more resistance
    result["prior_hospitalization"] = df_valid["Hospital_before"].apply(
        lambda x: 1 if str(x).strip().lower() in ("true", "yes", "1") else 0
    )
    # Infection frequency: how many prior infections (more = more antibiotic exposure = more resistance)
    def parse_freq(val):
        try:
            return float(val)
        except Exception:
            return np.nan
    result["infection_freq"] = df_valid["Infection_Freq"].apply(parse_freq)
    result["infection_freq"] = result["infection_freq"].fillna(result["infection_freq"].median())

    # ── Antibiotic resistance features ──
    for src_col, dest_col in SECONDARY_ANTIBIOTIC_MAP.items():
        if src_col not in df_valid.columns:
            result[dest_col] = 0
            continue
        normalized = df_valid[src_col].apply(normalize_label)
        mapped = normalized.map({"R": 1, "I": 1, "S": 0})
        mode_val = mapped.mode()[0] if not mapped.mode().empty else 0
        result[dest_col] = mapped.fillna(mode_val).astype(int)

    # ── Species (top 8 + Other) ──
    df_valid["_species"] = df_valid["Souches"].apply(extract_species_name)
    top_species = df_valid["_species"].value_counts().head(8).index.tolist()
    df_valid["_species_clean"] = df_valid["_species"].apply(
        lambda x: x if x in top_species else "Other"
    )
    species_dummies = pd.get_dummies(df_valid["_species_clean"], prefix="species")
    for col in species_dummies.columns:
        result[col] = species_dummies[col].values
    print(f"[SECONDARY] Species encoded: {list(species_dummies.columns)}")

    # Location columns (primary-only — set to 0 for secondary rows)
    # These will be filled with zeros when combined
    result["dataset_source"] = 1  # 1 = secondary
    result["target"] = y_secondary.values

    result = result.fillna(0)
    print(f"[SECONDARY] Final shape: {result.shape}")
    print(f"[SECONDARY] CIP Resistant: {result['target'].sum()} / {len(result)} "
          f"({result['target'].mean()*100:.1f}%)")
    return result


# ════════════════════════════════════════════════════════════════════════════
# COMBINE BOTH DATASETS
# ════════════════════════════════════════════════════════════════════════════

def get_processed_data():
    """
    Master pipeline: processes both datasets and returns a single combined
    (X, y, feature_names) ready for model training.

    Combination logic:
    - Both DataFrames are aligned to the same column set using pd.concat(align=True)
    - Missing columns in either DataFrame are filled with 0 (unknown = no data)
    - The combined dataset has ~10,984 rows total

    Returns:
        X             — pd.DataFrame, numeric feature matrix
        y             — pd.Series, binary labels (0=Susceptible, 1=Resistant)
        feature_names — list of feature column names (used for SHAP labeling)
    """
    print("\n" + "="*60)
    print("  ResistAI — Combined Preprocessing Pipeline")
    print("  Primary (Mendeley) + Secondary (Kaggle)")
    print("="*60)

    df_primary   = process_primary()
    df_secondary = process_secondary()

    # Combine with outer join — missing columns filled with 0
    combined = pd.concat([df_primary, df_secondary], axis=0, join="outer").fillna(0)
    combined = combined.reset_index(drop=True)

    print(f"\n[COMBINED] Total rows: {len(combined)}")
    print(f"[COMBINED] Total columns: {len(combined.columns)}")

    # Separate features from target
    # Drop: 'target' and 'dataset_source' (not a real feature, just bookkeeping)
    feature_cols = [c for c in combined.columns if c not in ("target", "dataset_source")]

    X = combined[feature_cols].copy()
    X = X.apply(pd.to_numeric)
    y = combined["target"].astype(int)

    # Final class distribution
    total = len(y)
    n_resistant = y.sum()
    imbalance_ratio = (total - n_resistant) / n_resistant if n_resistant > 0 else 1.0

    print(f"\n[COMBINED] Class distribution:")
    print(f"  Susceptible (0): {total - n_resistant} ({(total-n_resistant)/total*100:.1f}%)")
    print(f"  Resistant   (1): {n_resistant} ({n_resistant/total*100:.1f}%)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x  "
          f"{'(will apply class_weight in training)' if imbalance_ratio > 1.5 else '(balanced)'}")

    feature_names = list(X.columns)

    print(f"\n[DONE] Features ({len(feature_names)}):")
    for i, f in enumerate(feature_names):
        print(f"  {i:2d}. {f}")

    print(f"\n{'='*60}")
    print(f"  Ready for training. Run: python ml/train.py")
    print(f"{'='*60}\n")

    return X, y, feature_names


# ── Run directly to verify preprocessing output ──────────────────────────────

if __name__ == "__main__":
    X, y, features = get_processed_data()
    print(f"\nSample rows from X:")
    print(X.head(3).to_string())
