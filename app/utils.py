"""
app/utils.py — SHAP-to-Biology Bridge (Differentiation Strategy #2)

What this file does:
Maps raw ML feature names (like "imipenem_resistance") to:
  1. A human-readable clinical label   — shown on the SHAP chart
  2. A one-line biological explanation — fed into the Groq LLM prompt

Why this matters:
SHAP tells us WHICH features drove the prediction. This file tells us WHAT THAT MEANS
biologically. Without this, the chart just says "imipenem_resistance = 0.31" which
means nothing to a clinician. With this, it says "Imipenem resistance — a key
carbapenem, resistance may indicate carbapenemase production."

This is the bridge between ML output and clinical meaning.
"""

# ── Feature Label Map ─────────────────────────────────────────────────────────
# Keys:   exact feature names as stored in ml/feature_names.pkl
# Values: dict with:
#   "label"    — short clinical name (shown on SHAP chart axis)
#   "biology"  — one sentence explaining WHY this feature matters biologically

FEATURE_LABELS = {

    # ── Antibiotic Resistance Features ────────────────────────────────────────

    "imipenem_resistance": {
        "label":   "Imipenem resistance",
        "biology": "Imipenem is a last-resort carbapenem antibiotic; resistance signals "
                   "carbapenemase enzyme production (KPC, NDM-1), a critical clinical emergency "
                   "indicating very limited treatment options."
    },
    "ceftazidime_resistance": {
        "label":   "Ceftazidime resistance",
        "biology": "Ceftazidime is a 3rd-generation cephalosporin; resistance often indicates "
                   "extended-spectrum beta-lactamase (ESBL) production, which can spread "
                   "horizontally between bacteria."
    },
    "gentamicin_resistance": {
        "label":   "Gentamicin resistance",
        "biology": "Gentamicin is an aminoglycoside antibiotic; resistance is linked to "
                   "aminoglycoside-modifying enzymes (AMEs) and suggests prior aminoglycoside "
                   "exposure or co-resistance with other drug classes."
    },
    "augmentin_resistance": {
        "label":   "Augmentin resistance",
        "biology": "Augmentin (amoxicillin-clavulanate) resistance indicates beta-lactamase "
                   "activity that overcomes the clavulanate inhibitor, often seen in "
                   "TEM or SHV enzyme variants."
    },
    "amoxicillin_resistance": {
        "label":   "Amoxicillin resistance",
        "biology": "Amoxicillin resistance is extremely common in gram-negative bacteria due "
                   "to plasmid-mediated beta-lactamases; it signals the strain has acquired "
                   "basic antibiotic resistance mechanisms."
    },
    "cefazolin_resistance": {
        "label":   "Cefazolin resistance",
        "biology": "Cefazolin is a 1st-generation cephalosporin; resistance suggests "
                   "beta-lactamase activity and predicts cross-resistance to other "
                   "early-generation beta-lactam antibiotics."
    },
    "cefoxitin_resistance": {
        "label":   "Cefoxitin resistance",
        "biology": "Cefoxitin is a cephamycin; resistance is a marker for AmpC "
                   "beta-lactamase production, which is not inhibited by standard "
                   "beta-lactamase inhibitors like clavulanate."
    },
    "amikacin_resistance": {
        "label":   "Amikacin resistance",
        "biology": "Amikacin is a reserve aminoglycoside used when gentamicin fails; "
                   "resistance here indicates high-level aminoglycoside resistance and "
                   "severely narrows aminoglycoside treatment options."
    },
    "ofloxacin_resistance": {
        "label":   "Ofloxacin resistance",
        "biology": "Ofloxacin is a fluoroquinolone — the same drug class as Ciprofloxacin. "
                   "Resistance strongly predicts Ciprofloxacin resistance via shared "
                   "QRDR mutations in DNA gyrase and topoisomerase IV."
    },
    "chloramphenicol_resistance": {
        "label":   "Chloramphenicol resistance",
        "biology": "Chloramphenicol resistance is mediated by chloramphenicol acetyltransferase "
                   "(CAT); its presence alongside fluoroquinolone resistance suggests "
                   "multi-drug resistant plasmid carriage."
    },
    "cotrimoxazole_resistance": {
        "label":   "Co-trimoxazole resistance",
        "biology": "Co-trimoxazole (TMP-SMX) resistance reflects DHFR/DHPS mutations; "
                   "this is one of the most common resistance traits in gram-negative UTI "
                   "pathogens and is often co-selected with fluoroquinolone resistance."
    },
    "nitrofurantoin_resistance": {
        "label":   "Nitrofurantoin resistance",
        "biology": "Nitrofurantoin resistance requires specific nitroreductase gene mutations "
                   "(nfsA/nfsB); its presence alongside fluoroquinolone resistance suggests "
                   "a broadly resistant urinary tract pathogen."
    },
    "colistin_resistance": {
        "label":   "Colistin resistance",
        "biology": "Colistin is a last-resort antibiotic for gram-negative superbugs; "
                   "resistance (often via mcr genes) signals extreme drug resistance and "
                   "is a global public health threat."
    },

    # ── Patient Demographics ─────────────────────────────────────────────────

    "age_years": {
        "label":   "Patient age (years)",
        "biology": "Older patients have often received more antibiotic courses over their "
                   "lifetime, increasing selective pressure on their commensal bacteria "
                   "and raising the probability of carrying resistant strains."
    },
    "is_female": {
        "label":   "Patient sex (female)",
        "biology": "Female sex is associated with higher UTI frequency, leading to more "
                   "antibiotic prescriptions and higher probability of selecting for "
                   "resistant Ciprofloxacin-resistant strains in urinary pathogens."
    },
    "has_diabetes": {
        "label":   "Diabetes diagnosis",
        "biology": "Diabetic patients have compromised innate immunity and receive more "
                   "antibiotics for recurrent infections, creating selection pressure "
                   "for antibiotic-resistant organisms."
    },
    "has_hypertension": {
        "label":   "Hypertension diagnosis",
        "biology": "Hypertension itself does not drive resistance, but it is a comorbidity "
                   "marker for patients with complex medical histories and higher "
                   "cumulative antibiotic exposure."
    },
    "prior_hospitalization": {
        "label":   "Prior hospitalization",
        "biology": "Hospital environments harbor multi-drug resistant organisms; prior "
                   "hospitalization is one of the strongest risk factors for acquiring "
                   "a resistant bacterial strain (healthcare-associated infection)."
    },
    "infection_freq": {
        "label":   "Prior infection frequency",
        "biology": "Patients with recurrent infections receive repeated antibiotic courses, "
                   "directly applying selection pressure that drives bacterial resistance "
                   "acquisition over time."
    },

    # ── Dataset Source (internal flag — low clinical weight) ──────────────────

    "dataset_source": {
        "label":   "Dataset origin",
        "biology": "Internal flag indicating primary (Mendeley) or secondary (Kaggle) dataset; "
                   "not a true clinical feature but captures dataset-level batch effects."
    },

    # ── Engineered Features ────────────────────────────────────────────────────

    "fluoro_cotrim_coresistance": {
        "label":   "Fluoro + Cotrim Co-resistance",
        "biology": "Ofloxacin and Co-trimoxazole resistance co-occurring on the same isolate "
                   "is the strongest plasmid-linked signal for Ciprofloxacin resistance — "
                   "both resistances are frequently encoded on the same integron in gram-negative UTI pathogens."
    },
    "esbl_pattern": {
        "label":   "ESBL resistance pattern",
        "biology": "Ceftazidime and Augmentin resistance together define the classic Extended-Spectrum "
                   "Beta-Lactamase (ESBL) phenotype. ESBL-producing strains frequently co-carry "
                   "fluoroquinolone resistance genes on the same resistance plasmid."
    },
    "extreme_resistance": {
        "label":   "Last-Resort Drug Resistance",
        "biology": "Resistance to Imipenem (carbapenem) or Colistin (polymyxin) signals an extensively "
                   "drug-resistant or pan-resistant profile — strains with last-resort resistance almost "
                   "universally carry resistance to all other antibiotic classes including fluoroquinolones."
    },
    "aminoglycoside_coresistance": {
        "label":   "Aminoglycoside Co-resistance",
        "biology": "Gentamicin and Amikacin both resistant indicates high-level aminoglycoside-modifying "
                   "enzyme activity. These enzymes are often encoded on mobile genetic elements that also "
                   "carry fluoroquinolone resistance determinants."
    },
    "total_resistance_count": {
        "label":   "MDR Score",
        "biology": "Total number of antibiotics this isolate is resistant to across all 13 tested classes. "
                   "A higher score indicates more resistance genes are present — each additional resistance "
                   "gene increases the probability that fluoroquinolone resistance genes are also carried."
    },
    "clinical_risk_score": {
        "label":   "Clinical Risk Score",
        "biology": "Composite of prior hospitalization, diabetes diagnosis, and recurrent infection history "
                   "(>2 prior infections). Risk factors compound: each additional factor multiplies antibiotic "
                   "exposure history and the probability of harboring a resistant strain."
    },
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_feature_label(feature_name: str) -> str:
    """
    Returns a human-readable clinical label for a given feature name.

    Handles three cases:
    1. Known feature in FEATURE_LABELS → return its label
    2. One-hot species column (starts with 'species_') → format nicely
    3. One-hot location column (starts with 'location_') → format nicely
    4. Unknown feature → title-case the raw name as fallback
    """
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]["label"]

    if feature_name.startswith("species_"):
        # "species_Escherichia coli" → "Bacterial species: Escherichia coli"
        species = feature_name[len("species_"):].replace("_", " ")
        return f"Bacterial species: {species}"

    if feature_name.startswith("location_"):
        # "location_ICU" → "Sample location: ICU"
        loc = feature_name[len("location_"):].replace("_", " ").title()
        return f"Sample location: {loc}"

    # Fallback: clean up underscores and title-case
    return feature_name.replace("_", " ").title()


def get_feature_biology(feature_name: str) -> str:
    """
    Returns a one-sentence biological explanation for a feature.
    Used in the Groq LLM prompt to give context beyond just the feature name.
    Returns empty string for unknown features (LLM will skip them gracefully).
    """
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]["biology"]

    if feature_name.startswith("species_"):
        species = feature_name[len("species_"):].replace("_", " ")
        return (f"{species} is the infecting bacterial species; different species carry "
                "different intrinsic resistance mechanisms and plasmid profiles.")

    if feature_name.startswith("location_"):
        loc = feature_name[len("location_"):].replace("_", " ")
        return (f"Sample collected from {loc}; geographic location reflects local "
                "antibiotic usage patterns and regional resistance prevalence.")

    return ""
