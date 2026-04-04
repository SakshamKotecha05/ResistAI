# ResistAI: Antibiotic Resistance Prediction & Clinical Decision Support

> Predicts Ciprofloxacin resistance in bacterial isolates, explains the biological reasoning via SHAP, and generates plain-language clinical recommendations using an LLM: giving clinicians a tool they can actually act on.

**Live demo:** https://resistai-2026.streamlit.app/

---

## The Problem

Antimicrobial resistance (AMR) kills 1.27 million people annually and is projected to surpass cancer as the leading cause of death by 2050. Clinicians must choose antibiotics under time pressure with incomplete resistance data, often defaulting to broad-spectrum drugs that accelerate resistance further.

Existing prediction tools output a binary label: Resistant or Susceptible, and stop there. They don't explain *why*, and they don't tell the clinician *what to prescribe instead*.

---

## The Innovation

Most teams build a resistance classifier. ResistAI adds an **LLM-powered clinical decision-support layer** on top:

| Stage | What happens | Why it matters |
|---|---|---|
| **ML Classifier** | XGBoost predicts Ciprofloxacin resistance from antibiotic profile + demographics | Fast, interpretable classification |
| **SHAP Explainability** | Top 3 features driving the prediction, each paired with a biological explanation | Clinicians trust what they can understand |
| **Groq LLM (Llama 3.3 70B)** | Generates: clinical summary, ranked alternative antibiotics, chart-ready risk flag | Bridges ML output to clinical language |

This is the pipeline clinicians actually need: not a black box, but a reasoning tool.

---

## Model Performance

| Metric | Value |
|---|---|
| **Recall (clinical sensitivity)** | **83.0%**: catches 292 of 352 resistant cases in held-out test data |
| ROC-AUC | 0.693 |
| CV Recall (5-fold mean) | 62.7% ± 2.3% |
| Classification threshold | 0.35 (tuned to maximize recall, precision floor ≥ 20%) |

**Why recall over accuracy?** A false negative: missing a resistant case, means the patient receives an ineffective antibiotic. That causes treatment failure, longer hospital stays, and resistance spread. A false positive just triggers a broader-spectrum alternative: far less harmful. We optimize for sensitivity.

---

## Setup

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com)

**1. Clone the repo**
```bash
git clone https://github.com/SakshamKotecha05/ResistAI
cd ResistAI
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

**5. Run the app**
```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Click **Load Example Patient** to see a demo instantly.

> **Model artifacts** (`ml/model.pkl`, `ml/threshold.pkl`, etc.) are committed: no retraining required to run the app. To retrain from scratch: `cd ml && python train.py`.

---

## Usage

1. In the sidebar, check every antibiotic the bacterial strain is **resistant to**
2. Fill in patient demographics (age, sex, comorbidities, prior hospitalization)
3. Select bacterial species and sample location (optional)
4. Click **Analyze Resistance**

The results panel shows:
- **Verdict card**: RESISTANT / SUSCEPTIBLE with confidence gauge
- **SHAP feature analysis**: which factors drove the prediction with biological context
- **Clinical Decision Support**: LLM-generated summary, alternative antibiotics, and a chart-ready risk flag

Use **Load Example Patient** to instantly load a confirmed-resistant *E. coli* isolate (87% confidence) for a live demo.

---

## Screenshots

**Resistant prediction with SHAP analysis and clinical recommendation**
![Resistant prediction result](assets/screenshots/02-resistant-result.png)

**Dashboard: empty state**
![Dashboard empty state](assets/screenshots/01-dashboard-empty.png)

---

## Dataset Sources

| Dataset | Source | Contents |
|---|---|---|
| Antimicrobial Resistance Dataset | [Mendeley Data](https://data.mendeley.com/datasets/w6pjrymhtv/2) | 274 bacterial isolates, antibiotic susceptibility outcomes (R/S/I), location metadata |
| Multi-Drug Resistance Profiles | [Kaggle](https://www.kaggle.com/datasets/resumeforsachin/amr-antibiotics-prescription) | 10,710 patient records with full resistance profiles, demographics, species |

Datasets are not committed to this repo. Download them and place CSV files in `data/` before retraining.

---

## Pipeline

```
Clinician Input (resistance profile + demographics)
        |
        v
  streamlit_app.py  (single process, no API server needed)
        |
        |---> app/predict.py
        |         |-- XGBoost classifier: Resistant / Susceptible + confidence
        |         `-- SHAP TreeExplainer: top 3 driving features
        |
        |---> app/utils.py
        |         `-- SHAP-to-biology bridge: clinical labels + one-line explanations
        |
        `---> app/llm.py
                  `-- Groq Llama 3.3 70B
                        `-- Clinical Summary · Alternative Antibiotics · Risk Flag
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | Clinical UI: pure Python, zero JS |
| ML Model | XGBoost + scikit-learn | Resistance classifier (auto-selected by CV recall) |
| Explainability | SHAP TreeExplainer | Feature importance with biological context |
| LLM Layer | Groq API: Llama 3.3 70B | Clinical language generation from SHAP evidence |
| Visualizations | Plotly | Interactive SHAP bar chart |
| Data Processing | pandas, numpy | Preprocessing and feature engineering |

---

## Project Structure

```
ResistAI/
├── streamlit_app.py       # App entry point
├── app/
│   ├── predict.py         # ML inference + SHAP
│   ├── llm.py             # Groq clinical decision layer
│   └── utils.py           # SHAP-to-biology bridge
├── ml/
│   ├── train.py           # Training pipeline
│   ├── preprocess.py      # Feature engineering
│   ├── model.pkl          # Trained XGBoost model
│   ├── threshold.pkl      # Tuned classification threshold (0.35)
│   ├── feature_names.pkl  # Feature column order for inference
│   └── metrics.pkl        # Evaluation metrics for UI display
├── data/                  # AMR datasets (gitignored)
├── .streamlit/
│   └── config.toml        # Theme config
├── requirements.txt
└── README.md
```

---

*Built for **CodeCure @ IIT BHU SPIRIT 2026**: Track B: Antibiotic Resistance Prediction*
