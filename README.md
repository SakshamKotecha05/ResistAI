# ResistAI: Antibiotic Resistance Prediction & Clinical Decision Support

> An AI-powered tool that predicts Ciprofloxacin resistance in bacterial isolates, explains the biological reasoning behind each prediction, and recommends alternative antibiotics in plain clinical language.

**Live demo:** https://resistai-2026.streamlit.app/

## The Problem

Antimicrobial resistance (AMR) is one of the most urgent threats in global healthcare. Bacteria evolve resistance faster than new antibiotics are developed, and clinicians must choose antibiotics under time pressure with incomplete resistance data.

Existing tools predict resistance, but they don't explain *why*, and they don't tell the clinician *what to use instead*. ResistAI bridges that gap.

## Innovation Angle

Most resistance prediction tools stop at a binary label: **Resistant** or **Susceptible**.

ResistAI adds an **LLM-powered clinical decision layer** on top of the ML model:

1. A **Random Forest / XGBoost classifier** (auto-selected by ROC-AUC) predicts Ciprofloxacin resistance from a bacterial isolate's antibiotic profile and patient demographics
2. **SHAP explainability** identifies which features drove the prediction: paired with a hand-built biology bridge that translates raw feature names into clinical English with one-sentence biological explanations
3. **Groq's Llama 3.3 70B** takes the SHAP-grounded evidence and generates a clinical report: plain-language summary, ranked alternative antibiotics, and a risk flag ready for a patient's medical chart

This is the pipeline clinicians actually need: not a black box, but a reasoning tool.

## Pipeline

```
Clinician Input (resistance profile + patient demographics)
        │
        ▼
  streamlit_app.py  (single-process, no API server)
        │
        ├──► app/predict.py
        │         ├── Random Forest / XGBoost Classifier
        │         │     └── predict_proba() → Resistant / Susceptible + confidence
        │         └── SHAP TreeExplainer
        │               └── Top 3 features driving the prediction
        │
        ├──► app/utils.py  (SHAP-to-biology bridge)
        │         └── Maps feature names → clinical labels + biological explanations
        │
        └──► app/llm.py
                  └── Groq API (Llama 3.3 70B)
                        └── Clinical Summary + Alternative Antibiotics + Risk Flag
                                          │
                                          ▼
                              Streamlit UI (clinician-facing)
```

## Setup

**Prerequisites:** Python 3.10+, a [Groq API key](https://console.groq.com) (free tier)

**1. Clone the repository**

```bash
git clone https://github.com/SakshamKotecha05/ResistAI
cd ResistAI
```

**2. Create and activate a virtual environment**

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

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

**5. Train the model**

```bash
cd ml
python train.py
```

This reads the AMR datasets from `data/`, trains and compares Random Forest vs XGBoost, tunes the classification threshold, and saves `ml/model.pkl`, `ml/feature_names.pkl`, and `ml/threshold.pkl`.

## Running the App

From the project root (one terminal, one command):

```bash
streamlit run streamlit_app.py
```

UI opens at `http://localhost:8501`

> On Streamlit Community Cloud, deploy `streamlit_app.py` as the entry point and set `GROQ_API_KEY` in the Cloud secrets dashboard.

## Usage

1. Open the app at `http://localhost:8501`
2. In the sidebar, check every antibiotic the bacterial strain is **resistant to**
3. Fill in patient demographics (age, sex, comorbidities, prior hospitalization)
4. Select the bacterial species and sample collection location
5. Click **Analyze Resistance**

The results panel shows:

- **Verdict card**: RESISTANT or SUSCEPTIBLE with confidence score and animated SVG gauge
- **SHAP feature chart**: which factors drove the prediction and what they mean biologically
- **Clinical Decision Support**: LLM-generated summary, alternative antibiotic options with drug class reasoning, and a chart-ready risk flag

Use **Load Example Patient** to instantly populate a demo case (E. coli, IFE-C, female/45/diabetic) that reliably produces a Resistant prediction.

## Screenshots

**Dashboard: empty state (no isolate loaded)**
![Dashboard empty state](assets/screenshots/01-dashboard-empty.png)

**Resistant prediction with SHAP feature analysis and clinical decision support**
![Resistant prediction result](assets/screenshots/02-resistant-result.png)

**Susceptible prediction**
![Susceptible prediction result](assets/screenshots/03-susceptible-result.png)

## Dataset Sources

| Dataset                          | Source                                     |
| -------------------------------- | ------------------------------------------ |
| Antimicrobial Resistance Dataset | [Mendeley Data](https://data.mendeley.com) |
| Multi-Drug Resistance Profiles   | [Kaggle](https://www.kaggle.com)           |

Datasets are not committed to this repository. Download them and place CSV files in the `data/` directory before running `ml/train.py`.

## Tech Stack

| Layer           | Technology              | Role                                                       |
| --------------- | ----------------------- | ---------------------------------------------------------- |
| Frontend + Host | Streamlit               | Clinical UI: pure Python, deployed on Streamlit Cloud      |
| ML Model        | sklearn + XGBoost       | Resistance classification (auto-selects best by ROC-AUC)   |
| Explainability  | SHAP (TreeExplainer)    | Feature importance with hand-built biological context      |
| LLM Layer       | Groq API: Llama 3.3 70B | Clinical recommendation generation from SHAP evidence      |
| Visualizations  | Plotly                  | Interactive SHAP bar chart                                 |
| Data Processing | pandas, numpy           | Preprocessing and feature engineering                      |

## Project Structure

```
ResistAI/
├── streamlit_app.py     # App entry point: imports app modules directly (no API server)
├── app/
│   ├── predict.py       # ML inference + SHAP explanation engine
│   ├── llm.py           # Groq clinical decision layer (prompt builder + fallback)
│   └── utils.py         # SHAP-to-biology bridge (clinical labels + biological context)
├── ml/
│   ├── train.py         # Trains RF + XGBoost, tunes threshold, saves best model
│   ├── preprocess.py    # Data cleaning and feature engineering
│   └── *.pkl            # Saved model artifacts (model, feature_names, threshold)
├── data/                # AMR datasets (gitignored: download separately)
├── .streamlit/
│   └── config.toml      # Streamlit theme config
├── .env                 # API keys (gitignored: never committed)
├── requirements.txt
└── README.md
```

*Built for CodeCure @ IIT BHU SPIRIT 2026: Track B: Antibiotic Resistance Prediction*
