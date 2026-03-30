"""
app/main.py — FastAPI Application Entry Point

What this file does:
1. Creates the FastAPI application instance
2. Adds CORS middleware so Streamlit (port 8501) can call this API (port 8000)
3. Defines the Pydantic input/output models for request validation
4. Exposes two endpoints:
     GET  /health   — health check (Streamlit uses this to verify the API is up)
     POST /predict  — main prediction endpoint

How to run:
    uvicorn app.main:app --reload
    (run this from the project root directory)

How it connects:
    Streamlit → POST /predict → predict.py (ML + SHAP) → response back to Streamlit
    (Groq LLM layer will be wired in as a separate /recommend endpoint in app/llm.py)

Pydantic explanation:
FastAPI uses Pydantic models to automatically validate incoming JSON.
If the user sends a string instead of an int for a resistance flag, FastAPI
rejects the request with a clear error message — before our code even runs.
This is much cleaner than writing manual validation checks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from app.predict import predict as run_predict
from app.llm import generate_clinical_report

# ── FastAPI App Instance ──────────────────────────────────────────────────────

app = FastAPI(
    title="ResistAI — Antibiotic Resistance Prediction API",
    description=(
        "Predicts Ciprofloxacin resistance for bacterial isolates. "
        "Returns ML prediction, confidence score, and top SHAP-explained features."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",    # ReDoc UI at http://localhost:8000/redoc
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) explanation:
# Browsers and some HTTP clients block requests between different origins
# (e.g., localhost:8501 calling localhost:8000). We need to explicitly allow this.
# allow_origins=["*"] means: any origin can call this API (fine for a hackathon).
# In production, you would restrict this to your specific frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow Streamlit and any other client to call this API
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],   # allow all headers (e.g., Content-Type)
)


# ── Pydantic Models ───────────────────────────────────────────────────────────
# These define the expected shape of request and response bodies.
# FastAPI uses them to auto-validate inputs and auto-generate the /docs UI.

class PredictRequest(BaseModel):
    """
    Input schema for the /predict endpoint.
    All fields are optional — a clinician may not have complete data.
    Missing fields default to 0 (= not observed / unknown).
    """

    # -- Antibiotic resistance flags --
    # 1 = this strain is resistant to this antibiotic
    # 0 = susceptible, intermediate, or unknown
    imipenem_resistance:        Optional[int]   = Field(default=0, ge=0, le=1, description="Imipenem resistance (0=No, 1=Yes)")
    ceftazidime_resistance:     Optional[int]   = Field(default=0, ge=0, le=1, description="Ceftazidime resistance")
    gentamicin_resistance:      Optional[int]   = Field(default=0, ge=0, le=1, description="Gentamicin resistance")
    augmentin_resistance:       Optional[int]   = Field(default=0, ge=0, le=1, description="Augmentin (Amox-Clav) resistance")
    amoxicillin_resistance:     Optional[int]   = Field(default=0, ge=0, le=1, description="Amoxicillin resistance")
    cefazolin_resistance:       Optional[int]   = Field(default=0, ge=0, le=1, description="Cefazolin resistance")
    cefoxitin_resistance:       Optional[int]   = Field(default=0, ge=0, le=1, description="Cefoxitin resistance")
    amikacin_resistance:        Optional[int]   = Field(default=0, ge=0, le=1, description="Amikacin resistance")
    ofloxacin_resistance:       Optional[int]   = Field(default=0, ge=0, le=1, description="Ofloxacin resistance (same class as CIP)")
    chloramphenicol_resistance: Optional[int]   = Field(default=0, ge=0, le=1, description="Chloramphenicol resistance")
    cotrimoxazole_resistance:   Optional[int]   = Field(default=0, ge=0, le=1, description="Co-trimoxazole resistance")
    nitrofurantoin_resistance:  Optional[int]   = Field(default=0, ge=0, le=1, description="Nitrofurantoin resistance")
    colistin_resistance:        Optional[int]   = Field(default=0, ge=0, le=1, description="Colistin resistance")

    # -- Patient demographics --
    age_years:             Optional[float] = Field(default=0.0, ge=0, le=120, description="Patient age in years")
    is_female:             Optional[int]   = Field(default=0,   ge=0, le=1,   description="Patient sex: 1=Female, 0=Male")
    has_diabetes:          Optional[int]   = Field(default=0,   ge=0, le=1,   description="Diabetes diagnosis: 1=Yes, 0=No")
    has_hypertension:      Optional[int]   = Field(default=0,   ge=0, le=1,   description="Hypertension diagnosis: 1=Yes, 0=No")
    prior_hospitalization: Optional[int]   = Field(default=0,   ge=0, le=1,   description="Prior hospitalization: 1=Yes, 0=No")
    infection_freq:        Optional[float] = Field(default=0.0, ge=0,         description="Number of prior infections")

    # -- Bacterial species and sample location --
    # Species example: "Escherichia coli", "Klebsiella pneumoniae"
    # Location example: "ICU", "Urology", "Outpatient"
    species:  Optional[str] = Field(default="", description="Bacterial species name (e.g., 'Escherichia coli')")
    location: Optional[str] = Field(default="", description="Sample collection location (e.g., 'ICU')")

    class Config:
        # Show example values in the /docs UI so users know what to send
        json_schema_extra = {
            "example": {
                "ofloxacin_resistance":    1,
                "cotrimoxazole_resistance": 1,
                "gentamicin_resistance":   0,
                "age_years":               45,
                "is_female":               1,
                "has_diabetes":            0,
                "prior_hospitalization":   1,
                "species":                 "Escherichia coli",
            }
        }


class TopFeature(BaseModel):
    """One SHAP-explained feature in the prediction response."""
    feature:    str   # raw feature name, e.g. "ofloxacin_resistance"
    label:      str   # clinical label, e.g. "Ofloxacin resistance"
    shap_value: float # positive = pushes toward Resistant, negative = toward Susceptible
    direction:  str   # "increases_risk" or "decreases_risk"


class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    prediction:     str             # "Resistant" or "Susceptible"
    confidence:     float           # probability of Resistant (0.0–1.0), e.g. 0.73
    threshold_used: float           # threshold applied, e.g. 0.35
    top_features:   List[TopFeature] # top 3 SHAP-explained features


class RecommendResponse(BaseModel):
    """
    Response schema for the /recommend endpoint.
    Combines the full ML prediction with the Groq LLM clinical recommendation.
    The frontend only needs to call one endpoint to render the complete UI.
    """
    # ── ML prediction fields (same as PredictResponse) ──
    prediction:     str
    confidence:     float
    threshold_used: float
    top_features:   List[TopFeature]

    # ── LLM recommendation fields ──
    recommendation: str   # Full LLM text: clinical summary + alternatives + risk flag
    model_used:     str   # "llama-3.3-70b-versatile" or "fallback"
    prompt_tokens:  int   # Groq prompt tokens used (for rate-limit awareness)
    fallback_used:  bool  # True if Groq failed and fallback text was returned


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Streamlit calls GET /health before showing the prediction UI.
    If this returns anything other than 200 OK, Streamlit shows a warning.
    """
    return {"status": "ok", "model": "ResistAI v1.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    ML-only prediction endpoint.

    Flow:
    1. FastAPI validates the request body against PredictRequest schema
    2. We pass the validated data as a dict to predict.py
    3. predict.py builds the feature vector, runs the model, computes SHAP
    4. We return the result as a PredictResponse

    The 'response_model=PredictResponse' tells FastAPI to validate and
    serialize the response — this also powers the /docs auto-documentation.
    """
    try:
        input_data = request.model_dump()
        result = run_predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: PredictRequest):
    """
    Combined ML + LLM clinical decision endpoint.

    This is the main endpoint the Streamlit frontend calls.
    It runs the full pipeline in one request:
      1. ML inference + SHAP (predict.py)
      2. Groq LLM clinical recommendation (llm.py)

    Why combine into one endpoint?
    The frontend needs both outputs at the same time to render the full UI.
    One round-trip is simpler and faster than two sequential calls from Streamlit.

    Fallback behavior:
    If Groq is unavailable (rate-limited, network error), the LLM fields still
    return a graceful fallback message. The ML prediction is always returned.
    """
    try:
        input_data   = request.model_dump()
        ml_result    = run_predict(input_data)
        llm_result   = generate_clinical_report(ml_result, input_data)

        # Merge ML result and LLM result into a single flat response dict
        return {**ml_result, **llm_result}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )
