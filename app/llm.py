"""
app/llm.py — Groq Clinical Decision Layer

What this file does:
1. Takes the ML prediction result from predict.py (outcome, confidence, top SHAP features)
2. Builds a structured, evidence-grounded prompt for the Groq LLM
3. Calls Groq's Llama 3.3 70B model to generate a clinical recommendation
4. Returns a plain-English clinical report: explanation + alternative antibiotics + risk flag

Why this is the differentiator:
Every team will build a resistance classifier. This layer translates the ML output into
clinical language a doctor can actually act on — explaining WHY the strain is resistant
and WHAT to use instead. The SHAP biology bridge (utils.py) feeds real biological context
into the prompt, so the LLM reasons from evidence, not just the label "Resistant."

Groq explanation:
Groq is an AI inference provider that runs open-source LLMs (like Meta's Llama 3.3 70B)
at very high speed on custom hardware called LPUs (Language Processing Units).
We use it because it has a generous free tier (14,400 requests/day) and the fastest
inference latency of any free LLM API — critical for a live demo.

Temperature explanation:
Temperature controls how "creative" vs "deterministic" the LLM is.
- Temperature 0.0 = always the same output (too rigid for natural language)
- Temperature 1.0 = very varied, sometimes incoherent output
- Temperature 0.3 = consistent clinical language with slight natural variation
  This is the right setting for medical recommendation text.
"""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

from app.utils import get_feature_label, get_feature_biology

# Load API keys from .env using an absolute path anchored to this file's location.
# This works regardless of which directory uvicorn is launched from.
# app/llm.py → parent = app/ → parent = project root → .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


# ── Constants ─────────────────────────────────────────────────────────────────

GROQ_MODEL    = "llama-3.3-70b-versatile"   # Best reasoning model on Groq free tier
TEMPERATURE   = 0.3                          # Low = consistent clinical language
MAX_TOKENS    = 600                          # Enough for 3-section response, not wasteful

# Fallback message shown when Groq is unavailable (rate limit, network, etc.)
FALLBACK_TEXT = (
    "Clinical recommendation unavailable — Groq API could not be reached. "
    "Please review the SHAP feature analysis above and consult standard antibiotic "
    "stewardship guidelines for your institution."
)


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def build_prompt(predict_result: dict, input_data: dict) -> str:
    """
    Builds the structured LLM prompt from the ML prediction result and raw input.

    Args:
        predict_result: dict returned by predict.py — contains prediction, confidence,
                        threshold_used, top_features (list of SHAP-explained features)
        input_data:     dict of raw user inputs — species, location, demographics

    Returns:
        A fully-formed prompt string ready to send to the Groq API.

    Why build the prompt separately from the API call?
    Separation of concerns: we can inspect, print, or adjust the prompt independently.
    This is especially useful for debugging when the LLM output isn't quite right.
    """

    prediction  = predict_result.get("prediction", "Unknown")
    confidence  = predict_result.get("confidence", 0.0)
    top_feats   = predict_result.get("top_features", [])

    # -- Section 1: Prediction summary ----------------------------------------
    confidence_pct = round(confidence * 100, 1)
    prediction_line = (
        f"PREDICTION: {prediction} (model confidence: {confidence_pct}%)"
    )

    # -- Section 2: SHAP evidence with biological context ---------------------
    # This is the key innovation: each SHAP feature is paired with a one-sentence
    # biological explanation from utils.py. The LLM gets real evidence, not just
    # a list of numbers.

    shap_lines = []
    for i, feat in enumerate(top_feats[:3], start=1):
        feat_name  = feat.get("feature", "")
        label      = feat.get("label", feat_name)
        shap_val   = feat.get("shap_value", 0.0)
        direction  = feat.get("direction", "")
        biology    = get_feature_biology(feat_name)

        direction_text = "increases resistance risk" if direction == "increases_risk" else "decreases resistance risk"
        shap_str = f"{shap_val:+.3f}"

        line = f"{i}. {label} (SHAP: {shap_str}, {direction_text})"
        if biology:
            line += f"\n   Biology: {biology}"
        shap_lines.append(line)

    shap_block = "\n".join(shap_lines) if shap_lines else "No SHAP features available."

    # -- Section 3: Patient and bacterial context ------------------------------
    species  = (input_data.get("species")  or "").strip() or "Not specified"
    location = (input_data.get("location") or "").strip() or "Not specified"

    # Build a readable demographics summary from whatever was provided
    demo_parts = []
    age = input_data.get("age_years")
    if age and float(age) > 0:
        demo_parts.append(f"Age {int(age)}")
    if input_data.get("is_female") == 1:
        demo_parts.append("Female")
    elif input_data.get("is_female") == 0:
        demo_parts.append("Male")
    if input_data.get("has_diabetes") == 1:
        demo_parts.append("Diabetic")
    if input_data.get("has_hypertension") == 1:
        demo_parts.append("Hypertensive")
    if input_data.get("prior_hospitalization") == 1:
        demo_parts.append("Prior hospitalization")

    demographics = ", ".join(demo_parts) if demo_parts else "Not provided"

    # -- Assemble the full prompt ----------------------------------------------
    # The output format is strictly constrained so the frontend can parse it cleanly.
    # We tell the LLM exactly what sections to write and in what order.

    prompt = f"""You are an expert clinical microbiologist providing antibiotic stewardship guidance.
You are reviewing an AI-generated Ciprofloxacin resistance prediction for a bacterial isolate.
Write clearly for a clinical audience (infectious disease specialists, hospitalists).
Be concise, evidence-based, and actionable. Do NOT add disclaimers or introductory sentences.

--- PREDICTION DATA ---

{prediction_line}

TOP CONTRIBUTING FACTORS (from SHAP explainability analysis):
{shap_block}

PATIENT AND BACTERIAL CONTEXT:
- Bacterial species: {species}
- Sample location: {location}
- Patient demographics: {demographics}

--- OUTPUT REQUIRED ---

Write exactly three sections with these exact headers:

CLINICAL SUMMARY:
Explain in plain language why this strain is predicted {"resistant" if prediction == "Resistant" else "susceptible"} to Ciprofloxacin, referencing the key resistance factors above. (2lines at maximum.)

ALTERNATIVE ANTIBIOTICS:
(List 2–3 options) For each antibiotic, give: name, drug class, and one sentence explaining why it may retain activity against this strain given the resistance profile above. Focus on options that are not cross-resistant with Ciprofloxacin. (Limit this to 4 lines at maximum.)

RISK FLAG:
(1 sentence) Write a brief clinical risk note suitable for a patient's medical chart, highlighting the most critical concern from this resistance profile. (Restrict this output to 2-3 lines maximum.)"""

    return prompt


# ── Groq API Call ─────────────────────────────────────────────────────────────

def get_recommendation(prompt: str) -> tuple[str, int]:
    """
    Sends the prompt to Groq's Llama 3.3 70B model and returns the response.

    Args:
        prompt: the fully assembled prompt string from build_prompt()

    Returns:
        tuple of (response_text: str, prompt_tokens_used: int)

    Raises:
        Exception if the Groq call fails (caller handles this with fallback)

    Why use the system role vs. putting everything in user?
    The "system" role sets persistent behavioral context for the model —
    it's more reliable for maintaining tone and format than putting everything
    in the user message. We keep it minimal here since all detail is in the prompt.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior clinical microbiologist. Provide evidence-based, "
                    "concise antibiotic guidance. Follow the output format exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    text         = response.choices[0].message.content.strip()
    tokens_used  = response.usage.prompt_tokens if response.usage else 0

    return text, tokens_used


# ── Public Interface ───────────────────────────────────────────────────────────

def generate_clinical_report(predict_result: dict, input_data: dict) -> dict:
    """
    Main function called by main.py's /recommend endpoint.

    Orchestrates: build_prompt → get_recommendation → return structured result.

    Args:
        predict_result: output dict from predict.py
        input_data:     raw input dict from the API request

    Returns:
        dict with keys:
            recommendation  — full LLM text (3 sections: summary, alternatives, risk flag)
            model_used      — model name (for transparency in the UI)
            prompt_tokens   — tokens used in the prompt (optional, for rate-limit awareness)
            fallback_used   — True if Groq failed and the fallback message was returned

    Why return fallback_used as a flag?
    The Streamlit frontend can check this flag and show a subtle warning like
    "Groq unavailable — showing SHAP analysis only" instead of silently serving
    the fallback text as if it were a real recommendation.
    """

    try:
        prompt              = build_prompt(predict_result, input_data)
        recommendation, tokens = get_recommendation(prompt)

        return {
            "recommendation": recommendation,
            "model_used":     GROQ_MODEL,
            "prompt_tokens":  tokens,
            "fallback_used":  False,
        }

    except Exception as e:
        # Log the error server-side so we can debug, but don't crash the API
        print(f"[llm] Groq call failed: {e}")

        return {
            "recommendation": FALLBACK_TEXT,
            "model_used":     "fallback",
            "prompt_tokens":  0,
            "fallback_used":  True,
        }
