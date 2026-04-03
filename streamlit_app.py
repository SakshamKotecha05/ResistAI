"""
streamlit_app.py — ResistAI entry point for Streamlit Cloud deployment

This file merges the FastAPI backend logic directly into Streamlit so the app
runs as a single process — no uvicorn required. All ML inference, SHAP, and
Groq LLM calls are made via direct Python function calls instead of HTTP.

Local dev:  streamlit run streamlit_app.py
Cloud:      deploy this file as the app entry point on Streamlit Community Cloud
"""

import os
import math
import re
import sys
from typing import Optional
from pathlib import Path

import streamlit as st

# ── Secrets Bridge ────────────────────────────────────────────────────────────
# Streamlit Cloud stores secrets in st.secrets (set via the Cloud dashboard).
# Locally, app/llm.py loads from .env automatically.
# This bridge sets the env var before any app module is imported, so both
# paths (Cloud secrets and local .env) work without changing llm.py.

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Path fix ──────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `from app.predict import predict`
# resolves correctly when Streamlit runs this file.

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Direct imports from app modules ──────────────────────────────────────────
# These replace the HTTP calls to FastAPI. Same logic, zero network overhead.

from app.predict import predict as run_predict
from app.llm import generate_clinical_report

import plotly.graph_objects as go


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResistAI — Antibiotic Resistance Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS Theme ─────────────────────────────────────────────────────────────────

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-base:              #060D1B;
  --bg-surface:           #0B1628;
  --bg-elevated:          #0F2044;
  --border:               #1B3A6E;
  --border-dim:           #0D1E3A;
  --accent:               #00D4FF;
  --accent-dim:           rgba(0,212,255,0.10);
  --resistant:            #FF2B4E;
  --resistant-dim:        rgba(255,43,78,0.08);
  --susceptible:          #00F096;
  --susceptible-dim:      rgba(0,240,150,0.07);
  --text-primary:         #C8DCEF;
  --text-secondary:       #6A8BA8;
  --text-dim:             #3A5870;
  --font-display:         'Rajdhani', sans-serif;
  --font-body:            'Source Sans 3', sans-serif;
  --font-mono:            'JetBrains Mono', monospace;
}

html, body, .stApp {
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}

.stApp {
  background-color: var(--bg-base) !important;
  background-image:
    radial-gradient(circle at 1px 1px, rgba(27,58,110,0.22) 1px, transparent 0);
  background-size: 28px 28px;
}

#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #080F1E 0%, #050C17 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
  padding: 1rem 1.1rem !important;
}
[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid var(--border-dim) !important;
  margin: 10px 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
  color: var(--text-secondary) !important;
  font-size: 0.81rem !important;
  font-family: var(--font-body) !important;
}

[data-testid="stCheckbox"] label {
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  color: var(--text-secondary) !important;
}
[data-testid="stCheckbox"] label:hover { color: var(--text-primary) !important; }

.stButton > button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-display) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  border-radius: 3px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-dim) !important;
}

[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #000 !important;
  font-weight: 700 !important;
}
[data-testid="baseButton-primary"]:hover {
  background: #00B8DC !important;
  box-shadow: 0 0 18px rgba(0,212,255,0.28) !important;
}
[data-testid="baseButton-primary"]:disabled {
  background: var(--border-dim) !important;
  border-color: transparent !important;
  color: var(--text-dim) !important;
  box-shadow: none !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-dim) !important;
  color: var(--text-primary) !important;
  border-radius: 3px !important;
  font-family: var(--font-body) !important;
  font-size: 0.86rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px rgba(0,212,255,0.25) !important;
}
.stTextInput label, .stNumberInput label, .stSlider label {
  color: var(--text-secondary) !important;
  font-size: 0.81rem !important;
  font-family: var(--font-body) !important;
}

.main .block-container {
  padding-top: 16px !important;
  padding-bottom: 40px !important;
}

[data-testid="stSpinner"] > div {
  border-top-color: var(--accent) !important;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

EXAMPLE_PATIENT = {
    "ofloxacin_resistance":       1,
    "cotrimoxazole_resistance":   1,
    "gentamicin_resistance":      1,
    "amoxicillin_resistance":     1,
    "ceftazidime_resistance":     0,
    "imipenem_resistance":        0,
    "augmentin_resistance":       1,
    "cefazolin_resistance":       1,
    "cefoxitin_resistance":       0,
    "amikacin_resistance":        0,
    "chloramphenicol_resistance": 1,
    "nitrofurantoin_resistance":  0,
    "colistin_resistance":        0,
    "age_years":                  45.0,
    "is_female":                  1,
    "has_diabetes":               1,
    "has_hypertension":           0,
    "prior_hospitalization":      1,
    "infection_freq":             3.0,
    "species":                    "Escherichia coli",
    "location":                   "IFE-C",
}

BLANK_PATIENT = {
    "ofloxacin_resistance":       0,
    "cotrimoxazole_resistance":   0,
    "gentamicin_resistance":      0,
    "amoxicillin_resistance":     0,
    "ceftazidime_resistance":     0,
    "imipenem_resistance":        0,
    "augmentin_resistance":       0,
    "cefazolin_resistance":       0,
    "cefoxitin_resistance":       0,
    "amikacin_resistance":        0,
    "chloramphenicol_resistance": 0,
    "nitrofurantoin_resistance":  0,
    "colistin_resistance":        0,
    "age_years":                  0.0,
    "is_female":                  0,
    "has_diabetes":               0,
    "has_hypertension":           0,
    "prior_hospitalization":      0,
    "infection_freq":             0.0,
    "species":                    None,
    "location":                   None,
}


# ── Session State ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None


# ── Local Inference (replaces HTTP call to FastAPI) ───────────────────────────

def run_local_recommend(input_data: dict) -> Optional[dict]:
    """
    Runs ML inference + SHAP + Groq LLM directly in-process.
    Previously this was a POST to http://localhost:8000/recommend.
    Now it calls the same functions directly — identical output, no network needed.
    """
    try:
        ml_result  = run_predict(input_data)
        llm_result = generate_clinical_report(ml_result, input_data)
        return {**ml_result, **llm_result}
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None


# ── LLM Text Parser ───────────────────────────────────────────────────────────

def parse_llm_sections(text: str) -> dict:
    sections = {"summary": "", "alternatives": "", "risk_flag": ""}
    parts = re.split(
        r"(CLINICAL SUMMARY:|ALTERNATIVE ANTIBIOTICS:|RISK FLAG:)",
        text,
        flags=re.IGNORECASE,
    )
    current_key = None
    for part in parts:
        part_upper = part.strip().upper().rstrip(":")
        if part_upper == "CLINICAL SUMMARY":
            current_key = "summary"
        elif part_upper == "ALTERNATIVE ANTIBIOTICS":
            current_key = "alternatives"
        elif part_upper == "RISK FLAG":
            current_key = "risk_flag"
        elif current_key:
            sections[current_key] = part.strip()
    if not any(sections.values()):
        sections["summary"] = text.strip()
    return sections


# ── Confidence Gauge ──────────────────────────────────────────────────────────

def make_gauge_svg(confidence: float, color: str) -> str:
    cx, cy, r = 100, 88, 68
    end_angle = math.pi * (1.0 - max(confidence, 0.01))
    ex = cx + r * math.cos(end_angle)
    ey = cy - r * math.sin(end_angle)
    large_arc = 1 if confidence > 0.5 else 0

    bg_path  = f"M {cx - r},{cy} A {r},{r} 0 1,1 {cx + r},{cy}"
    val_path = f"M {cx - r},{cy} A {r},{r} 0 {large_arc},1 {ex:.2f},{ey:.2f}"

    ticks = []
    for pct in (0.25, 0.5, 0.75):
        a  = math.pi * (1.0 - pct)
        x1 = cx + (r - 7) * math.cos(a)
        y1 = cy - (r - 7) * math.sin(a)
        x2 = cx + (r + 7) * math.cos(a)
        y2 = cy - (r + 7) * math.sin(a)
        ticks.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="#1B3A6E" stroke-width="1.5"/>'
        )

    return f"""<svg viewBox="0 0 200 112" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:200px;display:block;margin:0 auto;">
  <path d="{bg_path}" fill="none" stroke="#0F2044" stroke-width="13" stroke-linecap="round"/>
  <path d="{val_path}" fill="none" stroke="{color}" stroke-width="13" stroke-linecap="round"
    style="filter:drop-shadow(0 0 5px {color}99);"/>
  {"".join(ticks)}
  <text x="100" y="83" text-anchor="middle"
    font-family="Rajdhani,sans-serif" font-size="29" font-weight="700"
    fill="{color}">{round(confidence * 100)}%</text>
  <text x="100" y="101" text-anchor="middle"
    font-family="Source Sans 3,sans-serif" font-size="9" letter-spacing="2"
    fill="#3A5870">CONFIDENCE</text>
  <text x="{cx - r - 6}" y="{cy + 13}" text-anchor="middle"
    font-family="JetBrains Mono,monospace" font-size="8" fill="#3A5870">0%</text>
  <text x="{cx + r + 6}" y="{cy + 13}" text-anchor="middle"
    font-family="JetBrains Mono,monospace" font-size="8" fill="#3A5870">100%</text>
</svg>"""


# ── Verdict Card ──────────────────────────────────────────────────────────────

def render_verdict_card(prediction: str, confidence: float, threshold: float):
    is_resistant = prediction == "Resistant"

    if is_resistant:
        color        = "#FF2B4E"
        bg           = "rgba(255,43,78,0.07)"
        border_color = "rgba(255,43,78,0.40)"
        border_left  = "#FF2B4E"
        label        = "RESISTANT"
        sub          = "to Ciprofloxacin — clinical intervention recommended"
        icon         = "⚠"
        anim_kf      = """@keyframes glow-r {
          0%,100%{box-shadow:0 0 0 0 rgba(255,43,78,0),0 0 0 0 rgba(255,43,78,0);}
          50%{box-shadow:0 0 22px 2px rgba(255,43,78,0.10),inset 0 0 20px rgba(255,43,78,0.04);}
        }"""
        anim_prop    = "animation:glow-r 3s ease-in-out infinite;"
    else:
        color        = "#00F096"
        bg           = "rgba(0,240,150,0.07)"
        border_color = "rgba(0,240,150,0.40)"
        border_left  = "#00F096"
        label        = "SUSCEPTIBLE"
        sub          = "to Ciprofloxacin — standard dosing protocol may apply"
        icon         = "✓"
        anim_kf      = """@keyframes glow-s {
          0%,100%{box-shadow:0 0 0 0 rgba(0,240,150,0),0 0 0 0 rgba(0,240,150,0);}
          50%{box-shadow:0 0 22px 2px rgba(0,240,150,0.10),inset 0 0 20px rgba(0,240,150,0.04);}
        }"""
        anim_prop    = "animation:glow-s 3s ease-in-out infinite;"

    gauge = make_gauge_svg(confidence, color)
    thr   = f"{round(threshold * 100)}%"

    st.markdown(f"""<style>{anim_kf}</style>
<div style="background:{bg};border:1px solid {border_color};border-left:4px solid {border_left};border-radius:8px;padding:22px 28px;margin-bottom:22px;display:flex;align-items:center;gap:28px;position:relative;overflow:hidden;{anim_prop}">
  <div style="position:absolute;top:0;left:0;right:0;bottom:0;
    background:radial-gradient(ellipse at top left,{bg.replace('0.07','0.13')} 0%,transparent 60%);
    pointer-events:none;"></div>
  <div style="font-size:2.8rem;color:{color};font-family:Rajdhani,sans-serif;
    font-weight:700;line-height:1;flex:0 0 auto;
    text-shadow:0 0 22px {color}88;position:relative;z-index:1;">{icon}</div>
  <div style="flex:1;position:relative;z-index:1;">
    <div style="font-family:Rajdhani,sans-serif;font-size:2.1rem;font-weight:700;
      color:{color};letter-spacing:0.08em;line-height:1;
      text-shadow:0 0 24px {color}44;">{label}</div>
    <div style="font-family:'Source Sans 3',sans-serif;font-size:0.87rem;
      color:#6A8BA8;margin-top:5px;">{sub}</div>
    <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.69rem;
        padding:3px 9px;border-radius:3px;background:rgba(27,58,110,0.45);
        border:1px solid #1B3A6E;color:#3A5870;">threshold: {thr}</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.69rem;
        padding:3px 9px;border-radius:3px;background:rgba(27,58,110,0.45);
        border:1px solid #1B3A6E;color:#3A5870;">Random Forest + SHAP</span>
    </div>
  </div>
  <div style="flex:0 0 190px;position:relative;z-index:1;">{gauge}</div>
</div>
""", unsafe_allow_html=True)


# ── SHAP Chart ────────────────────────────────────────────────────────────────

def render_shap_chart(top_features: list):
    if not top_features:
        st.caption("No SHAP feature data available.")
        return

    labels = [f.get("label", f.get("feature", "?")) for f in top_features]
    values = [f.get("shap_value", 0.0)              for f in top_features]
    colors = ["#FF2B4E" if v > 0 else "#00F096"     for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.88, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#6A8BA8"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            title=None,
            tickfont=dict(family="JetBrains Mono", size=9, color="#3A5870"),
            gridcolor="#0D1E3A",
            zeroline=True, zerolinecolor="#1B3A6E", zerolinewidth=1,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Source Sans 3", size=12, color="#6A8BA8"),
            gridcolor="rgba(0,0,0,0)", zeroline=False,
        ),
        plot_bgcolor="#0B1628",
        paper_bgcolor="#0B1628",
        font=dict(family="Source Sans 3", color="#C8DCEF"),
        height=195,
        margin=dict(l=0, r=64, t=8, b=28),
        bargap=0.42,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div style="font-size:0.71rem;color:#3A5870;font-family:\'Source Sans 3\',sans-serif;margin-top:-6px;">'
        '<span style="color:#FF2B4E;">■</span> Increases resistance risk &nbsp;·&nbsp;'
        '<span style="color:#00F096;">■</span> Decreases resistance risk &nbsp;·&nbsp;'
        'Bar length = feature importance</div>',
        unsafe_allow_html=True,
    )


# ── Section Header ────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = ""):
    sub_html = (
        f'<div style="font-family:\'Source Sans 3\',sans-serif;'
        f'font-size:0.77rem;color:#3A5870;margin-top:3px;">{subtitle}</div>'
    ) if subtitle else ""

    st.markdown(
        f'<div style="margin-bottom:14px;">'
        f'<div style="font-family:Rajdhani,sans-serif;font-size:0.7rem;font-weight:600;'
        f'letter-spacing:0.16em;text-transform:uppercase;color:#6A8BA8;">{title}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


# ── Clinical Report ───────────────────────────────────────────────────────────

def render_clinical_report(recommend: str, fallback: bool):
    if fallback:
        st.markdown(
            '<div style="background:rgba(255,43,78,0.06);border:1px solid rgba(255,43,78,0.22);'
            'border-radius:4px;padding:8px 14px;font-size:0.77rem;color:#FF2B4E;margin-bottom:12px;'
            'font-family:\'Source Sans 3\',sans-serif;">'
            '⚠ Groq API unavailable — SHAP analysis above is still valid.</div>',
            unsafe_allow_html=True,
        )

    sections = parse_llm_sections(recommend)

    def card(text: str, accent: str, dim: str, heading: str):
        body = text.replace("\n", "<br>")
        st.markdown(
            f'<div style="background:rgba({dim},0.05);border-left:3px solid rgba({accent},0.45);'
            f'border-radius:0 4px 4px 0;padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-family:Rajdhani,sans-serif;font-size:0.67rem;font-weight:600;'
            f'letter-spacing:0.16em;text-transform:uppercase;color:#3A5870;margin-bottom:8px;">{heading}</div>'
            f'<div style="font-family:\'Source Sans 3\',sans-serif;font-size:0.87rem;'
            f'line-height:1.68;color:#C8DCEF;">{body}</div></div>',
            unsafe_allow_html=True,
        )

    if sections["summary"]:
        card(sections["summary"],     "0,212,255", "0,212,255", "Clinical Summary")
    if sections["alternatives"]:
        card(sections["alternatives"], "0,240,150", "0,240,150", "Alternative Antibiotics")
    if sections["risk_flag"]:
        card(sections["risk_flag"],    "255,43,78", "255,43,78", "Risk Flag")


# ── Results Panel ─────────────────────────────────────────────────────────────

def render_results(result: dict):
    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0.0)
    threshold  = result.get("threshold_used", 0.35)
    top_feats  = result.get("top_features", [])
    recommend  = result.get("recommendation", "")
    model_used = result.get("model_used", "")
    fallback   = result.get("fallback_used", False)
    tokens     = result.get("prompt_tokens", 0)

    render_verdict_card(prediction, confidence, threshold)

    col_shap, col_llm = st.columns([1, 1], gap="large")

    with col_shap:
        section_header("Feature Analysis", "SHAP — which factors drove this prediction")
        render_shap_chart(top_feats)

    with col_llm:
        section_header("Clinical Decision Support", "Llama 3.3 70B via Groq")
        render_clinical_report(recommend, fallback)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #0D1E3A;margin:24px 0 14px;'>",
        unsafe_allow_html=True,
    )
    foot_col, btn_col = st.columns([5, 1])
    with foot_col:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;color:#3A5870;margin:4px 0;">'
            f'model: <span style="color:#6A8BA8;">{model_used}</span> &nbsp;·&nbsp;'
            f'tokens: <span style="color:#6A8BA8;">{tokens}</span> &nbsp;·&nbsp;'
            f'ResistAI v1.0 — research and educational use only</div>',
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.result = None
            for key, val in BLANK_PATIENT.items():
                st.session_state[key] = val
            st.rerun()


# ── Empty State ───────────────────────────────────────────────────────────────

def render_empty_state():
    st.markdown(
        """<div style="text-align:center;padding:72px 30px 80px;">
          <div style="font-size:3.6rem;margin-bottom:18px;
            filter:drop-shadow(0 0 22px rgba(0,212,255,0.28));">🧬</div>
          <div style="font-family:Rajdhani,sans-serif;font-size:1.5rem;font-weight:700;
            color:#6A8BA8;letter-spacing:0.07em;margin-bottom:10px;">NO ISOLATE LOADED</div>
          <div style="font-family:'Source Sans 3',sans-serif;font-size:0.9rem;color:#3A5870;
            max-width:370px;margin:0 auto;line-height:1.72;">
            Enter the antibiotic resistance profile in the sidebar and click
            <span style="color:#00D4FF;font-weight:600;">Analyze Resistance</span>,
            or use <span style="color:#C8DCEF;font-weight:600;">Load Example Patient</span>
            to run a pre-loaded demo case.
          </div>
          <div style="margin-top:28px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;
              padding:4px 11px;border:1px solid #1B3A6E;border-radius:3px;
              color:#3A5870;background:#0B1628;">Random Forest</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;
              padding:4px 11px;border:1px solid #1B3A6E;border-radius:3px;
              color:#3A5870;background:#0B1628;">SHAP Explainability</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;
              padding:4px 11px;border:1px solid #1B3A6E;border-radius:3px;
              color:#3A5870;background:#0B1628;">Groq Llama 3.3 70B</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.67rem;
              padding:4px 11px;border:1px solid #1B3A6E;border-radius:3px;
              color:#3A5870;background:#0B1628;">Ciprofloxacin AMR</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.markdown(
        """<div style="padding:2px 0 14px;">
          <div style="font-family:Rajdhani,sans-serif;font-size:1.45rem;
            font-weight:700;letter-spacing:0.05em;color:#C8DCEF;">
            Resist<span style="color:#00D4FF;">AI</span>
          </div>
          <div style="margin-top:5px;display:flex;align-items:center;gap:6px;">
            <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
              background:#00F096;box-shadow:0 0 6px #00F096;"></span>
            <span style="font-family:'Source Sans 3',sans-serif;
              font-size:0.74rem;color:#3A5870;">model ready</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    if st.sidebar.button("Load Example Patient", use_container_width=True):
        for key, val in EXAMPLE_PATIENT.items():
            st.session_state[key] = val
        st.rerun()

    st.sidebar.markdown(
        '<div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;font-weight:600;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#6A8BA8;padding:14px 0 7px;">'
        'Antibiotic Resistance Profile</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Mark every antibiotic this strain is resistant to.")

    ab_fields = {
        "ofloxacin_resistance":       "Ofloxacin (fluoroquinolone)",
        "cotrimoxazole_resistance":   "Co-trimoxazole (TMP-SMX)",
        "gentamicin_resistance":      "Gentamicin (aminoglycoside)",
        "amoxicillin_resistance":     "Amoxicillin (penicillin)",
        "augmentin_resistance":       "Augmentin (amox-clav)",
        "ceftazidime_resistance":     "Ceftazidime (3rd-gen ceph)",
        "cefazolin_resistance":       "Cefazolin (1st-gen ceph)",
        "cefoxitin_resistance":       "Cefoxitin (cephamycin)",
        "amikacin_resistance":        "Amikacin (reserve aminoglycoside)",
        "imipenem_resistance":        "Imipenem (carbapenem) ⚠",
        "colistin_resistance":        "Colistin (last resort) ⚠",
        "chloramphenicol_resistance": "Chloramphenicol",
        "nitrofurantoin_resistance":  "Nitrofurantoin",
    }

    antibiotic_values = {}
    for field, label in ab_fields.items():
        antibiotic_values[field] = int(st.sidebar.checkbox(label, key=field))

    st.sidebar.divider()
    st.sidebar.markdown(
        '<div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;font-weight:600;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#6A8BA8;padding:12px 0 7px;">'
        'Patient Demographics</div>',
        unsafe_allow_html=True,
    )

    age              = st.sidebar.slider("Patient age (years)", 0, 100, key="age_years")
    is_female        = int(st.sidebar.checkbox("Female patient",        key="is_female"))
    has_diabetes     = int(st.sidebar.checkbox("Diabetes diagnosis",    key="has_diabetes"))
    has_hypertension = int(st.sidebar.checkbox("Hypertension",          key="has_hypertension"))
    prior_hosp       = int(st.sidebar.checkbox("Prior hospitalization", key="prior_hospitalization"))
    infection_freq   = st.sidebar.number_input(
        "Prior infection count", min_value=0.0, max_value=50.0, step=1.0, key="infection_freq"
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;font-weight:600;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#6A8BA8;padding:12px 0 7px;">'
        'Bacterial Context</div>',
        unsafe_allow_html=True,
    )

    species = st.sidebar.selectbox(
        "Bacterial species",
        options=[
            "Escherichia coli",
            "Klebsiella pneumoniae",
            "Proteus mirabilis",
            "Citrobacter spp.",
            "Enterobacteria spp.",
            "Other",
        ],
        index=None,
        placeholder="Select species...",
        key="species",
    )
    location = st.sidebar.selectbox(
        "Sample location",
        options=[
            "EDE-C", "EDE-S", "EDE-T",
            "IFE-C", "IFE-S", "IFE-T",
            "IWO-C", "IWO-S", "IWO-T",
            "OSU-C", "OSU-S", "OSU-T",
        ],
        index=None,
        placeholder="Select location...",
        key="location",
    )

    st.sidebar.divider()

    return {
        **antibiotic_values,
        "age_years":             float(age),
        "is_female":             is_female,
        "has_diabetes":          has_diabetes,
        "has_hypertension":      has_hypertension,
        "prior_hospitalization": prior_hosp,
        "infection_freq":        float(infection_freq),
        "species":               species,
        "location":              location,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        """<div style="margin-bottom:18px;display:flex;align-items:baseline;gap:14px;">
          <div style="font-family:Rajdhani,sans-serif;font-size:1.75rem;font-weight:700;
            letter-spacing:0.05em;color:#C8DCEF;">
            Resist<span style="color:#00D4FF;">AI</span>
          </div>
          <div style="font-family:'Source Sans 3',sans-serif;font-size:0.82rem;color:#3A5870;">
            Ciprofloxacin resistance prediction &nbsp;·&nbsp;
            ML classifier &nbsp;+&nbsp; SHAP explainability &nbsp;+&nbsp; Groq LLM
          </div>
        </div>
        <hr style="border:none;border-top:1px solid #0D1E3A;margin:0 0 22px;">""",
        unsafe_allow_html=True,
    )

    input_data = render_sidebar()

    analyze_clicked = st.sidebar.button(
        "Analyze Resistance",
        use_container_width=True,
        type="primary",
    )

    if analyze_clicked:
        with st.spinner("Running ML inference + Groq clinical analysis..."):
            result = run_local_recommend(input_data)
            if result:
                st.session_state.result = result

    if st.session_state.result:
        render_results(st.session_state.result)
    else:
        render_empty_state()


main()
