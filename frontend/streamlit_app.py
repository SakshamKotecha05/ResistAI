"""
frontend/streamlit_app.py — ResistAI Streamlit UI (v2 — Clinical Diagnostic Interface)

Design: Clinical Diagnostic Interface
- Dark navy (#060D1B) base with electric cyan (#00D4FF) accents
- Typography: Rajdhani (display) + Source Sans 3 (body) + JetBrains Mono (data)
- Animated glowing verdict card with SVG confidence gauge
- Themed Plotly SHAP chart matching the dark palette
- All API logic identical to v1
"""

import math
import re
from typing import Optional

import requests
import streamlit as st
import plotly.graph_objects as go


# ── Page Config ───────────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script.

st.set_page_config(
    page_title="ResistAI — Antibiotic Resistance Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS Theme Injection ───────────────────────────────────────────────────────
# Injected once at the top via st.markdown. Streamlit renders this into the page
# <head>. CSS variables centralise all colour decisions so one edit recolours everything.

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
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

/* ── Base ── */
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

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Sidebar ── */
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

/* ── Sidebar labels ── */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
  color: var(--text-secondary) !important;
  font-size: 0.81rem !important;
  font-family: var(--font-body) !important;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label {
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  color: var(--text-secondary) !important;
}
[data-testid="stCheckbox"] label:hover { color: var(--text-primary) !important; }

/* ── Buttons — base ── */
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

/* Primary button */
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

/* ── Text + Number inputs ── */
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

/* ── Main block container ── */
.main .block-container {
  padding-top: 16px !important;
  padding-bottom: 40px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
  border-top-color: var(--accent) !important;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE    = "http://localhost:8000"
PREDICT_URL = f"{API_BASE}/recommend"
HEALTH_URL  = f"{API_BASE}/health"

# Pre-loaded case that reliably produces "Resistant" — used during demos.
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
    "location":                   "ICU",
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
    "species":                    "",
    "location":                   "",
}


# ── Session State Init ────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None


# ── API Functions ─────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Returns True if the FastAPI backend responds on /health."""
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_recommend_api(input_data: dict) -> Optional[dict]:
    """
    Posts to /recommend and returns the parsed JSON dict.
    Returns None on any error so the caller can show a message instead of crashing.
    """
    try:
        response = requests.post(PREDICT_URL, json=input_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend API. Is `uvicorn app.main:app --reload` running?")
        return None
    except requests.exceptions.Timeout:
        st.error("API timed out — Groq call may be slow. Try again.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── LLM Text Parser ───────────────────────────────────────────────────────────

def parse_llm_sections(text: str) -> dict:
    """
    Splits the LLM recommendation text into three named sections.
    The LLM is instructed to use exact headers: CLINICAL SUMMARY, ALTERNATIVE ANTIBIOTICS, RISK FLAG.
    If a header is missing the full text falls through to 'summary'.
    """
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
    """
    Builds an inline SVG semicircle gauge showing model confidence.

    Geometry:
      - Arc spans 180° across the top half of a circle (left → right).
      - At confidence=0  : end point == start point  → no visible fill.
      - At confidence=0.5: end point is at the top center.
      - At confidence=1.0: end point == right side → full semicircle.

    SVG arc path: M sx,sy A r,r 0 large-arc-flag,sweep-flag ex,ey
      - sweep=1 means clockwise.
      - large-arc-flag=1 when arc > 180° (i.e. confidence > 0.5).
    """
    cx, cy, r = 100, 88, 68
    # End angle: π at confidence=0 (same as start), 0 at confidence=1 (right side).
    end_angle = math.pi * (1.0 - max(confidence, 0.01))
    ex = cx + r * math.cos(end_angle)
    ey = cy - r * math.sin(end_angle)  # subtract because SVG y increases downward

    large_arc = 1 if confidence > 0.5 else 0

    bg_path  = f"M {cx - r},{cy} A {r},{r} 0 1,1 {cx + r},{cy}"
    val_path = f"M {cx - r},{cy} A {r},{r} 0 {large_arc},1 {ex:.2f},{ey:.2f}"

    # Tick marks at 25 / 50 / 75 %
    ticks = []
    for pct in (0.25, 0.5, 0.75):
        a   = math.pi * (1.0 - pct)
        x1  = cx + (r - 7) * math.cos(a)
        y1  = cy - (r - 7) * math.sin(a)
        x2  = cx + (r + 7) * math.cos(a)
        y2  = cy - (r + 7) * math.sin(a)
        ticks.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="#1B3A6E" stroke-width="1.5"/>'
        )

    return f"""<svg viewBox="0 0 200 112" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:200px;display:block;margin:0 auto;">
  <!-- Background track -->
  <path d="{bg_path}" fill="none" stroke="#0F2044" stroke-width="13" stroke-linecap="round"/>
  <!-- Value fill -->
  <path d="{val_path}" fill="none" stroke="{color}" stroke-width="13" stroke-linecap="round"
    style="filter:drop-shadow(0 0 5px {color}99);"/>
  <!-- Ticks -->
  {"".join(ticks)}
  <!-- Percentage label -->
  <text x="100" y="83" text-anchor="middle"
    font-family="Rajdhani,sans-serif" font-size="29" font-weight="700"
    fill="{color}">{round(confidence * 100)}%</text>
  <text x="100" y="101" text-anchor="middle"
    font-family="Source Sans 3,sans-serif" font-size="9" letter-spacing="2"
    fill="#3A5870">CONFIDENCE</text>
  <!-- Range labels -->
  <text x="{cx - r - 6}" y="{cy + 13}" text-anchor="middle"
    font-family="JetBrains Mono,monospace" font-size="8" fill="#3A5870">0%</text>
  <text x="{cx + r + 6}" y="{cy + 13}" text-anchor="middle"
    font-family="JetBrains Mono,monospace" font-size="8" fill="#3A5870">100%</text>
</svg>"""


# ── Verdict Card ──────────────────────────────────────────────────────────────

def render_verdict_card(prediction: str, confidence: float, threshold: float):
    """
    Full-width card showing the resistance verdict.
    Resistant cards pulse with a red glow using a CSS @keyframes animation.
    Susceptible cards are static with a green border accent.
    """
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
        bg           = "rgba(0,240,150,0.06)"
        border_color = "rgba(0,240,150,0.30)"
        border_left  = "#00F096"
        label        = "SUSCEPTIBLE"
        sub          = "to Ciprofloxacin — standard dosing protocol may apply"
        icon         = "✓"
        anim_kf      = ""
        anim_prop    = ""

    gauge = make_gauge_svg(confidence, color)
    thr   = f"{round(threshold * 100)}%"

    st.markdown(f"""
<style>{anim_kf}</style>
<div style="
  background:{bg};
  border:1px solid {border_color};
  border-left:4px solid {border_left};
  border-radius:8px;
  padding:22px 28px;
  margin-bottom:22px;
  display:flex;
  align-items:center;
  gap:28px;
  position:relative;
  overflow:hidden;
  {anim_prop}
">
  <!-- Radial glow overlay -->
  <div style="position:absolute;top:0;left:0;right:0;bottom:0;
    background:radial-gradient(ellipse at top left,{bg.replace('0.07','0.13')} 0%,transparent 60%);
    pointer-events:none;"></div>

  <!-- Icon -->
  <div style="
    font-size:2.8rem;color:{color};font-family:Rajdhani,sans-serif;
    font-weight:700;line-height:1;flex:0 0 auto;
    text-shadow:0 0 22px {color}88;position:relative;z-index:1;
  ">{icon}</div>

  <!-- Text -->
  <div style="flex:1;position:relative;z-index:1;">
    <div style="
      font-family:Rajdhani,sans-serif;font-size:2.1rem;font-weight:700;
      color:{color};letter-spacing:0.08em;line-height:1;
      text-shadow:0 0 24px {color}44;
    ">{label}</div>
    <div style="
      font-family:'Source Sans 3',sans-serif;font-size:0.87rem;
      color:#6A8BA8;margin-top:5px;
    ">{sub}</div>
    <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.69rem;
        padding:3px 9px;border-radius:3px;background:rgba(27,58,110,0.45);
        border:1px solid #1B3A6E;color:#3A5870;">threshold: {thr}</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.69rem;
        padding:3px 9px;border-radius:3px;background:rgba(27,58,110,0.45);
        border:1px solid #1B3A6E;color:#3A5870;">Random Forest + SHAP</span>
    </div>
  </div>

  <!-- Gauge -->
  <div style="flex:0 0 190px;position:relative;z-index:1;">{gauge}</div>
</div>
""", unsafe_allow_html=True)


# ── SHAP Chart ────────────────────────────────────────────────────────────────

def render_shap_chart(top_features: list):
    """
    Horizontal Plotly bar chart of top SHAP features styled to match the dark theme.
    Red = pushes toward Resistant, Green = pushes toward Susceptible.
    """
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
        marker=dict(
            color=colors,
            opacity=0.88,
            line=dict(width=0),
        ),
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
            zeroline=True,
            zerolinecolor="#1B3A6E",
            zerolinewidth=1,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Source Sans 3", size=12, color="#6A8BA8"),
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
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
        '<p style="font-size:0.71rem;color:#3A5870;font-family:\'Source Sans 3\',sans-serif;margin-top:-6px;">'
        '<span style="color:#FF2B4E;">■</span> Increases resistance risk &nbsp;·&nbsp;'
        '<span style="color:#00F096;">■</span> Decreases resistance risk &nbsp;·&nbsp;'
        'Bar length = feature importance</p>',
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
    """Renders the three LLM output sections as styled clinical cards."""

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
        # Convert newlines to <br> so LLM list formatting survives HTML rendering
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
        card(sections["summary"],  "0,212,255",  "0,212,255",  "Clinical Summary")
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

    # ── Verdict + confidence gauge ────────────────────────────────────────────
    render_verdict_card(prediction, confidence, threshold)

    # ── SHAP (left) | Clinical report (right) ─────────────────────────────────
    col_shap, col_llm = st.columns([1, 1], gap="large")

    with col_shap:
        section_header("Feature Analysis", "SHAP — which factors drove this prediction")
        render_shap_chart(top_feats)

    with col_llm:
        section_header("Clinical Decision Support", f"Llama 3.3 70B via Groq")
        render_clinical_report(recommend, fallback)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid #0D1E3A;margin:24px 0 14px;'>",
        unsafe_allow_html=True,
    )
    foot_col, btn_col = st.columns([5, 1])
    with foot_col:
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.69rem;color:#3A5870;margin:4px 0;">'
            f'model: <span style="color:#6A8BA8;">{model_used}</span> &nbsp;·&nbsp;'
            f'tokens: <span style="color:#6A8BA8;">{tokens}</span> &nbsp;·&nbsp;'
            f'ResistAI v1.0 — research and educational use only</p>',
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

def render_sidebar() -> tuple:
    """
    Renders all input widgets in the sidebar and returns (input_data, api_ok).

    Pattern note — Load Example Patient:
    Streamlit ignores the `value=` param if a widget key already exists in session_state.
    We set st.session_state[key] directly BEFORE the widgets render to force the update.
    This is the correct Streamlit pattern for programmatic widget pre-population.
    """

    api_ok = check_api_health()

    # ── Logo + status ─────────────────────────────────────────────────────────
    dot_color   = "#00F096" if api_ok else "#FF2B4E"
    dot_shadow  = f"box-shadow:0 0 6px {dot_color};" if api_ok else ""
    status_text = "backend connected" if api_ok else "backend offline"

    st.sidebar.markdown(
        f"""<div style="padding:2px 0 14px;">
          <div style="font-family:Rajdhani,sans-serif;font-size:1.45rem;
            font-weight:700;letter-spacing:0.05em;color:#C8DCEF;">
            Resist<span style="color:#00D4FF;">AI</span>
          </div>
          <div style="margin-top:5px;display:flex;align-items:center;gap:6px;">
            <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
              background:{dot_color};{dot_shadow}"></span>
            <span style="font-family:'Source Sans 3',sans-serif;
              font-size:0.74rem;color:#3A5870;">{status_text}</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    # ── Load Example button ───────────────────────────────────────────────────
    if st.sidebar.button("Load Example Patient", use_container_width=True):
        for key, val in EXAMPLE_PATIENT.items():
            st.session_state[key] = val
        st.rerun()

    # ── Antibiotic Resistance Profile ─────────────────────────────────────────
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

    # ── Patient Demographics ──────────────────────────────────────────────────
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

    # ── Bacterial Context ─────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;font-weight:600;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#6A8BA8;padding:12px 0 7px;">'
        'Bacterial Context</div>',
        unsafe_allow_html=True,
    )

    species  = st.sidebar.text_input(
        "Bacterial species", placeholder="e.g. Escherichia coli", key="species"
    )
    location = st.sidebar.text_input(
        "Sample location", placeholder="e.g. ICU, Urology, Outpatient", key="location"
    )

    st.sidebar.divider()

    input_data = {
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

    return input_data, api_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # App header
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

    input_data, api_ok = render_sidebar()

    analyze_clicked = st.sidebar.button(
        "Analyze Resistance",
        use_container_width=True,
        disabled=not api_ok,
        type="primary",
    )

    if not api_ok:
        st.sidebar.markdown(
            '<p style="font-size:0.74rem;color:#3A5870;text-align:center;'
            'font-family:\'Source Sans 3\',sans-serif;">Start the backend to enable analysis.</p>',
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        with st.spinner("Running ML inference + Groq clinical analysis..."):
            result = call_recommend_api(input_data)
            if result:
                st.session_state.result = result

    if st.session_state.result:
        render_results(st.session_state.result)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
