You are working on the CodeCure hackathon project (Track B — Antibiotic Resistance Prediction).

Execute Phase 4 — System Architecture.

Tech stack is locked:
- Frontend: Streamlit (Python)
- Backend: FastAPI (Python)
- ML: scikit-learn + XGBoost
- Explainability: SHAP
- LLM Layer: Groq API (Llama 3.3 70B) — free tier
- Data: pandas + numpy
- Visualizations: Plotly
- Dependencies: requirements.txt

Do the following in order:

1. State: "**Phase 4 — System Architecture**" as a header

2. **System Overview**: Describe the system in 3–4 sentences. What are its components and how do they relate?

3. **Architecture Diagram** (text-based): Draw a clear ASCII or markdown representation showing all components and how they connect.

4. **Data Flow** (step by step, numbered): Trace the path from user input to final output:
   - Step 1: User submits bacterial strain data via Streamlit
   - Step N: Final output displayed (resistance prediction + LLM recommendation)
   - Include every transformation, API call, and model inference step

5. **Technology Justification Table**: For each technology in the stack, give a 1-sentence reason why it was chosen over alternatives.

6. **Folder Structure**: Show the exact project folder structure using the target layout from CLAUDE.md.

7. **Integration Points**: List every place where two components connect (e.g., Streamlit ↔ FastAPI, FastAPI ↔ Groq API) and what data format is passed.

8. Ask: "Architecture approved? Move to Phase 5 — AI Layer Design (`/plan-ai-layer`)?"

Be verbose and educational. Explain every architectural decision. The user is a 2nd-year CS student learning system design.
