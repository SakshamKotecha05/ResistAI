You are working on the CodeCure hackathon project (Track B — Antibiotic Resistance Prediction).

Execute Phase 5 — AI Layer Design.

Do the following in order:

1. State: "**Phase 5 — AI Layer Design**" as a header

2. **Model Choice Breakdown**:
   - ML component: Why Random Forest + XGBoost for resistance classification? What does each model bring?
   - LLM component: Why Groq (Llama 3.3 70B)? What can LLM do here that pure ML cannot?
   - SHAP: Why is explainability a separate layer and not just model output?

3. **Exact Role of AI** (be precise): What specific problem does each AI component solve that code alone cannot solve?

4. **Full Pipeline** (Input → Processing → Output):
   - Describe each stage with inputs, transformations, and outputs
   - Include data types at each step (e.g., DataFrame → probability score → JSON → natural language)

5. **Prompt Engineering Strategy**:
   - Write the actual system prompt for the Groq/LLM clinical decision layer
   - Explain the reasoning behind each part of the prompt
   - Show an example input/output pair for the LLM call

6. **Advanced AI Feature** (the one most teams won't think of): Describe 1 advanced feature that demonstrates deeper AI thinking. Be specific about how to implement it within the hackathon timeline.

7. **Demo Intelligence**: How does the AI *appear* smart and clinically meaningful during the judge demo? What moments in the UI will impress?

8. Ask: "AI layer approved? Move to Phase 6 — MVP Planning (`/plan-mvp`)?"

Be verbose and educational. Explain ML and LLM concepts clearly. The user is a 2nd-year CS student.
