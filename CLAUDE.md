# CLAUDE.md — CodeCure Hackathon Project

> This file governs how Claude assists with this project. Read it fully before responding to any task.

---

## Project Overview

**Hackathon:** CodeCure @ IIT BHU SPIRIT 2026
**Phase:** Phase 1 (Prototype)
**Track Selected:** Track B — Antibiotic Resistance Prediction
**Goal:** Build a hackathon-winning AI healthcare prototype that judges remember

---

## Problem Context

### Core Problem

Antimicrobial resistance (AMR) is a global health emergency — bacteria evolve faster than new antibiotics are developed. Clinicians lack fast, data-driven tools to predict which antibiotics a bacterial strain will resist _before_ prescribing.

### Hidden Pain Points (not stated in problem statement)

- Resistance data is siloed by geography and lab — no unified view
- Doctors make antibiotic choices under time pressure with incomplete resistance profiles
- Existing prediction tools are black boxes — clinicians don't trust what they can't explain
- Multi-drug resistance patterns are rarely visualized in a clinically actionable way
- Feature importance outputs from ML models are not translated into biological meaning

### Target Users

- Clinical microbiologists interpreting lab susceptibility results
- Hospital infection control teams making antibiotic stewardship decisions
- Public health researchers tracking resistance trends

### Gaps in Existing Solutions

- Tools predict resistance but don't recommend alternatives
- No plain-language explanation of _why_ a strain is predicted resistant
- Visualizations exist but are not interactive or clinically interpretable
- No LLM layer to bridge the gap between ML output and clinical decision-making

### What a Winning Solution Looks Like (Judge Perspective)

- Demonstrates clear biological understanding, not just ML accuracy
- Has an interface that feels usable by a non-technical clinician
- Includes at least one feature no other team thought of
- Tells a compelling story: problem → prediction → actionable recommendation
- Shows feature importance in biologically meaningful terms, not just bar charts

---

## Datasets

| Dataset                                             | Type      | Contents                                                       |
| --------------------------------------------------- | --------- | -------------------------------------------------------------- |
| Antimicrobial Resistance Dataset (Mendeley)         | Primary   | Bacterial isolates, antibiotic susceptibility outcomes (R/S/I) |
| Kaggle Multi-Resistance Dataset                     | Secondary | Multi-drug resistance profiles across bacterial strains        |
| CARD (Comprehensive Antibiotic Resistance Database) | Optional  | Resistance genes, annotations, antibiotic associations         |

---

## Key Innovation Angle

> Most teams will build a resistance classifier. This project adds an **LLM-powered clinical decision-support layer** on top: given a bacterial strain's resistance profile, the system recommends alternative antibiotics and explains the reasoning in plain clinical language.

This is the differentiator judges have not seen before.

---

## Behavioral Guidelines

These rules govern how Claude behaves throughout this project. They are not suggestions — follow them strictly in every response.

### Communication Style

- Be **verbose and educational** by default. When making a decision (technical or architectural), explain _why_ that choice was made, not just what was done. The user is a 2nd-year CS student — build their mental model alongside the code.
- Use clear section headers, bullet points, and tables in responses to make information scannable.
- When introducing a new concept (e.g., SMOTE, Random Forest, SHAP values), give a one-line plain-language explanation before using it.
- Never assume domain knowledge in microbiology or clinical medicine — explain biological context when it appears.

### Plan-First Rule (Strict)

- **Before writing any code**, present a written implementation plan and wait for explicit approval.
- The plan must include:
  - What is being built and why
  - Files to be created or modified
  - Key decisions being made (libraries, structure, approach)
  - Any tradeoffs or alternatives considered
- Only proceed to code after the user confirms the plan. If the user says "go ahead" or "yes", that counts as approval.
- This rule applies to all code: new features, bug fixes, refactors, and scripts.

### Scope Discipline

- If a requested feature is outside the MVP scope or risks derailing the hackathon timeline, **warn the user explicitly** before asking whether to proceed. Format the warning clearly:
  > **Scope Warning:** [what the risk is] — should we proceed, defer, or skip?
- Actively flag when work is drifting from MVP into nice-to-have territory. Label features clearly as `[MVP]`, `[Secondary]`, or `[Demo-critical]` during planning.
- When in doubt about priority, ask rather than assume.

### Code Standards

- Backend: **Python + FastAPI** only. No Node, no Django.
- Frontend: **Streamlit** only. No React, no HTML/CSS/JS unless absolutely unavoidable.
- Write clean, readable code with inline comments explaining non-obvious logic.
- **Skip tests** — hackathon timeline does not permit them. Focus on correctness through careful planning instead.
- Use `requirements.txt` for dependency management. Keep it minimal.
- Structure code for readability, not cleverness. A 2nd-year student should be able to follow every line.

### What Claude Does NOT Do

- Does not write code without a confirmed plan.
- Does not add unrequested features, refactors, or "improvements."
- Does not use overly complex abstractions — prefer explicit, readable code.
- Does not skip explaining a decision just because it seems obvious.
- Does not optimize prematurely — get it working first.

---

## Phase Workflow Reference

This project follows a strict 9-phase strategy. Claude must track which phase is active and only work within the current phase unless instructed otherwise.

| Phase | Name                     | Goal                                                                                                          |
| ----- | ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| 1     | Problem Understanding    | Extract core problem, pain points, target users, gaps, and winning criteria from the problem statement        |
| 2     | Idea Generation          | Generate 5 high-impact AI-driven ideas, then rank and select the best one                                     |
| 3     | Product Design           | Define product vision, must-have vs nice-to-have features, innovation angle, and competitive advantage        |
| 4     | System Architecture      | Design hackathon-feasible tech stack, document complete data flow, justify every choice                       |
| 5     | AI Layer Design          | Define model choice, exact AI role, input→processing→output pipeline, prompt strategy, and 1 advanced feature |
| 6     | MVP Planning             | Break scope into MVP / Secondary / Demo-critical tiers. Remove all unnecessary scope. Set build checkpoints   |
| 7     | GitHub Commit Strategy   | Plan 8–12 meaningful commits showing clear project evolution. Define exact messages and milestones            |
| 8     | Differentiation Strategy | Generate 5 ways to stand out, pick top 2, define how to implement each in the build timeline                  |
| 9     | Development              | Build in stages: dev → staging → polished demo. Plan before every coding session                              |

**Rules:**

- Always state which phase is currently active at the start of a response when doing phase work.
- Do not skip phases or jump ahead without user instruction.
- After completing a phase, summarize the output and explicitly ask before moving to the next.

---

## Custom Skills (Slash Commands)

These are project-specific commands. When the user types any of these, Claude executes the defined behavior exactly — no improvisation.

---

### `/analyze-problem`

**Phase:** 1 — Problem Understanding
**Trigger:** User says "analyze the problem" or types `/analyze-problem`
**Claude must:**

1. Re-read the problem context from this CLAUDE.md
2. Output: Core problem (1 sentence), hidden pain points, target users, gaps in existing solutions, what a winning solution looks like from a judge's perspective
3. Ask: "Ready to move to Phase 2 — Idea Generation?"

---

### `/generate-ideas`

**Phase:** 2 — Idea Generation
**Trigger:** User says "generate ideas" or types `/generate-ideas`
**Claude must:**

1. Generate exactly 5 high-impact AI-driven ideas for Track B (AMR)
2. For each idea: problem-solution fit, key innovation, why it stands out, build complexity rating
3. Rank all 5 and recommend the best one with justification
4. Ask: "Do you approve this selection, or would you like a different idea?"

---

### `/design-product`

**Phase:** 3 — Product Design
**Trigger:** User says "design the product" or types `/design-product`
**Requires:** Idea must be selected (Phase 2 complete)
**Claude must:**

1. Write product vision (2 sentences max)
2. List must-have features `[MVP]` and nice-to-have features `[Secondary]`
3. Define the unique innovation angle clearly (the thing judges haven't seen)
4. State real-world impact and competitive advantage
5. Ask for approval before proceeding

---

### `/architect`

**Phase:** 4 — System Architecture
**Trigger:** User says "design the architecture" or types `/architect`
**Claude must:**

1. Produce the full system architecture using the locked tech stack (FastAPI + Streamlit + Groq)
2. Draw the data flow step by step: user input → preprocessing → ML model → LLM layer → output
3. Justify every technology choice in 1 sentence each
4. Show the folder structure (matches the target structure in Tech Stack section)
5. Ask for approval before any code is written

---

### `/plan-ai-layer`

**Phase:** 5 — AI Layer Design
**Trigger:** User says "design the AI layer" or types `/plan-ai-layer`
**Claude must:**

1. Specify model choice (ML classifier + Groq LLM) and why each is used
2. Define exactly what problem each AI component solves that code alone cannot
3. Document the full pipeline: Input → Preprocessing → ML Inference → SHAP → LLM prompt → Output
4. Write out the key prompt engineering strategy for the Groq/LLM layer
5. Include 1 advanced AI feature most teams won't think of
6. Ask for approval before building

---

### `/plan-mvp`

**Phase:** 6 — MVP Planning
**Trigger:** User says "plan the MVP" or types `/plan-mvp`
**Claude must:**

1. List all features in three tiers: `[MVP]`, `[Secondary]`, `[Demo-critical]`
2. Remove anything that doesn't fit a 1–2 day build
3. Define 3–5 build checkpoints with clear completion criteria
4. Flag any scope risks explicitly
5. Ask for approval before development begins

---

### `/commit-plan`

**Phase:** 7 — GitHub Commit Strategy
**Trigger:** User says "plan commits" or types `/commit-plan`
**Claude must:**

1. Generate a plan for 8–12 meaningful commits
2. For each commit: milestone name, files involved, content summary, exact commit message, why it impresses judges, what to do next
3. Ensure commits show clear project evolution (not a single dump)
4. Ask for approval before the user starts committing

---

### `/differentiate`

**Phase:** 8 — Differentiation Strategy
**Trigger:** User says "differentiation strategy" or types `/differentiate`
**Claude must:**

1. Generate 5 concrete ways this project stands out (across: innovation, UX, AI depth, real-world impact)
2. Pick the top 2 and explain how to implement each within the hackathon timeline
3. Ask for approval

---

### `/roadmap`

**Trigger:** User says "show roadmap" or types `/roadmap`
**Claude must:**

1. Print the phase table (Phase 1–9) with status: `[Done]`, `[In Progress]`, or `[Not Started]`
2. State which phase is currently active
3. List the immediate next action item
4. No approval needed — this is a status command

---

### `/scope-check`

**Trigger:** User says "check scope" or types `/scope-check`
**Claude must:**

1. List all features currently planned or in progress
2. Label each as `[MVP]`, `[Secondary]`, `[Demo-critical]`, or `[Out of Scope]`
3. Flag anything that risks exceeding the hackathon timeline
4. Recommend what to cut if behind schedule

---

### `/explain [term]`

**Trigger:** User says "explain [term]" or types `/explain [term]`
**Claude must:**

1. Give a plain-language explanation (1–2 sentences) — what it is, in simple terms
2. Give a technical explanation (2–3 sentences) — how it works under the hood
3. Give a project-specific example — how it applies to this AMR project specifically

- Examples of valid terms: SHAP, XGBoost, SMOTE, AMR, MIC, Random Forest, SMILES, feature importance, ROC curve, Groq, LLM, FastAPI, Streamlit

---

### `/demo-prep`

**Trigger:** User says "prepare demo" or types `/demo-prep`
**Claude must:**

1. Generate a judge-ready demo walkthrough (step by step, under 5 minutes)
2. Define what to say at each screen/step
3. Highlight the innovation angle during the walkthrough
4. List 3 likely judge questions and suggested answers
5. Flag any UI states that must be working before the demo

---

## Tech Stack

All technology choices are locked for this project. Do not suggest alternatives unless there is a hard blocker.

### Stack

| Layer           | Technology               | Reason                                                                                       |
| --------------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| Frontend        | Streamlit                | Pure Python, no JS needed, fast to build, clean UI out of the box                            |
| Backend         | FastAPI (Python)         | Lightweight, fast, easy to structure as an API                                               |
| ML Models       | scikit-learn + XGBoost   | Fast to train, well-documented, interpretable                                                |
| Explainability  | SHAP                     | Explains which bacterial features drive resistance — critical for biological storytelling    |
| LLM Layer       | Groq API (Llama 3.3 70B) | Free tier (14,400 req/day), extremely fast inference, strong reasoning for clinical language |
| Data Processing | pandas + numpy           | Standard, well-understood                                                                    |
| Visualizations  | Plotly                   | Interactive charts that work natively in Streamlit                                           |
| Dependency Mgmt | requirements.txt         | Simple, no poetry or conda overhead                                                          |

### Free API Policy

- **No paid APIs.** All external services must have a usable free tier for the hackathon duration.
- **Groq API** is the LLM provider. Sign up at console.groq.com. Free tier is sufficient.
- If Groq is unavailable, fallback to **Google Gemini Flash API** (also free, 1500 req/day).
- Store all API keys in a `.env` file. Never hardcode them. Add `.env` to `.gitignore`.

### Project Folder Structure (target)

```
codecure-amr/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── predict.py           # ML model inference
│   ├── llm.py               # Groq API clinical decision layer
│   └── utils.py             # Shared helpers
├── ml/
│   ├── train.py             # Model training script
│   ├── preprocess.py        # Data cleaning and feature engineering
│   └── model.pkl            # Saved trained model
├── frontend/
│   └── streamlit_app.py     # Streamlit UI
├── data/
│   └── (datasets go here, not committed to git)
├── .env                     # API keys (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Demo & Submission Rules

### What Judges Evaluate

Based on the hackathon problem statement, judges prioritize:

1. **Biological understanding** — does the solution reflect genuine domain insight, not just ML accuracy?
2. **Analytical reasoning** — are results interpreted meaningfully, not just printed?
3. **Computational approach** — is the AI doing something non-trivial?
4. **Actionable output** — can a real clinician use this?
5. **Code quality and repository** — clean commits, clear README, structured code

### GitHub Commit Rules

- Minimum **8–12 meaningful commits** — judges read commit history
- Every commit must show **forward progress** — no "fixed stuff" or "misc changes"
- Commit messages follow this format: `[phase] short description of what was added and why`
  - Example: `[ml] add SHAP explainability layer to surface resistance gene importance`
- Never commit: datasets, `.env`, `__pycache__`, `.pkl` files over 50MB
- Always commit: `README.md`, `requirements.txt`, `CLAUDE.md`

### README Requirements

The README must include:

- Project name and one-line description
- Problem statement (2–3 sentences)
- The innovation angle (what makes this different)
- Setup instructions (how to run locally in under 5 steps)
- Screenshots or a demo GIF
- Dataset sources with links

### Demo Principles

- Show **end-to-end flow** in under 5 minutes: input → prediction → explanation → recommendation
- The LLM clinical decision layer is the **highlight** — make sure it runs live, not mocked
- SHAP visualizations must be interpretable — label axes clearly, add a one-line caption
- Have a pre-loaded example input ready so the demo never stalls on data entry
- If anything is broken, cut it from the demo — never demo broken features

---

## Project Checklist

Track progress through all phases before submission.

- [ ] Phase 1 — Problem analyzed, pain points extracted
- [ ] Phase 2 — 5 ideas generated, best idea selected
- [ ] Phase 3 — Product design complete, innovation angle defined
- [ ] Phase 4 — System architecture designed and approved
- [ ] Phase 5 — AI layer defined, prompt strategy written
- [ ] Phase 6 — MVP scope locked, build checkpoints set
- [ ] Phase 7 — GitHub commit plan ready (8–12 commits planned)
- [ ] Phase 8 — Top 2 differentiation strategies chosen
- [ ] Project name finalized
- [ ] Development started
- [ ] ML model trained and evaluated
- [ ] LLM clinical decision layer working (Groq API)
- [ ] Streamlit frontend connected end-to-end
- [ ] README written with screenshots
- [ ] GitHub repository clean and presentable
- [ ] Demo rehearsed at least once

---
