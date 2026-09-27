# Handoff Document
## Student Academic Performance Prediction & Social Media Impact Analysis System

Status: **ML prototype validated and locked.** This document exists to move the
project into Google AI Studio for **UI/UX work only**, without disturbing the
working model.

---

## 1. File inventory

| File | Role | Modify for UI redesign? |
|---|---|---|
| `data/dataset.xlsx` | Raw cleaned Google Form dataset (208 rows × 20 cols). Source of truth for training. | **NO** |
| `src/preprocessing.py` | Column mapping, cleaning rules (age/time/frequency validity ranges), the `MultiLabelOneHotEncoder` class, feature/target column lists. | **NO** |
| `src/train_model.py` | Builds the preprocessing + Linear Regression pipeline, evaluates it (train/test split + 5-fold CV), writes the three files in `model/`. | **NO** |
| `src/recommendations.py` | Rule-based `analyze_responses()` function: thresholds, flagged areas, suggestion text. | **NO** (logic/text) — display styling only |
| `model/model_pipeline.joblib` | The trained, fitted scikit-learn pipeline (preprocessing + Linear Regression) — the actual model artifact. | **NO — never regenerate or replace** |
| `model/model_metrics.json` | The one and only valid evaluation result (MAE/RMSE/R², CV scores). | **NO — never overwrite or invent new values** |
| `model/feature_schema.json` | Allowed values/ranges for every input field, generated from the training data. Drives the form's dropdowns/sliders. | **NO** (read from, don't hardcode elsewhere) |
| `app.py` | Streamlit application: 4 pages (Home, Predict, My Analysis, About Model), loads the model + metrics + schema, renders the form, calls `analyze_responses()`. | **YES — this is the file to redesign** |
| `requirements.txt` | Python dependencies. | Only if a redesign adds a genuinely new dependency |
| `README.md` | Setup/run instructions and the "excluded fields" rationale table. | No need to touch for UI work |

No other config files or binary/image assets exist in the project — it's plain Python + one Excel file + three generated JSON/joblib artifacts.

---

## 2. Files that MUST NOT be modified

**Never touch, never regenerate, never let an AI tool "helpfully" rewrite these:**
- `data/dataset.xlsx`
- `src/preprocessing.py`
- `src/train_model.py`
- `model/model_pipeline.joblib`
- `model/model_metrics.json`
- `model/feature_schema.json`

**Touch only for display styling, not logic/text:** `src/recommendations.py`

**The one file meant to change:** `app.py`

Rationale: the model has already been trained and honestly evaluated. Any tool that retrains, re-derives, or hand-edits the artifacts above risks silently producing different numbers than the ones you've verified and documented — which would make your evaluation section inconsistent with your actual code.

---

## 3. Technical summary (for Google AI Studio)

**Project purpose:** Estimate a student's CGPA from a social-media/study-habits
questionnaire (Linear Regression), then give small, rule-based, personalized
improvement suggestions. Philosophy: **Prediction → Awareness → Small
Improvements → Reassessment.** This is a data-driven reflection tool, not a
judgment of intelligence or ability.

**Data flow:**
```
data/dataset.xlsx (208 responses)
      ↓
src/preprocessing.py  — rename columns, clean invalid values, select approved
                         feature columns (drops identifiers + leakage fields)
      ↓
src/train_model.py    — fit ColumnTransformer (numeric/categorical/multi-label)
                         + LinearRegression, evaluate, save 3 artifacts to model/
```

**Model flow (at prediction time, inside app.py):**
```
User answers questionnaire (Streamlit form)
      ↓
Assembled into a 1-row pandas DataFrame, same column names as training
      ↓
model_pipeline.joblib .predict()  (imputation → scaling/encoding → LinearRegression, all inside one fitted pipeline)
      ↓
Predicted CGPA (float, e.g. 7.91)
      ↓
src/recommendations.py .analyze_responses(inputs) — rule-based flags, no ML
      ↓
Flagged areas + suggestions shown to the student
```

**Input fields (12 features, all required):**
- `age` (numeric, 15–30 after cleaning)
- `year_of_study` (1–4)
- `platforms` (multi-select: Instagram, WhatsApp, YouTube, Snapchat, LinkedIn, Reddit, Discord, etc.)
- `purpose` (multi-select: Academic learning, Entertainment, Communication with friends/family, Professional networking)
- `daily_time_hours` (numeric, 0–24 valid range)
- `usage_timing` (single choice: Before/During/While/After studying, Late at night)
- `multitasking` (single choice: Never/Rarely/Sometimes/Often/Always)
- `use_during_study_hours` (yes/no)
- `study_time_frequency` (0–10 scale)
- `concentration_effect` (No effect/Slightly helpful/Very helpful/Moderately distracting/Highly distracting)
- `academic_purpose_use` (never/rarely/sometimes/often/always)
- `missed_deadlines` (Never/Rarely/Occasionally/Frequently)

All exact allowed values live in `model/feature_schema.json` — that file is the single source of truth for form options, not anything hardcoded in a UI redesign.

**Prediction process:** single call to a pre-fitted scikit-learn `Pipeline`
object (`model_pipeline.joblib`). No retraining happens at prediction time.
Unseen category values are safely ignored (`handle_unknown="ignore"` /
custom multi-label logic), so the form never crashes on an unexpected input.

**Analysis/recommendation process:** pure rule-based Python (`if` thresholds
on the same 12 inputs), no ML model involved. Returns a list of
`{area, observation, suggestion}` dicts.

**Evaluation metrics (from `model_metrics.json` — verified, current, correct):**

| Metric | Value |
|---|---|
| n_samples / train / test | 208 / 166 / 42 |
| Test MAE | 1.0553 |
| Test RMSE | 1.3751 |
| Test R² | -0.1956 |
| 5-fold CV R² mean | -0.1249 |
| 5-fold CV MAE mean | 1.1793 |
| random_state | 42 |

**Features used:** age, year_of_study, daily_time_hours, study_time_frequency,
usage_timing, multitasking, use_during_study_hours, concentration_effect,
academic_purpose_use, missed_deadlines, platforms, purpose.

**Deliberately excluded (and why):** Timestamp/Email/Roll No/Name (identifiers,
not predictive); `CGPA_Binary` (reserved for a future Phase 2 classification
model); the "compared to previous semester" field (an academic outcome —
using it to predict CGPA would leak target information); the self-reported
1–10 "how much has social media affected your grades" score (overlaps
conceptually with the target, same leakage risk as the previous-semester field).

**Limitations (must stay visible in any redesign):**
- Small dataset (208 self-reported responses from one institution).
- Negative R² on both the held-out test split and 5-fold CV — the model
  currently performs worse than predicting the mean CGPA for everyone. This
  is reported honestly, not hidden.
- Predictions are estimates/associations, never guaranteed outcomes and never
  causal claims about social media.
- An earlier, invalid model run (MAE 1.30 / RMSE 1.60 / R² -0.61, from a
  version that leaked `Name` and the previous-semester field into training)
  must never be cited as the project's result.

---

## 4. Explicit instructions for Google AI Studio

1. Preserve the existing ML logic and model behavior exactly as implemented.
2. Do not invent, synthesize, or augment training data.
3. Do not change, round away, hide, or "improve" the evaluation metrics —
   including the negative R² values.
4. Do not claim or imply that social media *causes* low CGPA — this is
   association language only, from questionnaire data.
5. Do not introduce fake/sample predictions, fake database records, or
   placeholder data anywhere in the UI to make it "look" more functional.
6. Do not remove or soften the model-limitations disclaimers.
7. UI/UX (layout, styling, navigation, visual presentation of metrics and
   recommendations) can be redesigned substantially.
8. All current functional behavior — the 4-page flow, all 12 input fields,
   the prediction call, the rule-based analysis, and the metrics display —
   must remain available and working after the redesign.

---

## 5. Safest way to recreate the interface in Google AI Studio

Google AI Studio is built around Gemini + web/app generation — it does not
natively execute a local scikit-learn `.joblib` pipeline the way Streamlit
does. Two realistic options, safest first:

**Option A — Keep Streamlit, use AI Studio only as a code-editing assistant.**
Give AI Studio `app.py` (plus the reference files below) and ask it to return
a revised `app.py`. You keep running it with `streamlit run app.py` exactly
as now. This is the lowest-risk path: zero architecture change, the trained
pipeline keeps running in-process exactly as it does today. This is the
option the master prompt below is written for.

**Option B — Separate frontend from the model (if you want a non-Streamlit,
AI-Studio-native web UI).** Google AI Studio is good at generating polished
web frontends but cannot host your Python model directly, so split the app
into three layers:

```
Frontend/UI  (AI Studio-generated web app: HTML/JS or a
              framework AI Studio supports)
      ↓  (HTTP request: JSON of the 12 questionnaire answers)
Prediction API/backend  (a small FastAPI or Flask service you run yourself,
                          e.g. one POST /predict endpoint)
      ↓
Existing preprocessing + Linear Regression model
   (loads model_pipeline.joblib, calls .predict(), returns JSON:
    {predicted_cgpa, flags: [...]})
```

- The backend is a thin wrapper: load `model_pipeline.joblib` once at
  startup, expose `/predict` that takes the same 12 fields as JSON, runs
  `pipeline.predict()`, calls `analyze_responses()` from
  `recommendations.py`, and returns both results as JSON.
- The frontend calls that endpoint and only renders what it gets back — it
  never reimplements prediction or recommendation logic itself.
- This is more work (you must deploy/run the backend somewhere reachable by
  the frontend) but gives you full design freedom in AI Studio.

Recommendation: start with **Option A**. Only move to Option B if you
specifically want a non-Streamlit, production-style web app later — that's
also a natural "Phase 2+" step alongside the database work already planned.

---

## 6. Files to actually give Google AI Studio

Upload only these four:
- `app.py`
- `src/recommendations.py`
- `model/model_metrics.json`
- `model/feature_schema.json`

Do not upload `train_model.py`, `preprocessing.py`, `model_pipeline.joblib`,
or `dataset.xlsx` — AI Studio doesn't need them for UI work, and keeping them
out of the conversation removes any chance of it "fixing" or retraining
something that isn't broken.
