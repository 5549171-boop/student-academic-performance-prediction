"""
app.py
------
Streamlit UI: Student Academic Performance Prediction &
Social Media Impact Analysis System.

UI/UX redesign only. The prediction logic, model loading, feature
handling, and recommendation logic are unchanged from the validated
prototype -- only how they are presented has changed.

Run with:
    streamlit run app.py

Flow: Home -> Predict -> My Analysis -> About Model
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from recommendations import analyze_responses

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_pipeline_v2.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "model", "model_metrics_v2.json")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "model", "feature_schema_v2.json")

st.set_page_config(
    page_title="Academic AI — CGPA Prediction & Social Media Impact",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------
# Loading helpers
# NOTE: no automatic retraining and no subprocess calls here on purpose.
# If the model artifacts are missing, the app tells the user to run
# `python src/train_model.py` themselves rather than triggering
# training as a side effect of opening the app.
# ---------------------------------------------------------------------
def check_model_artifacts_exist():
    missing = [
        p
        for p in [MODEL_PATH, METRICS_PATH, SCHEMA_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        st.error(
            "Model artifacts not found:\n\n"
            + "\n".join(f"- `{os.path.relpath(p, PROJECT_ROOT)}`" for p in missing)
            + "\n\nRun `python src/train_model.py` from the project root first, "
            "then restart the app."
        )
        st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


check_model_artifacts_exist()
model = load_model()
metrics = load_metrics()
schema = load_schema()


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "nav" not in st.session_state:
    st.session_state.nav = "Home"


def go_to(page_name: str):
    st.session_state.nav = page_name
    st.rerun()


# ---------------------------------------------------------------------
# Global styling (visual only — no widget behavior changes here)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    .hero {
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #eef4ff 0%, #f7fafc 100%);
        border: 1px solid #dce6f5;
        margin-bottom: 1.75rem;
    }
    .hero h1 { font-size: 2.1rem; margin-bottom: 0.1rem; color: #10233f; }
    .hero h3 { font-weight: 500; color: #3a5a8c; margin-top: 0; margin-bottom: 1rem; }
    .hero p { font-size: 1.02rem; color: #33415c; max-width: 760px; }

    .info-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        background: #ffffff;
        height: 100%;
    }
    .info-card .icon { font-size: 1.6rem; margin-bottom: 0.35rem; }
    .info-card .title { font-weight: 600; color: #10233f; margin-bottom: 0.15rem; }
    .info-card .desc { font-size: 0.88rem; color: #5b6b82; }

    .step-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.1rem;
        background: #fbfdff;
        height: 100%;
    }
    .step-card .num { font-size: 1.4rem; font-weight: 700; color: #4a7dd6; }
    .step-card .title { font-weight: 600; color: #10233f; margin: 0.2rem 0; }
    .step-card .desc { font-size: 0.87rem; color: #5b6b82; }

    .result-card {
        text-align: center;
        padding: 2.2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #eaf6ef 0%, #f4fbf6 100%);
        border: 1px solid #cdeedb;
        margin: 1rem 0 1.4rem 0;
    }
    .result-card .label { letter-spacing: 0.08em; font-size: 0.8rem; color: #2f7a4f; font-weight: 600; }
    .result-card .value { font-size: 3.4rem; font-weight: 700; color: #1c5c37; line-height: 1.1; margin: 0.2rem 0; }
    .result-card .sub { color: #3d6b4f; font-size: 0.9rem; }

    .rec-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        background: #ffffff;
        margin-bottom: 0.8rem;
    }
    .rec-card .head { font-weight: 600; color: #10233f; font-size: 1.02rem; margin-bottom: 0.3rem; }
    .rec-card .obs { color: #45526b; font-size: 0.92rem; margin-bottom: 0.4rem; }
    .rec-card .sugg { color: #1c5c37; font-size: 0.92rem; }

    .flow-step {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        background: #fbfdff;
        text-align: center;
        font-weight: 600;
        color: #10233f;
        font-size: 0.9rem;
    }
    .flow-arrow { text-align: center; color: #9aa8bd; font-size: 1.1rem; }

    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown("### 🎓 Academic AI")
st.sidebar.caption("Student Performance & Social Media Analysis")
st.sidebar.markdown("---")

NAV_OPTIONS = ["🏠 Home", "📊 Predict", "💡 My Analysis", "🤖 About Model"]
NAV_KEYS = ["Home", "Predict", "My Analysis", "About Model"]
current_index = NAV_KEYS.index(st.session_state.nav)

selected_label = st.sidebar.radio(
    "Navigate", NAV_OPTIONS, index=current_index, label_visibility="collapsed"
)
st.session_state.nav = NAV_KEYS[NAV_OPTIONS.index(selected_label)]
page = st.session_state.nav

st.sidebar.markdown("---")
st.sidebar.caption(
    "This prototype is trained on a limited questionnaire dataset "
    f"({metrics['n_samples']} responses). Predictions reflect patterns in "
    "that data and are not definitive academic forecasts."
)
st.sidebar.caption("B.Sc. AIML Field Project")


# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>🎓 Student Academic Performance Prediction</h1>
            <h3>Social Media Impact Analysis System</h3>
            <p>Use your academic and social-media study habits to estimate your current CGPA
            and identify small areas where your study routine could improve. This tool does
            not claim that social media causes changes in academic performance — it highlights
            patterns observed in survey data as a starting point for reflection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start Assessment →", type="primary"):
        go_to("Predict")

    st.info("**New here?** Start with **Predict** → get your estimate → open **My Analysis**.")

    st.markdown("#### How it works")
    steps = [
        ("01", "Enter Your Details", "Provide your basic academic and study information."),
        ("02", "Get Your Prediction", "The trained Linear Regression model estimates your CGPA."),
        ("03", "Understand Your Habits", "Review social-media and study-related patterns identified from your responses."),
        ("04", "Make Small Improvements", "Use the personalized suggestions and reassess later."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""<div class="step-card">
                    <div class="num">{num}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### At a glance")
    info_cards = [
        ("🤖", "AI Prediction", "Linear Regression model"),
        ("📚", "Academic Focus", "CGPA prediction"),
        ("📱", "Habit Analysis", "Social-media & study behavior"),
        ("💡", "Actionable Insights", "Small practical suggestions"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, info_cards):
        with col:
            st.markdown(
                f"""<div class="info-card">
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.warning(
        "This prototype is trained on a limited questionnaire dataset "
        f"(**{metrics['n_samples']} responses**). Its predictions represent patterns "
        "observed in the collected responses and should not be treated as definitive "
        "academic forecasts. See **About Model** for full evaluation metrics."
    )


# ---------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------
elif page == "Predict":
    st.markdown("## 📊 Predict Your Academic Performance")
    st.write(
        "Complete the assessment below. Your responses are processed by the trained "
        "model to generate an estimated CGPA."
    )

    with st.form("predict_form"):
        st.caption("Step 1 of 2 — Student Information")
        st.progress(0.5)
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "Age",
                    min_value=15,
                    max_value=30,
                    value=int(round(schema["numeric_ranges"]["age"]["median"])),
                )
            with col2:
                year_of_study = st.selectbox("Year of study", options=[1, 2, 3, 4], index=0)

        st.write("")
        st.caption("Step 2 of 2 — Social Media & Study Habits")
        st.progress(1.0)

        with st.container(border=True):
            st.markdown("**📱 Social Media Usage**")
            platforms = st.multiselect(
                "Which social media platforms do you use regularly?",
                options=schema["multi_label_options"]["platforms"],
            )
            purpose = st.multiselect(
                "Purpose of social media use",
                options=schema["multi_label_options"]["purpose"],
            )
            daily_time_hours = st.slider(
                "Average daily time spent on social media (hours)",
                min_value=0.0,
                max_value=12.0,
                value=float(schema["numeric_ranges"]["daily_time_hours"]["median"]),
                step=0.5,
            )
            usage_timing = st.selectbox(
                "When do you mostly use social media?",
                options=schema["categorical_options"]["usage_timing"],
            )

        st.write("")
        with st.container(border=True):
            st.markdown("**📚 Study Habits**")
            multitasking = st.selectbox(
                "Do you multitask with social media while studying or completing assignments?",
                options=schema["categorical_options"]["multitasking"],
            )
            use_during_study_hours = st.selectbox(
                "Do you use social media during study hours?",
                options=schema["categorical_options"]["use_during_study_hours"],
            )
            study_time_frequency = st.slider(
                "On a scale of 1-10, how often do you use social media during study time?",
                min_value=0,
                max_value=10,
                value=int(schema["numeric_ranges"]["study_time_frequency"]["median"]),
            )
            concentration_effect = st.selectbox(
                "Effect of social media on your concentration",
                options=schema["categorical_options"]["concentration_effect"],
            )
            academic_purpose_use = st.selectbox(
                "Do you use social media for academic purposes (study groups, resource sharing, "
                "consultations with professors)?",
                options=schema["categorical_options"]["academic_purpose_use"],
            )
            missed_deadlines = st.selectbox(
                "Has social media access ever led to missed deadlines or incomplete assignments?",
                options=schema["categorical_options"]["missed_deadlines"],
            )

        st.write("")
        submitted = st.form_submit_button("🔮 Predict My CGPA", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "age": age,
            "year_of_study": year_of_study,
            "daily_time_hours": daily_time_hours,
            "study_time_frequency": study_time_frequency,
            "usage_timing": usage_timing,
            "multitasking": multitasking,
            "use_during_study_hours": use_during_study_hours,
            "concentration_effect": concentration_effect,
            "academic_purpose_use": academic_purpose_use,
            "missed_deadlines": missed_deadlines,
        }

        # Build the exact engineered feature columns used by V2.
        for platform in schema["multi_label_options"]["platforms"]:
            inputs[f"platform_{platform}"] = 1 if platform in platforms else 0

        for purpose_item in schema["multi_label_options"]["purpose"]:
            inputs[f"purpose_{purpose_item}"] = 1 if purpose_item in purpose else 0

        inputs["platform_count"] = len(platforms)
        inputs["purpose_count"] = len(purpose)

        X_new = pd.DataFrame([inputs])

        # Keep the input order exactly aligned with the trained V2 model.
        if hasattr(model, "feature_names_in_"):
            X_new = X_new.reindex(columns=model.feature_names_in_)

        prediction = float(model.predict(X_new)[0])
        prediction = max(0.0, min(10.0, prediction))

        st.session_state.last_inputs = inputs
        st.session_state.last_prediction = prediction

    if st.session_state.last_prediction is not None:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">YOUR PREDICTION</div>
                <div class="value">{st.session_state.last_prediction:.2f}</div>
                <div class="sub">Predicted CGPA</div>
                <div class="sub" style="margin-top:0.6rem;">
                    Based on the information you provided. This is a data-driven estimate
                    based on patterns learned from the survey dataset. It is not a
                    guaranteed future grade.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("💡 View My Analysis →", type="primary"):
            go_to("My Analysis")


# ---------------------------------------------------------------------
# MY ANALYSIS
# ---------------------------------------------------------------------
elif page == "My Analysis":
    st.markdown("## 💡 My Analysis")
    st.write("Review the study and social-media habits identified from your responses.")

    if st.session_state.last_inputs is None:
        st.info("Complete the questionnaire on the **Predict** page first to see your analysis.")
        if st.button("📊 Go to Predict →", type="primary"):
            go_to("Predict")
    else:
        flags = analyze_responses(st.session_state.last_inputs)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.last_prediction is not None:
                st.metric("Predicted CGPA", f"{st.session_state.last_prediction:.2f}")
        with col2:
            st.metric("Areas Flagged", len(flags))
        with col3:
            st.metric("Key Habit Identified", flags[0]["area"] if flags else "None flagged")

        st.write("")

        ICONS = {
            "Daily social-media time": "📱",
            "Social media during study time": "🎯",
            "Multitasking while studying": "🔀",
            "Use during study hours": "⏱️",
            "Effect on concentration": "🧠",
            "Missed deadlines": "📅",
            "Academic use of social media": "🎓",
        }

        if not flags:
            st.success(
                "Nothing in your responses stood out as a habit to flag right now. "
                "Keep up your current study approach."
            )
        else:
            st.write(
                f"Based on your responses, **{len(flags)}** area(s) may be worth a small "
                "adjustment. These are suggestions, not requirements."
            )
            for flag in flags:
                icon = ICONS.get(flag["area"], "🔎")
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="head">{icon} {flag['area']}</div>
                        <div class="obs">{flag['observation']}</div>
                        <div class="sugg"><strong>Suggestion:</strong> {flag['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.caption(
            "These insights describe observed patterns in the survey data. They do not "
            "establish that social media directly causes changes in academic performance."
        )


# ---------------------------------------------------------------------
# ABOUT MODEL
# ---------------------------------------------------------------------
elif page == "About Model":
    st.markdown("## 🤖 About the Model")

    st.markdown("#### Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Responses", metrics["n_samples"])
    c2.metric("Training", metrics["n_train"])
    c3.metric("Testing", metrics["n_test"])
    c4.metric("Target", "CGPA")
    c5.metric("Algorithm", "Linear Regression")
    st.caption("Source: Google Form responses from FY B.Sc. AIML students.")

    st.write("")
    st.markdown("#### Model Pipeline")
    flow = ["Student Responses", "Data Preprocessing", "Feature Encoding", "Linear Regression", "Predicted CGPA"]
    for i, step in enumerate(flow):
        st.markdown(f'<div class="flow-step">{step}</div>', unsafe_allow_html=True)
        if i < len(flow) - 1:
            st.markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)
    st.caption(
        "Numeric fields: median imputation + standard scaling. Single-choice categorical "
        "fields: most-frequent imputation + one-hot encoding. Multi-select fields "
        "(platforms, purpose): custom multi-label one-hot encoding."
    )

    st.write("")
    st.markdown("#### Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", metrics["test_mae"])
    m2.metric("Test RMSE", metrics["test_rmse"])
    m3.metric("Test R²", metrics["test_r2"])
    m4, m5 = st.columns(2)
    m4.metric("5-fold CV R² (mean)", metrics["cv_r2_mean"], help=f"std: {metrics['cv_r2_std']}")
    m5.metric("5-fold CV MAE (mean)", metrics["cv_mae_mean"], help=f"std: {metrics['cv_mae_std']}")

    if metrics["test_r2"] < 0 or metrics["cv_r2_mean"] < 0:
        st.warning(
            "The test R² is negative, meaning that on this held-out test set the model "
            "performed worse than simply predicting the average CGPA. This result is "
            "retained as part of the project's evaluation rather than hidden.\n\n"
            "Likely reasons: a small dataset, self-reported responses, questionnaire-based "
            "features, and the fact that academic performance depends on many factors this "
            "survey does not capture."
        )

    st.write("")
    col_used, col_excluded = st.columns(2)
    with col_used:
        st.markdown("#### Features Used")
        st.code("\n".join(metrics["features_used"]))
    with col_excluded:
        st.markdown("#### Features Excluded")
        st.markdown(
            "- **Timestamp, Email, Roll No, Name** — personal identifiers\n"
            "- **CGPA_Binary** — reserved for a future classification phase\n"
            "- **'Compared to previous semester...'** — would leak target information\n"
            "- **Self-reported 1-10 academic-impact score** — overlaps conceptually "
            "with the target"
        )

    st.write("")
    st.markdown("#### Limitations")
    with st.expander("Dataset & prediction limitations"):
        st.write(
            "This prototype is trained on a limited questionnaire dataset. Its predictions "
            "represent patterns observed in the collected responses and should not be treated "
            "as definitive academic forecasts. Relationships described anywhere in this app "
            "are associations found in survey data, not proven causes."
        )
    with st.expander("Raw metrics JSON"):
        st.json(metrics)
