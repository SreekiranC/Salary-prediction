"""
Employee Salary Prediction Dashboard (INR)
============================================
Run with:
    streamlit run app.py

Expects the trained model artifact at ./model/salary_model.pkl
(produced by train_model.py).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom CSS — attractive + responsive
#
# IMPORTANT contrast rule followed throughout: every custom block sets
# BOTH background-color and color explicitly. Never rely on Streamlit's
# inherited/theme default text color on a custom background — the
# default color flips between light and dark depending on the
# viewer's OS/browser theme, which is exactly what caused the
# invisible ("white box, invisible text") bug before.
# ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0f1220; }

    .hero {
        background: linear-gradient(135deg, #6a5cff 0%, #8a5cff 40%, #ff5ca8 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(106, 92, 255, 0.25);
    }
    .hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .3rem 0; color: #ffffff; }
    .hero p { font-size: 1rem; opacity: .9; margin: 0; color: #ffffff; }

    .card {
        background: #ffffff;
        color: #1c1f33;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #eef0f5;
        margin-bottom: 1rem;
    }
    .card h4, .card p, .card strong { color: #1c1f33; }

    .result-card {
        background: linear-gradient(135deg, #1f2440 0%, #2c2f55 100%);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 26px rgba(31,36,64,0.35);
    }
    .result-card .label { font-size: .95rem; opacity: .8; letter-spacing: .04em; text-transform: uppercase; color: #d8d8f5; }
    .result-card .amount { font-size: 3rem; font-weight: 800; margin: .3rem 0; color: #ffffff; }
    .result-card .range { font-size: .9rem; opacity: .85; color: #d8d8f5; }

    .metric-pill {
        display: inline-block;
        background: #f4f2ff;
        color: #5b3df0;
        padding: .35rem .9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: .85rem;
        margin: .2rem .3rem .2rem 0;
    }

    /* Sidebar: dark background. Only headings/labels/markdown text are
       recolored light — the widget boxes themselves (selects, sliders,
       number inputs) keep their own theme-managed contrast, so their
       inner value text is never forced to match the dark backdrop. */
    section[data-testid="stSidebar"] {
        background: #161a2e;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #f0f0f5 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #a9adc7 !important;
    }
    /* Sidebar dropdown text */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {
    color: #f0f0f5 !important;
}

section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #f0f0f5 !important;
}

/* Selected value inside dropdown */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    color: #f0f0f5 !important;
    -webkit-text-fill-color: #f0f0f5 !important;
}

/* Dropdown menu and its options */
[data-baseweb="popover"] {
    color: #f0f0f5 !important;
}

[data-baseweb="popover"] * {
    color: #f0f0f5 !important;
    -webkit-text-fill-color: #f0f0f5 !important;
}

/* Individual dropdown options */
[role="option"] {
    color: #f0f0f5 !important;
}

[role="option"] * {
    color: #f0f0f5 !important;
    -webkit-text-fill-color: #f0f0f5 !important;
}
[data-testid="stMetricLabel"] {
    color: #f0f0f5 !important;
}

[data-testid="stMetricValue"] {
    color: #f0f0f5 !important;
}

[data-testid="stMetricDelta"] {
    color: #a9adc7 !important;
}

    /* Metrics placed directly on the dark app background need explicit
       light text too, since Streamlit's default metric styling assumes
       a light page background. */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #1c1f33 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6a5cff, #ff5ca8);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: .6rem 1.4rem;
        font-weight: 700;
        width: 100%;
        transition: transform .1s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); opacity: .95; }

    .footer-note {
        color: #a9adc7;
        font-size: .85rem;
    }

    @media (max-width: 640px) {
        .hero h1 { font-size: 1.5rem; }
        .result-card .amount { font-size: 2.1rem; }
    }
    /* Main prediction metrics */
[data-testid="stMetricLabel"] {
    color: #f0f0f5 !important;
}

[data-testid="stMetricLabel"] p {
    color: #f0f0f5 !important;
}

[data-testid="stMetricValue"] {
    color: #f0f0f5 !important;
}

[data-testid="stMetricValue"] div {
    color: #f0f0f5 !important;
}

[data-testid="stMetricValue"] * {
    color: #f0f0f5 !important;
}

/* Metric comparison/delta text */
[data-testid="stMetricDelta"] {
    color: #a9adc7 !important;
}

[data-testid="stMetricDelta"] * {
    color: #a9adc7 !important;
}

/* Main page headings */
.main h1,
.main h2,
.main h3,
.main h4 {
    color: #f0f0f5 !important;
}

/* Markdown headings/text on the main page */
[data-testid="stMarkdown"] h1,
[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3,
[data-testid="stMarkdown"] h4 {
    color: #f0f0f5 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load model artifact
# ---------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "salary_model.pkl"

@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    art = load_artifact()
except FileNotFoundError:
    st.error(
        "Model file not found. Run `python train_model.py` first to "
        "generate `model/salary_model.pkl`, then restart the app."
    )
    st.stop()

model = art["model"]
feature_cols = art["feature_cols"]
options = art["options"]
avg_by_education = art["avg_by_education"]
avg_by_job_title = art["avg_by_job_title"]
overall_avg = art["overall_avg"]
overall_salaries = np.array(art["overall_salaries"])
test_r2 = art["test_r2"]

EMPLOYMENT_LABELS = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
SIZE_LABELS = {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (250+)"}

def fmt_inr(x):
    return f"₹{x:,.0f}"

# ---------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>💰 Employee Salary Predictor</h1>
    <p>Select a profile on the left and get an instant salary estimate in INR.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Profile")

    experience_years = st.slider(
        "Years of experience",
        min_value=options["experience_min"],
        max_value=options["experience_max"],
        value=min(5, options["experience_max"]),
    )
    education_level = st.selectbox("Education", options["education_level"], index=1)
    job_title = st.selectbox("Job title", options["job_title"])
    employment_type = st.selectbox(
        "Employment type",
        options["employment_type"],
        format_func=lambda x: EMPLOYMENT_LABELS.get(x, x),
    )
    company_size = st.selectbox(
        "Company size",
        options["company_size"],
        format_func=lambda x: SIZE_LABELS.get(x, x),
        index=1,
    )
    remote_ratio = st.select_slider(
        "Remote work ratio (%)",
        options=options["remote_ratio"],
        value=100,
    )

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Salary")
    st.caption(f"Model: Random Forest · Test R² = {test_r2:.2f}")

# ---------------------------------------------------------------
# Build feature row matching training-time encoding
# ---------------------------------------------------------------
def build_feature_row():
    row = {col: 0 for col in feature_cols}
    row["experience_years"] = experience_years
    row["remote_ratio"] = remote_ratio

    for prefix, value in [
        ("education_level_", education_level),
        ("employment_type_", employment_type),
        ("job_title_", job_title),
        ("company_size_", company_size),
    ]:
        col = f"{prefix}{value}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[feature_cols]

# ---------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------
left, right = st.columns([1.1, 1], gap="large")

if predict_clicked or "last_pred" in st.session_state:
    X_input = build_feature_row()
    pred = model.predict(X_input)[0]
    st.session_state["last_pred"] = pred
else:
    pred = None

with left:
    if pred is not None:
        low, high = pred * 0.9, pred * 1.1
        percentile = (overall_salaries < pred).mean() * 100
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Annual Salary</div>
            <div class="amount">{fmt_inr(pred)}</div>
            <div class="range">Typical range: {fmt_inr(low)} – {fmt_inr(high)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <span class="metric-pill">📈 {percentile:.0f}th percentile overall</span>
        <span class="metric-pill">🎓 {education_level}</span>
        <span class="metric-pill">🧑‍💼 {experience_years} yrs experience</span>
        <span class="metric-pill">🏢 {SIZE_LABELS[company_size]}</span>
        <span class="metric-pill">🏠 {remote_ratio}% remote</span>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        role_avg = avg_by_job_title.get(job_title, overall_avg)
        edu_avg = avg_by_education.get(education_level, overall_avg)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall average", fmt_inr(overall_avg))
        c2.metric(f"Avg for {job_title}", fmt_inr(role_avg), f"{pred - role_avg:+,.0f}")
        c3.metric(f"Avg for {education_level}", fmt_inr(edu_avg), f"{pred - edu_avg:+,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
            <h4>👋 Get started</h4>
            <p>Fill in the profile in the sidebar and click <b>Predict Salary</b> to see
            an estimate along with how it compares to similar roles.</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Where this estimate sits in the market")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(overall_salaries, bins=30, color="#c9c3ff", edgecolor="white")
    if pred is not None:
        ax.axvline(pred, color="#ff5ca8", linewidth=2.5, label=f"Prediction: {fmt_inr(pred)}")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("Salary (INR)")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Average salary by education level")
    edu_order = options["education_level"]
    vals = [avg_by_education.get(e, 0) for e in edu_order]
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.2))
    bar_colors = ["#8a5cff" if e != education_level else "#ff5ca8" for e in edu_order]
    ax2.bar(edu_order, vals, color=bar_colors)
    ax2.set_ylabel("Avg salary (INR)")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">Model trained on a synthetic Indian data-science-salary '
    'dataset for demo purposes. Swap in a real dataset by re-running train_model.py against it.</p>',
    unsafe_allow_html=True,
)
