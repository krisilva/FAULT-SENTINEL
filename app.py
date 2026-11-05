"""
Fault Sentinel — Streamlit Dashboard
Machine Learning based Fault Detection, Classification and Early Fault
Prediction in Electrical Distribution Networks.

Entry point / Home page — open to everyone, no account needed.
Only the Admin page (pages/6_Admin.py) requires a login, for
viewing/managing usage data and the activity log.

Run with:  streamlit run app.py
"""

import streamlit as st

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header
from utils.model_loader import load_detection_metrics, load_classification_metrics, load_prediction_metrics

st.set_page_config(
    page_title="Fault Sentinel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
sidebar_branding()
sidebar_footer()

st.markdown("## ⚡ Fault Detection, Classification And Early Fault Prediction System")
st.markdown(
    '<div class="fs-subtitle">Machine Learning based Fault Detection, Classification and Early Fault '
    'Prediction in Electrical Distribution Networks — final year research project. Free to use — '
    'no account needed.</div>',
    unsafe_allow_html=True,
)

det_m = load_detection_metrics()
cls_m = load_classification_metrics()
pred_m = load_prediction_metrics()

c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
with c1:
    st.markdown(
        """
        <div class="fs-hero-card">
            <div style="font-size:0.85rem;opacity:0.9;">System</div>
            <h3 style="margin:2px 0 10px 0;">Fault Sentinel</h3>
            <p style="font-size:0.85rem;line-height:1.5;margin:0;">
                Three-stage ML pipeline — <b>Detection → Classification → Early Prediction</b>
                — trained on OpenDSS-simulated IEEE 13-bus feeder data (11kV/400V, Bus 675).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="fs-card"><div class="fs-metric-label">Module 1 · Detection</div>
        <div class="fs-metric-value">{float(det_m['Accuracy'])*100:.2f}%</div>
        <span class="fs-pill fs-pill-orange">XGBoost · Accuracy</span></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="fs-card"><div class="fs-metric-label">Module 2 · Classification</div>
        <div class="fs-metric-value">{float(cls_m['accuracy'])*100:.2f}%</div>
        <span class="fs-pill fs-pill-green">Random Forest · Accuracy</span></div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="fs-card"><div class="fs-metric-label">Module 3 · Prediction</div>
        <div class="fs-metric-value">{float(pred_m['hybrid_accuracy'])*100:.2f}%</div>
        <span class="fs-pill fs-pill-navy">LSTM+XGBoost · Accuracy</span></div>""",
        unsafe_allow_html=True,
    )

st.write("")
section_header(
    "How the pipeline works",
    "Use the sidebar to jump into any module — each page supports manual entry and CSV upload. "
    "Everyone can use Overview / Detection / Classification / Prediction / Model Comparison freely.",
)
st.markdown(
    """
    <div class="fs-card">
        <div class="fs-subtitle" style="margin-top:2px;line-height:1.8;font-size:0.92rem;">
        1️⃣ <b>Detection</b> — nine electrical features are checked by the XGBoost model to flag
        <b>Fault</b> vs <b>Normal</b> in real time.<br>
        2️⃣ <b>Classification</b> — once a fault is flagged, the Random Forest model narrows it down
        to one of <b>11 fault types</b> (HRGF, LL, LLG, LLLG, SGF/LG, Overcurrent, Overload,
        Voltage Sag/Swell, Transformer Abnormalities).<br>
        3️⃣ <b>Early Prediction</b> — a rolling 10-timestep window is passed through a hybrid
        <b>LSTM + XGBoost</b> soft-voting ensemble to warn <b>60 / 30 / 15 minutes</b> before a
        fault fully develops.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    '<div class="fs-subtitle"> Open <b>Overview</b> in the sidebar for sensor feature trend graphs, '
    'or jump straight into a module. The <b>Admin</b> page is for the system administrator only.</div>',
    unsafe_allow_html=True,
)
