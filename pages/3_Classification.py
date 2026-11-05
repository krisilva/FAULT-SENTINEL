import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header, CHART_SEQUENCE
from utils.model_loader import load_classification_model, load_classification_metrics, ASSETS_DIR
from utils.preprocessing import CLASSIFICATION_FEATURES, predict_classification_row, predict_classification_df
from utils.activity_log import log_activity

st.set_page_config(page_title="Classification — Fault Sentinel", page_icon="⚡", layout="wide")
inject_css()
sidebar_branding()
sidebar_footer()

# Defaults = the average of real "Normal" samples from the training data.
FEATURE_META = [
    ("Va", "Phase A Voltage", "kV", 6.4913),
    ("Vb", "Phase B Voltage", "kV", 6.5110),
    ("Vc", "Phase C Voltage", "kV", 6.4847),
    ("Ia", "Phase A Current", "A", 70.75),
    ("Ib", "Phase B Current", "A", 25.68),
    ("Ic", "Phase C Current", "A", 42.41),
    ("Active_Power_kW", "Active Power", "kW", 3309.56),
    ("Reactive_Power_kVAR", "Reactive Power", "kVAR", 1304.08),
    ("Frequency_Hz", "Frequency", "Hz", 50.00),
]


st.markdown("## ⚡ Module 2 — Fault Classification")
st.markdown(
    '<div class="fs-subtitle">Random Forest multi-class model — narrows a detected fault down to one '
    'of 11 fault types using the same 9 electrical features.</div>',
    unsafe_allow_html=True,
)

m = load_classification_metrics()
mc1, mc2, mc3, mc4 = st.columns(4)
for col, (label, key) in zip(
    [mc1, mc2, mc3, mc4],
    [("Accuracy", "accuracy"), ("Precision (wtd)", "precision_weighted"),
     ("Recall (wtd)", "recall_weighted"), ("F1-Score (wtd)", "f1_weighted")],
):
    with col:
        st.markdown(
            f"""<div class="fs-card" style="padding:14px 16px;">
            <div class="fs-metric-label">{label}</div>
            <div class="fs-metric-value" style="font-size:1.3rem;">{float(m[key])*100:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")
model = load_classification_model()
tab_manual, tab_csv, tab_insights = st.tabs(["✍️ Manual Entry", "📁 Upload CSV", "📊 Model Insights"])

# -------------------- Manual entry --------------------
with tab_manual:
    section_header("Enter a single reading", "Type in the 9 feature values, then classify the fault type.")
    values = {}
    cols = st.columns(3)
    for i, (key, label, unit, default) in enumerate(FEATURE_META):
        with cols[i % 3]:
            values[key] = st.number_input(
                f"{label} ({key}) — {unit}", value=float(default), key=f"cls_{key}",
            )

    if st.button("🔍 Classify Fault Type", key="cls_predict_btn"):
        top_name, top_prob, table = predict_classification_row(model, values)
        box_class = "fs-result-normal" if top_name == "Normal" else "fs-result-fault"
        st.markdown(
            f"""<div class="{box_class}">
            <span class="fs-pill fs-pill-navy">PREDICTED CLASS</span>
            <div class="fs-metric-value" style="margin-top:8px;">{top_name}</div>
            <div class="fs-subtitle">Confidence: {top_prob*100:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        fig = go.Figure(go.Bar(
            x=table["Probability"], y=table["Fault Type"], orientation="h",
            marker=dict(color=CHART_SEQUENCE * 3),
            text=[f"{v*100:.1f}%" for v in table["Probability"]], textposition="outside",
        ))
        fig.update_layout(
            height=380, template="plotly_white", margin=dict(l=10, r=30, t=20, b=10),
            xaxis_title="Probability", yaxis=dict(autorange="reversed"),
            font=dict(family="Poppins"),
        )
        st.plotly_chart(fig, use_container_width=True)

        log_activity(
            module="Classification", mode="Manual",
            input_summary=f"Va:{values['Va']:.2f}kV Ia:{values['Ia']:.1f}A "
                          f"P:{values['Active_Power_kW']:.0f}kW",
            result=top_name, confidence=top_prob, rows_processed=1,
        )

# -------------------- CSV upload --------------------
with tab_csv:
    section_header(
        "Batch classification from CSV",
        f"Upload a CSV containing the columns: {', '.join(CLASSIFICATION_FEATURES)}",
    )
    file = st.file_uploader("Drop your CSV file here", type=["csv"], key="cls_csv")
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = [c for c in CLASSIFICATION_FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                result = predict_classification_df(model, data)
                st.dataframe(result, use_container_width=True, height=340)

                counts = result["Predicted_Fault_Type"].value_counts()
                fig = go.Figure(go.Bar(
                    x=counts.index, y=counts.values,
                    marker=dict(color=CHART_SEQUENCE * 3),
                ))
                fig.update_layout(
                    height=320, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10),
                    xaxis_title="Fault Type", yaxis_title="Count",
                    font=dict(family="Poppins"),
                    title="Predicted fault type distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇ Download results as CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="classification_results.csv", mime="text/csv",
                )

                file_sig = (file.name, file.size)
                if st.session_state.get("cls_csv_last_logged") != file_sig:
                    top_counts = result["Predicted_Fault_Type"].value_counts()
                    top_summary = ", ".join(f"{k}:{v}" for k, v in top_counts.head(3).items())
                    log_activity(
                        module="Classification", mode="CSV Batch",
                        input_summary=f"{file.name} ({len(result)} rows)",
                        result=top_summary, confidence=None, rows_processed=len(result),
                    )
                    st.session_state["cls_csv_last_logged"] = file_sig
        except Exception as e:
            st.error(f"Could not process file: {e}")

# -------------------- Model insights --------------------
with tab_insights:
    section_header("Confusion matrix & feature importance", "Generated from the held-out test set.")
    ic1, ic2 = st.columns(2)
    cm_path = os.path.join(ASSETS_DIR, "classification_confusion_matrix.png")
    fi_path = os.path.join(ASSETS_DIR, "classification_feature_importance.png")
    if os.path.exists(cm_path):
        with ic1:
            st.image(Image.open(cm_path), caption="Confusion Matrix", use_container_width=True)
    if os.path.exists(fi_path):
        with ic2:
            st.image(Image.open(fi_path), caption="Feature Importance", use_container_width=True)
