import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header, ORANGE, GREEN, RED, NAVY
from utils.model_loader import load_detection_model, load_detection_metrics
from utils.preprocessing import DETECTION_FEATURES, predict_detection_row, predict_detection_df
from utils.activity_log import log_activity

st.set_page_config(page_title="Detection — Fault Sentinel", page_icon="⚡", layout="wide")
inject_css()
sidebar_branding()
sidebar_footer()

# Default values = the average of real "Normal" (Label=0) samples from the
# training data, NOT an arbitrary/random row - a fault-labelled default
# would make every fresh page load look like a fault even before the user
# changes anything.
FEATURE_META = [
    ("Va", "Phase A Voltage", "kV", 6.4908, 0.0, 15.0, 0.001),
    ("Vb", "Phase B Voltage", "kV", 6.5110, 0.0, 15.0, 0.001),
    ("Vc", "Phase C Voltage", "kV", 6.4841, 0.0, 15.0, 0.001),
    ("Ia", "Phase A Current", "A", 71.23, 0.0, 500.0, 0.01),
    ("Ib", "Phase B Current", "A", 25.65, 0.0, 500.0, 0.01),
    ("Ic", "Phase C Current", "A", 42.69, 0.0, 500.0, 0.01),
    ("Active_Power_kW", "Active Power", "kW", 3332.89, -5000.0, 20000.0, 0.1),
    ("Reactive_Power_kVAR", "Reactive Power", "kVAR", 1318.78, -5000.0, 20000.0, 0.1),
    ("Frequency_Hz", "Frequency", "Hz", 50.00, 40.0, 60.0, 0.001),
]


def _result_banner(label: str, proba: float):
    if label == "Fault":
        st.markdown(
            f"""<div class="fs-result-fault">
            <span class="fs-pill fs-pill-red">⚠ FAULT DETECTED</span>
            <div class="fs-metric-value" style="margin-top:8px;">Fault probability: {proba*100:.2f}%</div>
            <div class="fs-subtitle">The XGBoost detection model flags this reading as an electrical fault.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="fs-result-normal">
            <span class="fs-pill fs-pill-green">✓ NORMAL</span>
            <div class="fs-metric-value" style="margin-top:8px;">Fault probability: {proba*100:.2f}%</div>
            <div class="fs-subtitle">The reading looks consistent with normal feeder operation.</div>
            </div>""",
            unsafe_allow_html=True,
        )


st.markdown("## ⚡ Module 1 — Fault Detection")
st.markdown(
    '<div class="fs-subtitle">XGBoost binary classifier — flags Fault vs Normal from the 9 electrical '
    'features (Va, Vb, Vc, Ia, Ib, Ic, Active Power, Reactive Power, Frequency).</div>',
    unsafe_allow_html=True,
)

m = load_detection_metrics()
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
for col, (label, key) in zip(
    [mc1, mc2, mc3, mc4, mc5],
    [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
     ("F1-Score", "F1-Score"), ("ROC-AUC", "ROC-AUC")],
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
model = load_detection_model()
tab_manual, tab_csv = st.tabs(["✍️ Manual Entry", "📁 Upload CSV"])

# -------------------- Manual entry --------------------
with tab_manual:
    section_header("Enter a single reading", "Type in the 9 feature values, then run the detector.")
    values = {}
    cols = st.columns(3)
    for i, (key, label, unit, default, lo, hi, step) in enumerate(FEATURE_META):
        with cols[i % 3]:
            values[key] = st.number_input(
                f"{label} ({key}) — {unit}", value=float(default),
                min_value=float(lo), max_value=float(hi), step=float(step),
                key=f"det_{key}",
            )

    if st.button("🔍 Run Detection", key="det_predict_btn"):
        label, proba = predict_detection_row(model, values)
        _result_banner(label, proba)
        log_activity(
            module="Detection", mode="Manual",
            input_summary=f"Va:{values['Va']:.2f}kV Ia:{values['Ia']:.1f}A "
                          f"P:{values['Active_Power_kW']:.0f}kW",
            result=label, confidence=proba, rows_processed=1,
        )

# -------------------- CSV upload --------------------
with tab_csv:
    section_header(
        "Batch detection from CSV",
        f"Upload a CSV containing the columns: {', '.join(DETECTION_FEATURES)}",
    )
    file = st.file_uploader("Drop your CSV file here", type=["csv"], key="det_csv")
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = [c for c in DETECTION_FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                result = predict_detection_df(model, data)
                n_fault = int((result["Predicted_Label"] == "Fault").sum())
                n_normal = len(result) - n_fault

                b1, b2, b3 = st.columns(3)
                b1.markdown(f"""<div class="fs-card"><div class="fs-metric-label">Rows analysed</div>
                    <div class="fs-metric-value">{len(result)}</div></div>""", unsafe_allow_html=True)
                b2.markdown(f"""<div class="fs-card"><div class="fs-metric-label">Flagged as Fault</div>
                    <div class="fs-metric-value" style="color:{RED};">{n_fault}</div></div>""", unsafe_allow_html=True)
                b3.markdown(f"""<div class="fs-card"><div class="fs-metric-label">Normal</div>
                    <div class="fs-metric-value" style="color:{GREEN};">{n_normal}</div></div>""", unsafe_allow_html=True)

                st.write("")
                st.dataframe(result, use_container_width=True, height=340)

                fig = go.Figure(data=[go.Pie(
                    labels=["Fault", "Normal"], values=[n_fault, n_normal],
                    marker=dict(colors=[RED, GREEN]), hole=0.55,
                )])
                fig.update_layout(
                    height=280, margin=dict(l=10, r=10, t=10, b=10),
                    template="plotly_white", showlegend=True,
                    font=dict(family="Poppins"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇ Download results as CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="detection_results.csv", mime="text/csv",
                )

                # Guard against re-logging on every Streamlit rerun (e.g. clicking
                # the download button above re-runs this whole script) - only log
                # once per distinct uploaded file.
                file_sig = (file.name, file.size)
                if st.session_state.get("det_csv_last_logged") != file_sig:
                    log_activity(
                        module="Detection", mode="CSV Batch",
                        input_summary=f"{file.name} ({len(result)} rows)",
                        result=f"{n_fault} Fault / {n_normal} Normal",
                        confidence=None, rows_processed=len(result),
                    )
                    st.session_state["det_csv_last_logged"] = file_sig
        except Exception as e:
            st.error(f"Could not process file: {e}")
