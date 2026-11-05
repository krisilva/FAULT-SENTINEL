import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header, ORANGE, NAVY, GREEN, RED
from utils.model_loader import load_prediction_bundle, load_prediction_metrics, ASSETS_DIR
from utils.preprocessing import PREDICTION_FEATURES, WINDOW, hybrid_predict_window, severity, SEVERITY_TEXT
from utils.activity_log import log_activity

st.set_page_config(page_title="Early Prediction — Fault Sentinel", page_icon="⚡", layout="wide")
inject_css()
sidebar_branding()
sidebar_footer()

# Default window row = a real "No_Imminent_Fault" sample from the training data.
DEFAULT_ROW = {"Va": 6.4944, "Vb": 6.5114, "Vc": 6.4887, "Ia": 67.66, "Ib": 25.83,
               "Ic": 40.44, "Active_Power": 3166.89, "Reactive_Power": 1213.72, "Frequency": 50.01}

SEVERITY_COLOR = {0: GREEN, 1: "#F4A340", 2: ORANGE, 3: RED}


def _result_banner(label, prob):
    sev = severity(label)
    color = SEVERITY_COLOR[sev]
    box_class = "fs-result-normal" if sev == 0 else ("fs-result-warn" if sev < 3 else "fs-result-fault")
    st.markdown(
        f"""<div class="{box_class}">
        <span class="fs-pill" style="background:{color}22;color:{color};">{SEVERITY_TEXT[sev]}</span>
        <div class="fs-metric-value" style="margin-top:8px;">{label.replace('_', ' ')}</div>
        <div class="fs-subtitle">Ensemble confidence: {prob*100:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )


st.markdown("## ⚡ Module 3 — Early Fault Prediction")
st.markdown(
    '<div class="fs-subtitle">Hybrid LSTM + XGBoost soft-voting ensemble — reads a rolling 10-timestep '
    'window of the 9 electrical features and warns 60 / 30 / 15 minutes before a fault fully develops.</div>',
    unsafe_allow_html=True,
)

pm = load_prediction_metrics()
mc1, mc2, mc3, mc4 = st.columns(4)
for col, (label, key) in zip(
    [mc1, mc2, mc3, mc4],
    [("Hybrid Accuracy", "hybrid_accuracy"), ("Hybrid F1 (wtd)", "hybrid_f1_weighted"),
     ("LSTM weight", "lstm_weight"), ("XGBoost weight", "xgb_weight")],
):
    with col:
        val = float(pm[key])
        display = f"{val*100:.2f}%" if key not in ("lstm_weight", "xgb_weight") else f"{val:.1f}"
        st.markdown(
            f"""<div class="fs-card" style="padding:14px 16px;">
            <div class="fs-metric-label">{label}</div>
            <div class="fs-metric-value" style="font-size:1.3rem;">{display}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")
xgb_model, lstm_model, scaler, label_encoder = load_prediction_bundle()
lstm_w, xgb_w = float(pm["lstm_weight"]), float(pm["xgb_weight"])

tab_manual, tab_csv, tab_insights = st.tabs(["✍️ Manual Window Entry", "📁 Upload Episode CSV", "📊 Model Insights"])

# -------------------- Manual entry (10-step window) --------------------
with tab_manual:
    section_header(
        "Enter the last 10 timesteps",
        "Fill in the 9 features for each of the last 10 readings (oldest → newest), then predict "
        "the current early-warning tier.",
    )
    default_df = pd.DataFrame([DEFAULT_ROW] * WINDOW)
    default_df.insert(0, "Timestep", range(1, WINDOW + 1))
    edited = st.data_editor(
        default_df, num_rows="fixed", use_container_width=True, hide_index=True,
        key="pred_manual_editor",
    )

    if st.button("🔍 Predict Early-Warning Tier", key="pred_predict_btn"):
        try:
            window_df = edited[PREDICTION_FEATURES].astype(float)
            label, prob, hybrid_probs, classes = hybrid_predict_window(
                window_df, xgb_model, lstm_model, scaler, label_encoder, lstm_w, xgb_w
            )
            _result_banner(label, prob)

            st.write("")
            order = np.argsort(hybrid_probs)[::-1]
            fig = go.Figure(go.Bar(
                x=[hybrid_probs[i] for i in order],
                y=[str(classes[i]).replace("_", " ") for i in order],
                orientation="h",
                marker=dict(color=[SEVERITY_COLOR[severity(str(classes[i]))] for i in order]),
                text=[f"{hybrid_probs[i]*100:.1f}%" for i in order], textposition="outside",
            ))
            fig.update_layout(
                height=360, template="plotly_white", margin=dict(l=10, r=30, t=20, b=10),
                xaxis_title="Probability", yaxis=dict(autorange="reversed"),
                font=dict(family="Poppins"),
            )
            st.plotly_chart(fig, use_container_width=True)

            log_activity(
                module="Prediction", mode="Manual",
                input_summary="10-step manual window",
                result=label, confidence=prob, rows_processed=1,
            )
        except Exception as e:
            st.error(f"Could not run prediction: {e}")

# -------------------- CSV upload (full episode) --------------------
with tab_csv:
    section_header(
        "Upload an episode CSV",
        f"Upload a CSV with at least {WINDOW} consecutive rows containing: {', '.join(PREDICTION_FEATURES)} "
        "(optionally Episode_ID / Timestep — the app will sort by Timestep if present).",
    )
    file = st.file_uploader("Drop your CSV file here", type=["csv"], key="pred_csv")
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = [c for c in PREDICTION_FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif len(data) < WINDOW:
                st.error(f"Need at least {WINDOW} rows to build one prediction window — got {len(data)}.")
            else:
                if "Episode_ID" in data.columns and data["Episode_ID"].nunique() > 1:
                    first_ep = data["Episode_ID"].iloc[0]
                    data = data[data["Episode_ID"] == first_ep].reset_index(drop=True)
                    st.info(f"Multiple episodes detected — showing Episode_ID = {first_ep} only "
                            f"({len(data)} rows). Upload a single episode for other results.")
                if "Timestep" in data.columns:
                    data = data.sort_values("Timestep").reset_index(drop=True)

                MAX_ROWS = 300
                if len(data) > MAX_ROWS:
                    data = data.iloc[:MAX_ROWS].reset_index(drop=True)
                    st.info(f"Episode truncated to the first {MAX_ROWS} rows to keep predictions fast.")

                feats = data[PREDICTION_FEATURES].values
                n = len(data)
                timeline_labels, timeline_sev, timeline_t, timeline_prob = [], [], [], []

                progress = st.progress(0.0, text="Running rolling-window predictions…")
                for t in range(WINDOW - 1, n):
                    window = pd.DataFrame(feats[t - WINDOW + 1: t + 1], columns=PREDICTION_FEATURES)
                    label, prob, _, _ = hybrid_predict_window(
                        window, xgb_model, lstm_model, scaler, label_encoder, lstm_w, xgb_w
                    )
                    timeline_labels.append(label)
                    timeline_sev.append(severity(label))
                    timeline_t.append(t)
                    timeline_prob.append(prob)
                    progress.progress((t - WINDOW + 2) / (n - WINDOW + 1))
                progress.empty()

                final_label, final_prob = timeline_labels[-1], timeline_prob[-1]
                _result_banner(final_label, final_prob)
                st.markdown(
                    f'<div class="fs-subtitle">Current tier reflects the most recent 10-step window '
                    f'(row {timeline_t[-1]+1} of {n}).</div>', unsafe_allow_html=True
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timeline_t, y=timeline_sev, mode="lines+markers",
                    line=dict(color=NAVY, width=2),
                    marker=dict(size=7, color=[SEVERITY_COLOR[s] for s in timeline_sev]),
                    hovertext=[l.replace("_", " ") for l in timeline_labels],
                    hovertemplate="%{hovertext}<br>Sample: %{x}<extra></extra>",
                ))
                fig.update_layout(
                    title="Predicted risk tier across the episode",
                    yaxis=dict(tickmode="array", tickvals=[0, 1, 2, 3],
                               ticktext=["Normal", "60 min", "30 min", "15 min"]),
                    xaxis_title="Time (sample index)",
                    height=340, template="plotly_white", margin=dict(l=10, r=10, t=45, b=10),
                    font=dict(family="Poppins"),
                )
                st.plotly_chart(fig, use_container_width=True)

                result = data.iloc[WINDOW - 1:].copy().reset_index(drop=True)
                result["Predicted_Tier"] = timeline_labels
                st.dataframe(result, use_container_width=True, height=300)

                st.download_button(
                    "⬇ Download results as CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="prediction_results.csv", mime="text/csv",
                )

                file_sig = (file.name, file.size)
                if st.session_state.get("pred_csv_last_logged") != file_sig:
                    log_activity(
                        module="Prediction", mode="CSV Batch",
                        input_summary=f"{file.name} ({n} rows, episode)",
                        result=f"Final tier: {final_label}", confidence=final_prob,
                        rows_processed=n,
                    )
                    st.session_state["pred_csv_last_logged"] = file_sig
        except Exception as e:
            st.error(f"Could not process file: {e}")

# -------------------- Model insights --------------------
with tab_insights:
    section_header("Confusion matrix", "Hybrid LSTM + XGBoost model, held-out test episodes.")
    cm_path = os.path.join(ASSETS_DIR, "prediction_confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path), caption="Confusion Matrix — Hybrid Model", use_container_width=True)
