import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header, ORANGE, NAVY, GREEN, RED
from utils.model_loader import (
    load_detection_metrics, load_classification_metrics, load_prediction_metrics,
    load_module3_comparison,
)

st.set_page_config(page_title="Model Comparison — Fault Sentinel", page_icon="📊", layout="wide")
inject_css()
sidebar_branding()
sidebar_footer()

st.markdown("## 📊 Model Comparison")
st.markdown(
    '<div class="fs-subtitle">Final selected model for each module, and how Module 3\'s hybrid '
    'ensemble stacks up against its standalone LSTM and XGBoost components.</div>',
    unsafe_allow_html=True,
)

det_m = load_detection_metrics()
cls_m = load_classification_metrics()
pred_m = load_prediction_metrics()

summary = pd.DataFrame([
    {"Module": "1 · Detection", "Final Model": "XGBoost",
     "Accuracy": float(det_m["Accuracy"]), "Precision": float(det_m["Precision"]),
     "Recall": float(det_m["Recall"]), "F1-Score": float(det_m["F1-Score"])},
    {"Module": "2 · Classification", "Final Model": "Random Forest",
     "Accuracy": float(cls_m["accuracy"]), "Precision": float(cls_m["precision_weighted"]),
     "Recall": float(cls_m["recall_weighted"]), "F1-Score": float(cls_m["f1_weighted"])},
    {"Module": "3 · Early Prediction", "Final Model": "LSTM + XGBoost (Hybrid)",
     "Accuracy": float(pred_m["hybrid_accuracy"]), "Precision": float(pred_m["hybrid_precision_weighted"]),
     "Recall": float(pred_m["hybrid_recall_weighted"]), "F1-Score": float(pred_m["hybrid_f1_weighted"])},
])

section_header("Final model per module")
st.dataframe(
    summary.style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"}),
    use_container_width=True, hide_index=True,
)

fig = go.Figure()
for metric, color in zip(["Accuracy", "Precision", "Recall", "F1-Score"], [ORANGE, NAVY, GREEN, RED]):
    fig.add_trace(go.Bar(name=metric, x=summary["Module"], y=summary[metric], marker_color=color))
fig.update_layout(
    barmode="group", template="plotly_white", height=380,
    yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=20, b=10),
    font=dict(family="Poppins"), legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True)

st.write("")
section_header(
    "Module 3 — Hybrid vs standalone LSTM / XGBoost",
    "Ensemble weight selected on a held-out validation split (episode-level GroupShuffleSplit) "
    "before scoring on the test set, to avoid leakage.",
)
comp = load_module3_comparison()
comp_display = comp.rename_axis("Model").reset_index()
st.dataframe(
    comp_display.style.format({c: "{:.2%}" for c in comp_display.columns if c != "Model"}),
    use_container_width=True, hide_index=True,
)

fig2 = go.Figure()
colors = {"LSTM": NAVY, "XGBoost": ORANGE, "Hybrid": GREEN}
for model_name in comp_display["Model"]:
    row = comp_display[comp_display["Model"] == model_name].iloc[0]
    fig2.add_trace(go.Scatterpolar(
        r=[row["accuracy"], row["precision_weighted"], row["recall_weighted"], row["f1_weighted"], row["roc_auc_weighted"]],
        theta=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        fill="toself", name=model_name, line=dict(color=colors.get(model_name, NAVY)),
    ))
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0.85, 1.0], tickformat=".0%")),
    template="plotly_white", height=420, font=dict(family="Poppins"),
    legend=dict(orientation="h", y=-0.1),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    f"""
    <div class="fs-card">
    <div class="fs-section-title">Why the hybrid ensemble wins</div>
    <div class="fs-subtitle" style="margin-top:6px;line-height:1.7;">
    The hybrid model blends LSTM's sequence-pattern learning with XGBoost's strength on flattened
    tabular windows via <b>weighted soft-voting</b> (LSTM weight {float(pred_m['lstm_weight']):.1f} ·
    XGBoost weight {float(pred_m['xgb_weight']):.1f}). It consistently beats both standalone models
    on accuracy and F1-score, at effectively the same ROC-AUC.
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)
