import streamlit as st
import plotly.graph_objects as go

from utils.ui import inject_css, sidebar_branding, sidebar_footer, section_header, NAVY, ORANGE, GREEN, WHITE, GREY_TEXT
from utils.model_loader import load_detection_data

st.set_page_config(page_title="Overview — Fault Sentinel", page_icon="🏠", layout="wide")
inject_css()
sidebar_branding()
sidebar_footer()


def _line_chart(df, cols, colors, title, y_title):
    fig = go.Figure()
    for col, color in zip(cols, colors):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col,
            line=dict(color=color, width=2.4),
            hovertemplate=f"{col}: %{{y:.3f}}<br>Sample: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY, family="Poppins")),
        xaxis_title="Time (sample index)",
        yaxis_title=y_title,
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        plot_bgcolor=WHITE,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins", color=GREY_TEXT, size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(11,31,58,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(11,31,58,0.06)")
    st.plotly_chart(fig, use_container_width=True)


st.markdown("## 🏠 Overview — Sensor Feature Trends")
st.markdown(
    '<div class="fs-subtitle">Each electrical feature plotted one by one from the Detection dataset '
    '(first 100 samples), so you can see how it behaves over time before it feeds into the models.</div>',
    unsafe_allow_html=True,
)

df = load_detection_data().head(100).reset_index(drop=True)

_line_chart(df, ["Va", "Vb", "Vc"], [ORANGE, NAVY, GREEN],
            "1. Voltage — Va, Vb, Vc vs Time", "Voltage (kV)")
_line_chart(df, ["Ia", "Ib", "Ic"], [ORANGE, NAVY, GREEN],
            "2. Current — Ia, Ib, Ic vs Time", "Current (A)")

cc1, cc2 = st.columns(2)
with cc1:
    _line_chart(df, ["Active_Power_kW"], [ORANGE],
                "3. Active Power vs Time", "Active Power (kW)")
with cc2:
    _line_chart(df, ["Reactive_Power_kVAR"], [NAVY],
                "4. Reactive Power vs Time", "Reactive Power (kVAR)")

_line_chart(df, ["Frequency_Hz"], [GREEN],
            "5. Frequency vs Time", "Frequency (Hz)")
