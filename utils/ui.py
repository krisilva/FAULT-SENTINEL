"""
Fault Sentinel — UI helpers
Loads assets/style.css, exposes the colour palette, and small reusable
HTML-snippet helpers (metric cards, section headers, sidebar branding)
used across app.py and every page in pages/.
"""

import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ----------------------------------------------------------------------
# Palette (keep in sync with assets/style.css)
# ----------------------------------------------------------------------
NAVY        = "#0B1F3A"
NAVY_LIGHT  = "#132A4E"
WHITE       = "#FFFFFF"
OFFWHITE    = "#F2F5FA"
BLACK       = "#0A0A0A"
GREEN       = "#1FB56E"
GREEN_SOFT  = "#E6F8EF"
ORANGE      = "#FF6A2B"
ORANGE_SOFT = "#FFE8DB"
RED         = "#E63946"
RED_SOFT    = "#FDE7E9"
GREY_TEXT   = "#6B7A99"

CHART_SEQUENCE = [ORANGE, NAVY, GREEN, RED, "#F4A340", "#3F6FA8"]


def inject_css():
    """Loads assets/style.css and injects it once per page render."""
    css_path = os.path.join(ASSETS_DIR, "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_branding():
    """Logo + title block shown at the top of the sidebar on every page."""
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        with c1:
            if os.path.exists(logo_path):
                st.image(logo_path, width=44)
        with c2:
            st.markdown(
                """
                <div style="padding-top:2px;">
                    <div style="font-weight:700;font-size:1.05rem;color:#FFFFFF;line-height:1.1;">Fault Sentinel</div>
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.6);">Distribution Network ML Suite</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("<hr>", unsafe_allow_html=True)


def sidebar_footer():
    """Small footer note shown at the bottom of the sidebar."""
    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);line-height:1.6;padding:0 4px;">
            Final Year Research Project<br>
            NSBM Green University<br>
            Modules: Detection (XGBoost) · Classification (Random Forest) ·
            Prediction (LSTM + XGBoost Hybrid)
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar_account_box(username: str):
    """Shows who's logged in + a Logout button. Call on every protected page."""
    from utils.auth import log_out_session

    with st.sidebar:
        st.markdown(
            f"""
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.85);
                        padding:8px 4px 2px 4px;display:flex;align-items:center;gap:6px;">
                🔒 Logged in as <b>{username}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚪 Logout", key="sidebar_logout_btn", use_container_width=True):
            log_out_session()
            st.switch_page("app.py")
        st.markdown("<hr>", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", pill_class: str = "fs-pill-navy"):
    st.markdown(
        f"""
        <div class="fs-card">
            <div class="fs-metric-label">{label}</div>
            <div class="fs-metric-value">{value}</div>
            <span class="fs-pill {pill_class}">{sub}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="fs-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="fs-subtitle">{subtitle}</div>', unsafe_allow_html=True)
