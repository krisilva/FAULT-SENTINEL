import time
import streamlit as st
import plotly.graph_objects as go

from utils.ui import inject_css, sidebar_branding, sidebar_footer, sidebar_account_box, section_header, ORANGE, NAVY, GREEN, RED, CHART_SEQUENCE
from utils.auth import authenticate, is_logged_in, log_in_session, get_last_login, change_password
from utils.activity_log import get_logs, delete_log, clear_all_logs, get_usage_stats

st.set_page_config(page_title="Admin — Fault Sentinel", page_icon="🔐", layout="wide")
inject_css()


def render_login():
    sidebar_branding()
    with st.sidebar:
        st.markdown(
            '<div style="font-size:0.78rem;color:rgba(255,255,255,0.65);padding:0 4px;">'
            'Admin access only. Everyone else can use the other pages freely — no account needed.'
            '</div>',
            unsafe_allow_html=True,
        )

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.write("")
        st.write("")
        st.markdown(
            """
            <div class="fs-card" style="text-align:center;padding:30px 32px 10px 32px;">
                <div style="font-size:2.2rem;">🔐</div>
                <h3 style="margin:6px 0 2px 0;">Admin Login</h3>
                <div class="fs-subtitle">Manage usage data & the activity log</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if authenticate(username.strip(), password):
                log_in_session(username.strip())
                st.success("Login successful — redirecting…")
                time.sleep(0.4)
                st.rerun()
            else:
                st.markdown(
                    '<div class="fs-result-fault">Incorrect username or password.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="fs-subtitle" style="text-align:center;margin-top:14px;">'
            'Default account: <b>admin</b> / <b>admin123</b> — change it below after your first login.'
            '</div>',
            unsafe_allow_html=True,
        )


def render_admin_panel():
    username = st.session_state.get("username", "admin")
    sidebar_branding()
    sidebar_account_box(username)
    sidebar_footer()

    st.markdown("## 🔐 Admin Panel")
    st.markdown(
        '<div class="fs-subtitle">View usage statistics, manage the activity log, and update your '
        'password. Public pages (Overview / Detection / Classification / Prediction / Model Comparison) '
        'remain free for anyone to use without logging in.</div>',
        unsafe_allow_html=True,
    )

    last_login = get_last_login(username)
    if last_login:
        st.markdown(
            f'<div class="fs-subtitle">Welcome back, <b>{username}</b> · previous login: {last_login}</div>',
            unsafe_allow_html=True,
        )

    tab_usage, tab_logs, tab_account = st.tabs(["📊 Usage Dashboard", "📋 Activity Log", "🔑 Change Password"])

    # -------------------- Usage Dashboard --------------------
    with tab_usage:
        stats = get_usage_stats()
        section_header("Overall usage", "Every prediction made on a public page is counted here.")

        c1, c2, c3 = st.columns(3)
        by_module = stats["by_module"]
        c1.markdown(
            f"""<div class="fs-card"><div class="fs-metric-label">Total events logged</div>
            <div class="fs-metric-value">{stats['total_events']}</div></div>""",
            unsafe_allow_html=True,
        )
        total_rows = int(by_module["total_rows"].sum()) if not by_module.empty else 0
        c2.markdown(
            f"""<div class="fs-card"><div class="fs-metric-label">Total rows processed</div>
            <div class="fs-metric-value">{total_rows}</div></div>""",
            unsafe_allow_html=True,
        )
        modules_used = len(by_module) if not by_module.empty else 0
        c3.markdown(
            f"""<div class="fs-card"><div class="fs-metric-label">Modules used</div>
            <div class="fs-metric-value">{modules_used} / 3</div></div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        if by_module.empty:
            st.markdown('<div class="fs-subtitle">No activity logged yet — usage will appear here once '
                        'someone uses Detection, Classification, or Prediction.</div>', unsafe_allow_html=True)
        else:
            fig = go.Figure(go.Bar(
                x=by_module["module"], y=by_module["events"],
                marker=dict(color=[ORANGE, GREEN, NAVY][:len(by_module)]),
            ))
            fig.update_layout(
                title="Events per module", template="plotly_white", height=320,
                margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Poppins"),
            )
            st.plotly_chart(fig, use_container_width=True)

            by_result = stats["by_result"]
            for mod in by_module["module"]:
                sub = by_result[by_result["module"] == mod]
                if sub.empty:
                    continue
                fig2 = go.Figure(go.Pie(
                    labels=sub["result"], values=sub["count"], hole=0.5,
                    marker=dict(colors=CHART_SEQUENCE * 3),
                ))
                fig2.update_layout(
                    title=f"{mod} — result breakdown", height=300,
                    margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Poppins"),
                )
                st.plotly_chart(fig2, use_container_width=True)

    # -------------------- Activity Log --------------------
    with tab_logs:
        section_header("Activity log", "Most recent 200 events. Delete individual entries or clear everything.")
        logs_df = get_logs(limit=200)

        if logs_df.empty:
            st.markdown('<div class="fs-subtitle">No activity logged yet.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(logs_df, use_container_width=True, height=380, hide_index=True)

            st.write("")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown("**Delete a single entry**")
                log_id = st.number_input("Log ID", min_value=int(logs_df["id"].min()),
                                          max_value=int(logs_df["id"].max()), step=1, key="admin_del_id")
                if st.button("🗑️ Delete This Entry", key="admin_del_one_btn"):
                    if delete_log(int(log_id)):
                        st.success(f"Deleted log entry #{int(log_id)}.")
                        st.rerun()
                    else:
                        st.error("That log ID doesn't exist.")

            with dcol2:
                st.markdown("**Clear everything**")
                confirm = st.checkbox("I understand this cannot be undone", key="admin_clear_confirm")
                if st.button("🗑️ Clear All Activity Logs", key="admin_clear_all_btn", disabled=not confirm):
                    clear_all_logs()
                    st.success("All activity logs cleared.")
                    st.rerun()

    # -------------------- Change Password --------------------
    with tab_account:
        section_header("Change password", "Choose a new password (minimum 6 characters).")
        with st.form("admin_change_password_form"):
            old_pw = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password", type="password")
            confirm_pw = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Update Password", use_container_width=True)

        if submitted:
            if new_pw != confirm_pw:
                st.markdown('<div class="fs-result-fault">New password and confirmation do not match.</div>',
                            unsafe_allow_html=True)
            else:
                ok, message = change_password(username, old_pw, new_pw)
                if ok:
                    st.markdown(f'<div class="fs-result-normal">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="fs-result-fault">{message}</div>', unsafe_allow_html=True)


if is_logged_in():
    render_admin_panel()
else:
    render_login()
