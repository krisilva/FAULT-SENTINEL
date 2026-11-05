"""
Fault Sentinel — Admin Authentication
SQLite-backed single-role (admin) login system.

- Passwords are never stored in plain text: PBKDF2-HMAC-SHA256 with a
  random per-user salt (both from Python's standard `hashlib` - no extra
  dependency needed).
- On first run, `init_db()` creates the database and a default admin
  account (username: admin / password: admin123) — change it right away
  from the Account page after logging in.
- `require_login()` is called at the top of every protected page. It
  checks Streamlit's session_state and, if the user isn't logged in,
  shows a message + button back to the Login page and stops the script.
"""

import os
import sqlite3
import hashlib
import hmac
from datetime import datetime

import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "fault_sentinel.db")

DEFAULT_ADMIN_USERNAME = "admin"
def _default_admin_password():
    """Initial admin password: Streamlit secret / env var if set, else a
    local-dev fallback. Always set ADMIN_PASSWORD when deploying publicly."""
    try:
        if "ADMIN_PASSWORD" in st.secrets:
            return str(st.secrets["ADMIN_PASSWORD"])
    except Exception:
        pass
    return os.environ.get("ADMIN_PASSWORD", "admin123")


DEFAULT_ADMIN_PASSWORD = _default_admin_password()

PBKDF2_ITERATIONS = 100_000


def _get_conn():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return pwd_hash.hex(), salt.hex()


def _verify_password(password: str, stored_hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    candidate_hash, _ = _hash_password(password, salt)
    return hmac.compare_digest(candidate_hash, stored_hash_hex)


def init_db():
    """Creates tables if they don't exist yet, and seeds a default admin
    account the first time the app runs. Safe to call on every import."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            login_time TEXT NOT NULL
        )
    """)
    conn.commit()

    cur.execute("SELECT COUNT(*) AS c FROM admins")
    if cur.fetchone()["c"] == 0:
        pwd_hash, salt = _hash_password(DEFAULT_ADMIN_PASSWORD)
        cur.execute(
            "INSERT INTO admins (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (DEFAULT_ADMIN_USERNAME, pwd_hash, salt, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    conn.close()


def authenticate(username: str, password: str) -> bool:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
    if row is None:
        conn.close()
        return False

    ok = _verify_password(password, row["password_hash"], row["salt"])
    if ok:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("UPDATE admins SET last_login = ? WHERE username = ?", (now, username))
        conn.execute("INSERT INTO login_history (username, login_time) VALUES (?, ?)", (username, now))
        conn.commit()
    conn.close()
    return ok


def change_password(username: str, old_password: str, new_password: str):
    """Returns (success: bool, message: str)."""
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    conn = _get_conn()
    row = conn.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
    if row is None:
        conn.close()
        return False, "Account not found."

    if not _verify_password(old_password, row["password_hash"], row["salt"]):
        conn.close()
        return False, "Current password is incorrect."

    new_hash, new_salt = _hash_password(new_password)
    conn.execute("UPDATE admins SET password_hash = ?, salt = ? WHERE username = ?",
                 (new_hash, new_salt, username))
    conn.commit()
    conn.close()
    return True, "Password updated successfully."


def get_last_login(username: str):
    conn = _get_conn()
    row = conn.execute("SELECT last_login FROM admins WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row["last_login"] if row else None


def get_login_history(username: str, limit: int = 10):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT login_time FROM login_history WHERE username = ? ORDER BY login_time DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    conn.close()
    return [r["login_time"] for r in rows]


# ------------------------------------------------------------------ #
# Streamlit session helpers
# ------------------------------------------------------------------ #
def is_logged_in() -> bool:
    return bool(st.session_state.get("logged_in", False))


def log_in_session(username: str):
    st.session_state["logged_in"] = True
    st.session_state["username"] = username


def log_out_session():
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)


def require_login():
    """Call at the top of every protected page. Stops the script and
    shows a way back to the Login page if the user isn't authenticated."""
    if not is_logged_in():
        st.warning("🔒 Please log in first to access this page.")
        if st.button("Go to Login"):
            st.switch_page("app.py")
        st.stop()


# Initialise the database as soon as this module is imported, so every
# page (and app.py) always has a ready database without extra setup code.
init_db()
