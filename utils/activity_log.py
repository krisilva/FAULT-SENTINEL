"""
Fault Sentinel — Activity Logging
Every prediction made from a public page (Detection / Classification /
Prediction) is logged here, so the admin can see how the system is being
used, and view/delete records from the Admin Panel.

Shares the same SQLite database file as utils/auth.py
(database/fault_sentinel.db), just a different table.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "fault_sentinel.db")


def _get_conn():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_activity_table():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            module TEXT NOT NULL,
            mode TEXT NOT NULL,
            input_summary TEXT,
            result TEXT NOT NULL,
            confidence REAL,
            rows_processed INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


def log_activity(module: str, mode: str, input_summary: str, result: str,
                  confidence: float = None, rows_processed: int = 1):
    """Records one usage event. Called automatically after every prediction
    on the public Detection / Classification / Prediction pages."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO activity_logs
           (timestamp, module, mode, input_summary, result, confidence, rows_processed)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(timespec="seconds"), module, mode,
         input_summary, result, confidence, rows_processed),
    )
    conn.commit()
    conn.close()


def get_logs(limit: int = 200) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM activity_logs ORDER BY id DESC LIMIT ?", conn, params=(limit,)
    )
    conn.close()
    return df


def delete_log(log_id: int) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM activity_logs WHERE id = ?", (log_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def clear_all_logs():
    conn = _get_conn()
    conn.execute("DELETE FROM activity_logs")
    conn.commit()
    conn.close()


def get_usage_stats() -> dict:
    """Returns counts per module and per-module result breakdowns for the
    Admin usage dashboard."""
    conn = _get_conn()
    by_module = pd.read_sql_query(
        "SELECT module, COUNT(*) AS events, COALESCE(SUM(rows_processed),0) AS total_rows "
        "FROM activity_logs GROUP BY module", conn,
    )
    by_result = pd.read_sql_query(
        "SELECT module, result, COUNT(*) AS count FROM activity_logs GROUP BY module, result", conn,
    )
    total_events = pd.read_sql_query("SELECT COUNT(*) AS c FROM activity_logs", conn).iloc[0]["c"]
    conn.close()
    return {"by_module": by_module, "by_result": by_result, "total_events": int(total_events)}


# Ensure the table exists as soon as this module is imported.
init_activity_table()
