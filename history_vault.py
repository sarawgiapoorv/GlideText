"""
history_vault.py — Persistent local history logging for GlideText.

Uses SQLite to store every dictation result with timestamps.
The database file is co-located with the application files.
"""

import sqlite3
import os
import shutil
from datetime import datetime


# ---------------------------------------------------------------------------
# Database path — same directory as this script
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(os.path.abspath(__file__))
_old_db = os.path.join(_base_dir, "localflow_history.db")
_new_db = os.path.join(_base_dir, "glidetext_history.db")

# Seamless migration from legacy LocalFlow database if present
if os.path.exists(_old_db) and not os.path.exists(_new_db):
    try:
        shutil.copy2(_old_db, _new_db)
    except Exception:
        pass

DB_PATH = _new_db


class HistoryVault:
    """Persistent SQLite-backed dictation history log."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """Create the history and api_metrics tables if they don't exist."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    raw_text      TEXT    DEFAULT '',
                    polished_text TEXT    NOT NULL,
                    created_at    REAL    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    provider      TEXT    NOT NULL,
                    model         TEXT    NOT NULL,
                    status        TEXT    NOT NULL,
                    latency_ms    INTEGER NOT NULL,
                    created_at    REAL    NOT NULL
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_entry(self, polished_text: str, raw_text: str = "") -> str:
        """
        Log a dictation result.

        Args:
            polished_text: The final LLM-polished text.
            raw_text:      The raw Whisper transcription (optional).

        Returns:
            The formatted timestamp string for display.
        """
        timestamp = datetime.now().strftime("%b %d, %Y -- %I:%M %p")
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                "INSERT INTO history (timestamp, raw_text, polished_text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, raw_text, polished_text, datetime.now().timestamp()),
            )
            conn.commit()
        return timestamp

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_recent(self, limit: int = 50) -> list[tuple[str, str]]:
        """
        Fetch recent history entries in reverse chronological order.

        Returns:
            List of (timestamp, polished_text) tuples.
        """
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.execute(
                "SELECT timestamp, polished_text FROM history "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return cursor.fetchall()

    def count(self) -> int:
        """Return the total number of history entries."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM history")
            return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Wipe all history entries."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()

    # ------------------------------------------------------------------
    # API Telemetry & Metrics
    # ------------------------------------------------------------------
    def log_api_call(self, provider: str, model: str, status: str, latency_ms: int) -> None:
        """
        Record an API attempt (FreeLLMAPI, Cloud Gemini, or Local LLM).

        Args:
            provider: e.g. 'FreeLLMAPI', 'Gemini (Slot 0)', 'Gemini (Slot 1)', 'Local LLM (llama3.2:3b)'
            model: e.g. 'auto', 'gemini-2.5-flash', 'llama3.2:3b'
            status: 'SUCCESS', 'RATE_LIMIT_429', 'TIMEOUT', 'OVERLOAD_503', 'CONNECTION_ERROR', 'ERROR'
            latency_ms: Roundtrip execution time in milliseconds.
        """
        timestamp = datetime.now().strftime("%b %d, %Y -- %I:%M:%S %p")
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                "INSERT INTO api_metrics (timestamp, provider, model, status, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, provider, model, status, int(latency_ms), datetime.now().timestamp()),
            )
            conn.commit()

    def get_api_analytics(self) -> dict:
        """
        Aggregate API call metrics for display in the UI.

        Returns:
            Dictionary with total_calls, total_success, success_rate, and per-provider stats.
        """
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            
            # Overall totals
            cursor.execute("SELECT COUNT(*), SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) FROM api_metrics")
            row = cursor.fetchone()
            total_calls = row[0] or 0
            total_success = row[1] or 0
            success_rate = (total_success / total_calls * 100.0) if total_calls > 0 else 100.0

            # Provider breakdown
            cursor.execute("""
                SELECT provider,
                       COUNT(*) AS total,
                       SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) AS success,
                       SUM(CASE WHEN status='RATE_LIMIT_429' THEN 1 ELSE 0 END) AS rate_limits,
                       SUM(CASE WHEN status NOT IN ('SUCCESS', 'RATE_LIMIT_429') THEN 1 ELSE 0 END) AS other_errors,
                       AVG(latency_ms) AS avg_latency
                FROM api_metrics
                GROUP BY provider
                ORDER BY total DESC
            """)
            providers = []
            for p_row in cursor.fetchall():
                p_total = p_row[1]
                p_success = p_row[2] or 0
                providers.append({
                    "provider": p_row[0],
                    "total": p_total,
                    "success": p_success,
                    "rate_limits": p_row[3] or 0,
                    "other_errors": p_row[4] or 0,
                    "avg_latency": int(p_row[5] or 0),
                    "success_rate": (p_success / p_total * 100.0) if p_total > 0 else 100.0,
                })

            # Recent 10 calls
            cursor.execute("""
                SELECT timestamp, provider, model, status, latency_ms
                FROM api_metrics
                ORDER BY id DESC LIMIT 10
            """)
            recent_calls = [
                {"timestamp": r[0], "provider": r[1], "model": r[2], "status": r[3], "latency_ms": r[4]}
                for r in cursor.fetchall()
            ]

            return {
                "total_calls": total_calls,
                "total_success": total_success,
                "success_rate": round(success_rate, 1),
                "providers": providers,
                "recent_calls": recent_calls,
            }

    def clear_api_metrics(self) -> None:
        """Clear all logged API metrics."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("DELETE FROM api_metrics")
            conn.commit()

