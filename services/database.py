import json
import logging
import sqlite3
from pathlib import Path

from config import DB_PATH

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS procedures (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            total_duration REAL NOT NULL,
            source_video TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_id TEXT NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
            step_number INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            timestamp_start REAL NOT NULL,
            timestamp_end REAL NOT NULL,
            tips TEXT NOT NULL DEFAULT '[]',
            warnings TEXT NOT NULL DEFAULT '[]',
            clip_url TEXT,
            UNIQUE(procedure_id, step_number)
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def save_procedure(procedure: dict, steps: list[dict]):
    """Insert a procedure and its steps in a single transaction."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO procedures (id, title, description, total_duration, source_video, created_at)
               VALUES (:id, :title, :description, :total_duration, :source_video, :created_at)""",
            procedure,
        )
        for step in steps:
            conn.execute(
                """INSERT INTO steps (procedure_id, step_number, title, description,
                                      timestamp_start, timestamp_end, tips, warnings, clip_url)
                   VALUES (:procedure_id, :step_number, :title, :description,
                           :timestamp_start, :timestamp_end, :tips, :warnings, :clip_url)""",
                {
                    "procedure_id": procedure["id"],
                    "step_number": step["step_number"],
                    "title": step["title"],
                    "description": step["description"],
                    "timestamp_start": step["timestamp_start"],
                    "timestamp_end": step["timestamp_end"],
                    "tips": json.dumps(step.get("tips", [])),
                    "warnings": json.dumps(step.get("warnings", [])),
                    "clip_url": step.get("clip_url"),
                },
            )
        conn.commit()
        logger.info("Saved procedure %s with %d steps", procedure["id"], len(steps))
    finally:
        conn.close()


def get_procedure(procedure_id: str) -> dict | None:
    """Retrieve a procedure with all its steps."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM procedures WHERE id = ?", (procedure_id,)
        ).fetchone()
        if not row:
            return None

        proc = dict(row)
        step_rows = conn.execute(
            "SELECT * FROM steps WHERE procedure_id = ? ORDER BY step_number",
            (procedure_id,),
        ).fetchall()

        proc["steps"] = []
        for s in step_rows:
            step = dict(s)
            step["tips"] = json.loads(step["tips"])
            step["warnings"] = json.loads(step["warnings"])
            # Remove internal DB fields from response
            step.pop("id", None)
            step.pop("procedure_id", None)
            proc["steps"].append(step)

        return proc
    finally:
        conn.close()


def list_procedures() -> list[dict]:
    """List all procedures with step counts, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT p.id, p.title, p.description, p.total_duration, p.created_at,
                   COUNT(s.id) as step_count
            FROM procedures p
            LEFT JOIN steps s ON s.procedure_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_procedure(procedure_id: str) -> bool:
    """Delete a procedure and its steps. Returns True if found."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM procedures WHERE id = ?", (procedure_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted procedure %s", procedure_id)
        return deleted
    finally:
        conn.close()
