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
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            error_message TEXT
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
            completion_criteria TEXT NOT NULL DEFAULT '',
            error_criteria TEXT NOT NULL DEFAULT '[]',
            clip_url TEXT,
            clip_status TEXT NOT NULL DEFAULT 'ready',
            UNIQUE(procedure_id, step_number)
        );
    """)
    # Migrate existing databases that lack the new columns
    for col, definition in [
        ("status", "TEXT NOT NULL DEFAULT 'completed'"),
        ("error_message", "TEXT"),
        # 'video' (expert upload) | 'manual' (ingested PDF via troubleshoot).
        # Defaulted so existing rows stay valid.
        ("source_type", "TEXT NOT NULL DEFAULT 'video'"),
        # Procedure-level iconography. SF Symbol name + emoji fallback,
        # picked by Gemini at extraction time. Empty string on legacy rows
        # until the backfill script runs.
        ("icon_symbol", "TEXT NOT NULL DEFAULT ''"),
        ("icon_emoji", "TEXT NOT NULL DEFAULT ''"),
        # Closed-enum category Gemini picks at extraction time. Used by the
        # iOS Discover chip filter. Legacy rows default to 'Other'.
        ("category", "TEXT NOT NULL DEFAULT 'Other'"),
        # Online sources cited by troubleshoot web-search flow.
        # JSON-serialized list of {title, url, snippet}; '[]' on legacy rows.
        ("sources", "TEXT NOT NULL DEFAULT '[]'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE procedures ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass  # column already exists
    for col, definition in [
        ("completion_criteria", "TEXT NOT NULL DEFAULT ''"),
        ("clip_status", "TEXT NOT NULL DEFAULT 'ready'"),
        ("error_criteria", "TEXT NOT NULL DEFAULT '[]'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE steps ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def create_procedure_placeholder(procedure_id: str, source_video: str, created_at: str):
    """Insert a placeholder procedure row with status='processing'."""
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO procedures (id, title, description, total_duration, source_video, created_at, status)
               VALUES (?, '', '', 0, ?, ?, 'processing')""",
            (procedure_id, source_video, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def update_procedure_status(procedure_id: str, status: str, error_message: str | None = None):
    """Update the processing status of a procedure."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE procedures SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, procedure_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_procedure(procedure: dict, steps: list[dict]):
    """Update a placeholder procedure with results and insert its steps.

    `procedure` must contain: id, title, description, total_duration, status.
    Optional: `source_type` ('video' default, 'manual' for PDF-ingested SOPs).
    """
    conn = _get_conn()
    source_type = procedure.get("source_type", "video")
    icon_symbol = procedure.get("icon_symbol", "") or ""
    icon_emoji = procedure.get("icon_emoji", "") or ""
    category = procedure.get("category", "Other") or "Other"
    sources_json = json.dumps(procedure.get("sources", []) or [])
    try:
        conn.execute(
            """UPDATE procedures SET title=:title, description=:description,
               total_duration=:total_duration, status=:status,
               source_type=:source_type,
               icon_symbol=:icon_symbol, icon_emoji=:icon_emoji,
               category=:category, sources=:sources
               WHERE id=:id""",
            {**procedure, "source_type": source_type,
             "icon_symbol": icon_symbol, "icon_emoji": icon_emoji,
             "category": category, "sources": sources_json},
        )
        for step in steps:
            conn.execute(
                """INSERT INTO steps (procedure_id, step_number, title, description,
                                      timestamp_start, timestamp_end, tips, warnings,
                                      completion_criteria, error_criteria,
                                      clip_url, clip_status)
                   VALUES (:procedure_id, :step_number, :title, :description,
                           :timestamp_start, :timestamp_end, :tips, :warnings,
                           :completion_criteria, :error_criteria,
                           :clip_url, :clip_status)""",
                {
                    "procedure_id": procedure["id"],
                    "step_number": step["step_number"],
                    "title": step["title"],
                    "description": step["description"],
                    "timestamp_start": step["timestamp_start"],
                    "timestamp_end": step["timestamp_end"],
                    "tips": json.dumps(step.get("tips", [])),
                    "warnings": json.dumps(step.get("warnings", [])),
                    "completion_criteria": step.get("completion_criteria", ""),
                    "error_criteria": json.dumps(step.get("error_criteria", [])),
                    "clip_url": step.get("clip_url"),
                    "clip_status": step.get("clip_status", "ready"),
                },
            )
        conn.commit()
        logger.info(
            "Saved procedure %s (status=%s) with %d steps",
            procedure["id"], procedure.get("status"), len(steps),
        )
    finally:
        conn.close()


def get_procedure(procedure_id: str) -> dict | None:
    """Retrieve a procedure with all its steps.

    `source_video` is normalized to its basename here so callers (API responses,
    learner sessions) can build `/api/uploads/{filename}` URLs without leaking
    the server's filesystem layout. Callers that need the on-disk path (e.g.
    `reprocess_clips.py`) read directly from SQL with the full path intact.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM procedures WHERE id = ?", (procedure_id,)
        ).fetchone()
        if not row:
            return None

        proc = dict(row)
        if proc.get("source_video"):
            proc["source_video"] = Path(proc["source_video"]).name
        # JSON-deserialize the sources column. Default to [] for rows that
        # predate the migration or for procedures without web sources.
        proc["sources"] = json.loads(proc.get("sources") or "[]")
        step_rows = conn.execute(
            "SELECT * FROM steps WHERE procedure_id = ? ORDER BY step_number",
            (procedure_id,),
        ).fetchall()

        proc["steps"] = []
        for s in step_rows:
            step = dict(s)
            step["tips"] = json.loads(step["tips"])
            step["warnings"] = json.loads(step["warnings"])
            # error_criteria may be absent on rows from pre-migration DBs if a
            # caller somehow reaches them (init_db migrates on server start so
            # this shouldn't happen in practice). Default to [] for safety.
            step["error_criteria"] = json.loads(step.get("error_criteria") or "[]")
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
                   p.status, p.error_message, p.source_type,
                   p.icon_symbol, p.icon_emoji, p.category, p.sources,
                   COUNT(s.id) as step_count
            FROM procedures p
            LEFT JOIN steps s ON s.procedure_id = p.id
            WHERE p.status IN ('processing', 'completed', 'completed_partial')
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d.get("sources") or "[]")
            result.append(d)
        return result
    finally:
        conn.close()


def update_procedure_fields(procedure_id: str, title: str | None = None,
                            description: str | None = None,
                            step_order: list[int] | None = None) -> bool:
    """Update procedure fields. Returns True if found."""
    conn = _get_conn()
    try:
        # Check existence
        row = conn.execute("SELECT id FROM procedures WHERE id = ?", (procedure_id,)).fetchone()
        if not row:
            return False

        if title is not None:
            conn.execute("UPDATE procedures SET title = ? WHERE id = ?", (title, procedure_id))
        if description is not None:
            conn.execute("UPDATE procedures SET description = ? WHERE id = ?", (description, procedure_id))

        # Step reordering: update step_number for each step to match new order
        if step_order is not None:
            for new_index, old_step_number in enumerate(step_order, start=1):
                conn.execute(
                    """UPDATE steps SET step_number = -? WHERE procedure_id = ? AND step_number = ?""",
                    (new_index, procedure_id, old_step_number),
                )
            # Negative numbers used as temp to avoid unique constraint violations, now make positive
            conn.execute(
                "UPDATE steps SET step_number = -step_number WHERE procedure_id = ? AND step_number < 0",
                (procedure_id,),
            )

        conn.commit()
        logger.info("Updated procedure %s", procedure_id)
        return True
    finally:
        conn.close()


def update_step_fields(procedure_id: str, step_number: int,
                       title: str | None = None, description: str | None = None,
                       tips: list[str] | None = None, warnings: list[str] | None = None,
                       timestamp_start: float | None = None,
                       timestamp_end: float | None = None,
                       error_criteria: list[str] | None = None) -> bool:
    """Update step fields. Returns True if found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM steps WHERE procedure_id = ? AND step_number = ?",
            (procedure_id, step_number),
        ).fetchone()
        if not row:
            return False

        if title is not None:
            conn.execute(
                "UPDATE steps SET title = ? WHERE procedure_id = ? AND step_number = ?",
                (title, procedure_id, step_number),
            )
        if description is not None:
            conn.execute(
                "UPDATE steps SET description = ? WHERE procedure_id = ? AND step_number = ?",
                (description, procedure_id, step_number),
            )
        if tips is not None:
            conn.execute(
                "UPDATE steps SET tips = ? WHERE procedure_id = ? AND step_number = ?",
                (json.dumps(tips), procedure_id, step_number),
            )
        if warnings is not None:
            conn.execute(
                "UPDATE steps SET warnings = ? WHERE procedure_id = ? AND step_number = ?",
                (json.dumps(warnings), procedure_id, step_number),
            )
        if timestamp_start is not None:
            conn.execute(
                "UPDATE steps SET timestamp_start = ? WHERE procedure_id = ? AND step_number = ?",
                (timestamp_start, procedure_id, step_number),
            )
        if timestamp_end is not None:
            conn.execute(
                "UPDATE steps SET timestamp_end = ? WHERE procedure_id = ? AND step_number = ?",
                (timestamp_end, procedure_id, step_number),
            )
        if error_criteria is not None:
            conn.execute(
                "UPDATE steps SET error_criteria = ? WHERE procedure_id = ? AND step_number = ?",
                (json.dumps(error_criteria), procedure_id, step_number),
            )

        conn.commit()
        logger.info("Updated step %d of procedure %s", step_number, procedure_id)
        return True
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
