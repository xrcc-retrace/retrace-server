#!/usr/bin/env python3
"""Backfill steps.display_summary on existing procedures.

Walks every completed procedure, finds steps whose display_summary is
still empty (rows created before the lens-summary field existed), and
asks Gemini Flash to condense each step's description into a glanceable
1-2 sentence summary (≤160 chars) for the glasses lens. One batched call
per procedure — all steps in, all summaries out.

This is the deliberate alternative to bumping EXTRACTION_PIPELINE_VERSION:
a full re-extraction would wholesale-overwrite hand-edited parts
glossaries and step text. Text-condensation needs no video, so a Flash
call over the existing descriptions is enough — and it covers manual-
and web-sourced procedures too, which re-extraction never touches.

Run once after deploying the display_summary schema. Idempotent —
re-running is a no-op for already-filled steps. Safe to run while the
server is up (SQLite WAL allows concurrent readers).

Usage:
    python backfill_summaries.py            # fill any empty steps
    python backfill_summaries.py --force    # regenerate for ALL steps
    python backfill_summaries.py --dry-run  # print proposals, write nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import DB_PATH
from services.database import init_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_summaries")

MAX_CHARS = 160

SYSTEM_PROMPT = """You condense procedure-step instructions for a tiny glasses heads-up display.

For EVERY step you are given, write a display summary:
- 1-2 short sentences, MAX 160 characters, self-contained enough to act on.
- Keep the key action, part names, and quantities.
- Cut rationale, caveats, reassurance, and anything a tips/warnings line already covers.
- Always materially shorter than the original description — never a copy of it. If the description is already terse, tighten it anyway (drop connectives, keep the imperative).
- Do not invent details that are not in the description.

Return one summary per step, in the same order, tagged with the same step_number."""


class _StepSummary(BaseModel):
    step_number: int
    display_summary: str = Field(description="1-2 short sentences, max 160 characters.")


class _SummaryBatch(BaseModel):
    summaries: list[_StepSummary]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summaries even for steps that already have one.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not write to the DB.",
    )
    p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    return p.parse_args()


def fetch_procedures(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute(
        """SELECT id, title FROM procedures
           WHERE status IN ('completed', 'completed_partial')
           ORDER BY created_at ASC"""
    ))


def fetch_steps(conn: sqlite3.Connection, procedure_id: str, force: bool) -> list[sqlite3.Row]:
    sql = """SELECT step_number, title, description, display_summary
             FROM steps WHERE procedure_id = ?"""
    if not force:
        sql += " AND (display_summary IS NULL OR display_summary = '')"
    sql += " ORDER BY step_number ASC"
    return list(conn.execute(sql, (procedure_id,)))


def summarize(client: genai.Client, model: str, proc_title: str,
              steps: list[sqlite3.Row]) -> dict[int, str] | None:
    """One batched Flash call → {step_number: summary}. None on failure."""
    lines = [f"Procedure: {proc_title}", ""]
    for s in steps:
        lines.append(f"Step {s['step_number']} — {s['title']}")
        lines.append(f"Description: {s['description']}")
        lines.append("")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[types.Part.from_text(text="\n".join(lines))])],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_SummaryBatch,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=60_000),
            ),
        )
        parsed = _SummaryBatch.model_validate_json(response.text)
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None
    out: dict[int, str] = {}
    for item in parsed.summaries:
        text = item.display_summary.strip()
        if not text:
            continue
        if len(text) > MAX_CHARS + 40:  # tolerate mild overshoot, reject copies
            logger.warning("  step %d summary too long (%d chars), skipping",
                           item.step_number, len(text))
            continue
        out[item.step_number] = text
    return out


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set (check .env).")
        return 2
    if not DB_PATH.exists():
        logger.error("Database not found at %s", DB_PATH)
        return 2

    # Ensure the display_summary column exists (idempotent migrations).
    init_db()

    client = genai.Client(api_key=api_key)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        updated = 0
        skipped = 0
        for proc in fetch_procedures(conn):
            steps = fetch_steps(conn, proc["id"], force=args.force)
            if not steps:
                continue
            logger.info("[%s] %s — %d step(s) to summarize%s",
                        proc["id"][:8], (proc["title"] or "")[:60], len(steps),
                        " (DRY RUN)" if args.dry_run else "")
            summaries = summarize(client, args.model, proc["title"] or "", steps)
            if summaries is None:
                skipped += len(steps)
                continue
            for s in steps:
                text = summaries.get(s["step_number"])
                if not text:
                    logger.warning("  step %d: no summary returned, skipping",
                                   s["step_number"])
                    skipped += 1
                    continue
                logger.info("  step %d (%d chars): %s", s["step_number"], len(text), text)
                if args.dry_run:
                    continue
                conn.execute(
                    "UPDATE steps SET display_summary = ? WHERE procedure_id = ? AND step_number = ?",
                    (text, proc["id"], s["step_number"]),
                )
                updated += 1
            conn.commit()

        if args.dry_run:
            logger.info("Dry run complete (nothing written).")
        else:
            logger.info("Done. Updated %d step(s), skipped %d.", updated, skipped)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
