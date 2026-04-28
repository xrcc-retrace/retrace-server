#!/usr/bin/env python3
"""Backfill icon_symbol + icon_emoji on existing procedures.

Walks the procedures table, finds rows where the icon fields are still
empty (i.e. were created before icons were a thing), asks Gemini Flash
to pick an SF Symbol name and a single emoji per procedure based on its
title + description, and writes the result back.

Run once after deploying the icon-aware schema. Idempotent — re-running
is a no-op for already-filled rows. Safe to run while the server is up
because SQLite WAL allows concurrent readers.

Usage:
    python backfill_icons.py            # backfill any empty rows
    python backfill_icons.py --force    # re-pick icons for ALL procedures
    python backfill_icons.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from config import DB_PATH
from services.database import init_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_icons")


PROMPT_TEMPLATE = """Pick a single Iconify icon ID and a single emoji that best represent the MAIN PHYSICAL OBJECT this procedure works on (the appliance / device / piece of equipment, not the action being performed).

Procedure title: {title}
Procedure description: {description}

`icon_symbol` rules:
- A single Iconify icon ID in the form "<prefix>:<name>" — e.g. "lucide:refrigerator", "material-symbols:humidifier-large", "phosphor:speaker-high", "tabler:washing-machine".
- Prefer these packs in order: lucide (clean, broad coverage), material-symbols (best for appliances), phosphor, tabler.
- If you don't know an exact icon name, prefer common nouns: refrigerator, microwave, oven, washing-machine, humidifier, fan, lightbulb, tv, speaker, monitor, laptop, mouse, headphones, printer, gamepad, camera, wrench, screwdriver, coffee, bicycle, car.
- The icons render white on a dark surface — pick line-style monochrome icons, not multi-color packs (avoid emojione, twemoji, fxemoji).

`icon_emoji` rules:
- A single emoji character that represents the same object as a fallback when the icon CDN is unreachable.
- Examples: 🧊 fridge, 🍞 oven, 🔥 stove, 🧺 washer, 💧 humidifier, 🪭 fan, 💡 lightbulb, 📺 tv, 🔊 speaker, 📱 iPhone, 💻 laptop, 🖥️ desktop, 🎧 headphones, 🖨️ printer, 🎮 controller, 📷 camera, 🔧 tool, ⛽ fuel, 🚗 car, ☕ cup.

Both fields are required. Pick a single icon_symbol and a single icon_emoji that represent the same physical object."""


class IconResponse(BaseModel):
    icon_symbol: str
    icon_emoji: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-pick icons even for procedures that already have them.",
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


def fetch_targets(conn: sqlite3.Connection, force: bool) -> list[sqlite3.Row]:
    if force:
        sql = """
            SELECT id, title, description, icon_symbol, icon_emoji
            FROM procedures
            WHERE status IN ('completed', 'completed_partial')
            ORDER BY created_at ASC
        """
    else:
        sql = """
            SELECT id, title, description, icon_symbol, icon_emoji
            FROM procedures
            WHERE status IN ('completed', 'completed_partial')
              AND (icon_symbol IS NULL OR icon_symbol = ''
                   OR icon_emoji IS NULL OR icon_emoji = '')
            ORDER BY created_at ASC
        """
    return list(conn.execute(sql))


def pick_icon(client: genai.Client, model: str, title: str, description: str) -> Optional[IconResponse]:
    prompt = PROMPT_TEMPLATE.format(title=title or "(no title)", description=description or "(no description)")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IconResponse,
                http_options=types.HttpOptions(timeout=30_000),
            ),
        )
        return IconResponse.model_validate_json(response.text)
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set (check .env).")
        return 2

    if not DB_PATH.exists():
        logger.error("Database not found at %s", DB_PATH)
        return 2

    # Ensure the icon_symbol / icon_emoji columns exist before we try to
    # write to them. init_db() is idempotent — the ALTER TABLE migrations
    # silently no-op if the columns already exist.
    init_db()

    client = genai.Client(api_key=api_key)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        rows = fetch_targets(conn, force=args.force)
        if not rows:
            logger.info("Nothing to backfill — all procedures already have icons.")
            return 0

        logger.info("Backfilling icons for %d procedure(s) using %s%s",
                    len(rows), args.model, " (DRY RUN)" if args.dry_run else "")

        updated = 0
        skipped = 0
        for row in rows:
            pid = row["id"]
            title = row["title"] or ""
            description = row["description"] or ""

            picked = pick_icon(client, args.model, title, description)
            if picked is None:
                logger.warning("[%s] %s — Gemini returned no icon, skipping", pid[:8], title[:60])
                skipped += 1
                continue

            symbol = (picked.icon_symbol or "").strip()
            emoji = (picked.icon_emoji or "").strip()
            if not symbol and not emoji:
                logger.warning("[%s] %s — empty icon fields, skipping", pid[:8], title[:60])
                skipped += 1
                continue

            logger.info("[%s] %s → symbol=%r emoji=%r",
                        pid[:8], title[:60], symbol, emoji)

            if args.dry_run:
                continue

            conn.execute(
                "UPDATE procedures SET icon_symbol = ?, icon_emoji = ? WHERE id = ?",
                (symbol, emoji, pid),
            )
            conn.commit()
            updated += 1

        if args.dry_run:
            logger.info("Dry run complete. Would have updated %d row(s).", len(rows) - skipped)
        else:
            logger.info("Done. Updated %d row(s), skipped %d.", updated, skipped)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
