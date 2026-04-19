"""Backfill `steps.error_criteria` from the transitional
`prompt_builder._ERROR_CRITERIA_OVERRIDES` dict into the SQLite DB.

Runs once after `init_db()` has added the `error_criteria` column (on the
next server start after this change lands). Idempotent — re-running writes
the same values. Safe to delete this script once the override dict is
removed from `prompt_builder.py`.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python scripts/backfill_error_criteria.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from config import DB_PATH  # noqa: E402
from services.prompt_builder import _ERROR_CRITERIA_OVERRIDES  # noqa: E402


def main() -> int:
    if not _ERROR_CRITERIA_OVERRIDES:
        print("No overrides in _ERROR_CRITERIA_OVERRIDES — nothing to backfill.")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        try:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(steps)").fetchall()]
        except sqlite3.OperationalError as exc:
            print(f"Failed to read steps schema: {exc}", file=sys.stderr)
            return 1
        if "error_criteria" not in cols:
            print(
                "steps.error_criteria column missing. Restart the backend once "
                "(init_db's migration loop adds the column) and re-run.",
                file=sys.stderr,
            )
            return 1

        updated = 0
        for (procedure_id, step_number), errors in _ERROR_CRITERIA_OVERRIDES.items():
            cur = conn.execute(
                "UPDATE steps SET error_criteria = ? "
                "WHERE procedure_id = ? AND step_number = ?",
                (json.dumps(errors), procedure_id, step_number),
            )
            if cur.rowcount == 0:
                print(
                    f"  ! no row for procedure_id={procedure_id} step={step_number} "
                    f"— skipped ({len(errors)} errors)",
                )
            else:
                print(
                    f"  ✓ procedure_id={procedure_id} step={step_number} "
                    f"← {len(errors)} errors",
                )
                updated += 1
        conn.commit()
        print(f"\nDone. {updated} step row(s) updated.")

        # Verify by reading back
        print("\nVerification:")
        for (procedure_id, step_number), _ in _ERROR_CRITERIA_OVERRIDES.items():
            row = conn.execute(
                "SELECT error_criteria FROM steps "
                "WHERE procedure_id = ? AND step_number = ?",
                (procedure_id, step_number),
            ).fetchone()
            if row:
                decoded = json.loads(row["error_criteria"])
                print(f"  procedure_id={procedure_id[:8]}... step={step_number}: "
                      f"{len(decoded)} errors → {decoded[0][:60]}...")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
