"""One-off: re-extract clips for every completed procedure using the updated ffmpeg command.

Keeps DB procedure rows + Gemini-extracted steps/timestamps intact. Only the per-step
.mp4 files under data/clips/{id}/ change, with clip_url/clip_status updated accordingly.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python reprocess_clips.py [--procedure ID_PREFIX]
"""

import argparse
import asyncio
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from config import CLIPS_DIR, DB_PATH
from services.clip_extractor import extract_clips

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def load_procedures() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        procs = []
        for row in conn.execute(
            """SELECT id, title, source_video, status FROM procedures
               WHERE status IN ('completed', 'completed_partial')
               ORDER BY created_at"""
        ).fetchall():
            steps = conn.execute(
                """SELECT step_number, timestamp_start, timestamp_end
                   FROM steps WHERE procedure_id = ? ORDER BY step_number""",
                (row["id"],),
            ).fetchall()
            procs.append({
                "id": row["id"],
                "title": row["title"],
                "source_video": row["source_video"],
                "old_status": row["status"],
                "steps": [dict(s) for s in steps],
            })
        return procs
    finally:
        conn.close()


def apply_update(procedure_id: str, failed_steps: list[int]) -> None:
    """Update each step's clip_status/clip_url and the procedure's overall status."""
    failed_set = set(failed_steps)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        step_nums = [
            r[0]
            for r in conn.execute(
                "SELECT step_number FROM steps WHERE procedure_id = ?",
                (procedure_id,),
            ).fetchall()
        ]
        for n in step_nums:
            if n in failed_set:
                conn.execute(
                    """UPDATE steps SET clip_url = NULL, clip_status = 'failed'
                       WHERE procedure_id = ? AND step_number = ?""",
                    (procedure_id, n),
                )
            else:
                conn.execute(
                    """UPDATE steps SET clip_url = ?, clip_status = 'ready'
                       WHERE procedure_id = ? AND step_number = ?""",
                    (f"/api/clips/{procedure_id}/step_{n}.mp4", procedure_id, n),
                )
        new_status = "completed_partial" if failed_steps else "completed"
        conn.execute(
            "UPDATE procedures SET status = ? WHERE id = ?",
            (new_status, procedure_id),
        )
        conn.commit()
    finally:
        conn.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--procedure",
        help="Only reprocess procedures whose id starts with this prefix",
    )
    args = parser.parse_args()

    procs = load_procedures()
    if args.procedure:
        procs = [p for p in procs if p["id"].startswith(args.procedure)]
        if not procs:
            print(f"No completed procedure matches prefix {args.procedure!r}")
            return 1
    print(f"Found {len(procs)} completed procedures to reprocess.\n")

    total_ok = 0
    total_failed = 0
    total_missing_source = 0

    for i, p in enumerate(procs, 1):
        proc_id = p["id"]
        title = p["title"] or "(untitled)"
        source = Path(p["source_video"])
        print(f"[{i}/{len(procs)}] {proc_id}  —  {title}")
        print(f"    source: {source.name}, steps: {len(p['steps'])}")

        if not source.exists():
            print("    SKIP: source video missing on disk\n")
            total_missing_source += 1
            continue

        clip_dir = CLIPS_DIR / proc_id
        if clip_dir.exists():
            shutil.rmtree(clip_dir)

        try:
            clip_paths, failed = await extract_clips(
                str(source), proc_id, p["steps"], CLIPS_DIR
            )
        except Exception as exc:
            print(f"    ERROR during extract_clips: {exc}\n")
            total_failed += 1
            continue

        apply_update(proc_id, failed)

        status = "completed_partial" if failed else "completed"
        print(
            f"    OK: {len(clip_paths)} clips, {len(failed)} failed → status={status}\n"
        )
        total_ok += 1

    print("=" * 60)
    print(f"Reprocessed successfully: {total_ok}")
    print(f"Source missing (skipped): {total_missing_source}")
    print(f"Errors:                   {total_failed}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
