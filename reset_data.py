"""Wipe all user-generated data: DB rows, source uploads, per-step clips.

Preserves: source code, .venv, .env, static/voice_previews/ (regenerated via
scripts/generate_voice_previews.py), and the empty data/ + uploads/ directories
themselves (config.py recreates them on import anyway).

In-memory learner sessions live in the server process — this script can't reach
them. Stop the server first for a fully clean slate.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python reset_data.py              # interactive confirm
    python reset_data.py --yes        # skip prompt
    python reset_data.py --dry-run    # just print what would happen
    python reset_data.py --yes --force  # also skip the running-server guard
"""

import argparse
import shutil
import sqlite3
import sys
from urllib.error import URLError
from urllib.request import urlopen

from config import CLIPS_DIR, DB_PATH, UPLOAD_DIR


def count_targets() -> tuple[int, int, int]:
    procedures = 0
    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        try:
            procedures = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        except sqlite3.OperationalError:
            procedures = 0
        finally:
            conn.close()

    upload_files = sum(1 for p in UPLOAD_DIR.iterdir() if p.is_file()) if UPLOAD_DIR.exists() else 0
    clip_dirs = sum(1 for p in CLIPS_DIR.iterdir() if p.is_dir()) if CLIPS_DIR.exists() else 0
    return procedures, upload_files, clip_dirs


def server_is_running() -> bool:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=0.5) as resp:
            return resp.status == 200
    except (URLError, TimeoutError, ConnectionError, OSError):
        return False


def wipe_db() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        deleted = conn.execute("DELETE FROM procedures").rowcount
        conn.commit()
        conn.execute("VACUUM")
        return deleted
    finally:
        conn.close()


def wipe_uploads() -> int:
    if not UPLOAD_DIR.exists():
        return 0
    removed = 0
    for p in UPLOAD_DIR.iterdir():
        if p.is_file():
            p.unlink()
            removed += 1
        elif p.is_dir():
            shutil.rmtree(p)
            removed += 1
    return removed


def wipe_clips() -> int:
    if not CLIPS_DIR.exists():
        return 0
    removed = 0
    for p in CLIPS_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
            removed += 1
        elif p.is_file():
            p.unlink()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--force", action="store_true", help="Proceed even if a server is detected on :8000.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary, make no changes.")
    args = parser.parse_args()

    procs, uploads, clips = count_targets()
    print("Targets for deletion:")
    print(f"  procedures (DB rows): {procs}")
    print(f"  source videos in uploads/: {uploads}")
    print(f"  clip directories in data/clips/: {clips}")
    print()

    if procs == 0 and uploads == 0 and clips == 0:
        print("Nothing to delete — already empty.")
        return 0

    if args.dry_run:
        print("--dry-run: no changes made.")
        return 0

    if server_is_running():
        print("WARNING: a server appears to be running on http://127.0.0.1:8000")
        print("  Learner sessions live in the server process and won't be cleared.")
        print("  Stop the server (Ctrl-C main.py) for a fully clean slate.")
        if not args.force:
            print("  Re-run with --force to proceed anyway.")
            return 2
        print("  --force set; proceeding.")
        print()

    if not args.yes:
        reply = input("Type 'yes' to delete all user data: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return 1

    deleted_procs = wipe_db()
    removed_uploads = wipe_uploads()
    removed_clips = wipe_clips()

    print()
    print("Done.")
    print(f"  procedures deleted: {deleted_procs}")
    print(f"  upload files removed: {removed_uploads}")
    print(f"  clip directories removed: {removed_clips}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
