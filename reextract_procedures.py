"""Re-extract video procedures with the current (tightened) extractor prompt.

Unlike reprocess_clips.py (which re-cuts clips from EXISTING timestamps), this
re-runs Gemini 2.5 Pro over each procedure's source video, so completion_criteria,
error_criteria, visual_verifiability, and timestamps are all regenerated — then
re-cuts the clips to match. Use after tightening the extraction prompt.

    # preview old→new for every video procedure (NO writes):
    python reextract_procedures.py --dry-run

    # preview one:
    python reextract_procedures.py --dry-run --procedure 173f7e92

    # apply (rewrites steps + clips in the DB):
    python reextract_procedures.py --apply [--procedure ID] [--limit N]

--dry-run is the default; nothing is written without --apply.
"""

import argparse
import asyncio
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from config import CLIPS_DIR, DB_PATH, GEMINI_API_KEY
from models.procedure import ProcedureStep
from services.clip_extractor import extract_clips
from services.database import save_procedure
from services.gemini_processor import process_expert_video
from services.video_probe import probe_duration

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

EVALS_FRAMES = Path(__file__).parent / "evals" / "fixtures" / "frames"

C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "d": "\033[90m",
     "b": "\033[1m", "x": "\033[0m"}


def load_video_procedures() -> list[dict]:
    """Video procedures with full on-disk source path + current steps (for the diff)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        procs = []
        for row in conn.execute(
            """SELECT id, title, source_video, status FROM procedures
               WHERE status IN ('completed', 'completed_partial')
                 AND source_type = 'video'
               ORDER BY created_at"""
        ).fetchall():
            steps = conn.execute(
                """SELECT step_number, completion_criteria, visual_verifiability
                   FROM steps WHERE procedure_id = ? ORDER BY step_number""",
                (row["id"],),
            ).fetchall()
            procs.append({
                "id": row["id"], "title": row["title"],
                "source_video": row["source_video"],
                "old_steps": {s["step_number"]: dict(s) for s in steps},
            })
        return procs
    finally:
        conn.close()


def _clamp_steps(gemini_steps: list[dict], duration: float) -> list[dict]:
    out = []
    for s in gemini_steps:
        start = max(0.0, min(s["timestamp_start"], duration))
        end = max(start, min(s["timestamp_end"], duration))
        d = dict(s)
        d["timestamp_start"], d["timestamp_end"] = start, end
        out.append(d)
    return out


def _delete_steps(procedure_id: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("DELETE FROM steps WHERE procedure_id = ?", (procedure_id,))
        conn.commit()
    finally:
        conn.close()


def _print_diff(proc: dict, new_steps: list[dict]) -> dict:
    """Print old→new completion_criteria + verifiability. Returns grade counts."""
    old = proc["old_steps"]
    counts = {"reliable": 0, "marginal": 0, "none": 0}
    print(f"\n{C['b']}{proc['id'][:8]}  {(proc['title'] or '')[:52]}{C['x']}")
    for s in new_steps:
        n = s["step_number"]
        nv = s.get("visual_verifiability", "reliable")
        counts[nv] = counts.get(nv, 0) + 1
        o = old.get(n, {})
        ov = o.get("visual_verifiability")
        flip = "" if ov == nv else f"  {C['y']}[{ov}→{nv}]{C['x']}"
        color = {"reliable": C["g"], "marginal": C["y"], "none": C["r"]}.get(nv, "")
        print(f"  step {n}  {color}{nv:8}{C['x']}{flip}")
        print(f"    {C['d']}old:{C['x']} {o.get('completion_criteria','')[:96]}")
        print(f"    {C['b']}new:{C['x']} {s.get('completion_criteria','')[:96]}")
    return counts


async def reextract_one(proc: dict, apply: bool) -> dict | None:
    source = Path(proc["source_video"])
    if not source.exists():
        print(f"  {C['y']}SKIP{C['x']} {proc['id'][:8]} — source video missing: {source.name}")
        return None
    try:
        out = await process_expert_video(str(source), GEMINI_API_KEY)
    except Exception as e:
        print(f"  {C['r']}ERROR{C['x']} {proc['id'][:8]} — extraction failed: {e}")
        return None

    duration = await probe_duration(str(source))
    steps = _clamp_steps(out["steps"], duration)
    counts = _print_diff(proc, steps)
    if not apply:
        return counts

    # re-cut clips
    clip_dir = CLIPS_DIR / proc["id"]
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    try:
        clip_paths, failed = await extract_clips(str(source), proc["id"], steps, CLIPS_DIR)
    except Exception as e:
        print(f"  {C['r']}ERROR{C['x']} clip extraction: {e}")
        clip_paths, failed = {}, [s["step_number"] for s in steps]
    failed_set = set(failed)

    step_models = []
    for s in steps:
        n = s["step_number"]
        ready = n not in failed_set
        step_models.append(ProcedureStep(
            step_number=n, title=s["title"], description=s["description"],
            timestamp_start=s["timestamp_start"], timestamp_end=s["timestamp_end"],
            tips=s.get("tips", []), warnings=s.get("warnings", []),
            completion_criteria=s.get("completion_criteria", ""),
            error_criteria=s.get("error_criteria", []),
            visual_verifiability=s.get("visual_verifiability", "reliable"),
            clip_url=(f"/api/clips/{proc['id']}/step_{n}.mp4" if ready else None),
            clip_status=("ready" if ready else "failed"),
        ))

    _delete_steps(proc["id"])
    procedure_dict = {
        "id": proc["id"], "title": out["title"], "description": out["description"],
        "total_duration": duration,
        "status": "completed_partial" if failed else "completed",
        "source_type": "video",
        "icon_symbol": out["icon_symbol"], "icon_emoji": out["icon_emoji"],
        "category": out["category"],
    }
    save_procedure(procedure_dict, [m.model_dump() for m in step_models])
    # invalidate harvested eval frames so harvest re-cuts from the new clips
    shutil.rmtree(EVALS_FRAMES / proc["id"], ignore_errors=True)
    print(f"  {C['g']}APPLIED{C['x']} {proc['id'][:8]} — {len(step_models)} steps, "
          f"{len(failed)} clip(s) failed")
    return counts


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default is dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only (the default; accepted for explicitness)")
    ap.add_argument("--procedure", help="Only this procedure id (prefix ok)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    procs = load_video_procedures()
    if args.procedure:
        procs = [p for p in procs if p["id"].startswith(args.procedure)]
    if args.limit:
        procs = procs[: args.limit]

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{C['b']}Re-extract ({mode}) — {len(procs)} video procedure(s){C['x']}")
    if not args.apply:
        print(f"{C['d']}No writes. Re-run with --apply to persist.{C['x']}")

    totals = {"reliable": 0, "marginal": 0, "none": 0}
    done = 0
    for p in procs:
        counts = await reextract_one(p, args.apply)
        if counts:
            done += 1
            for k in totals:
                totals[k] += counts.get(k, 0)

    print(f"\n{C['b']}{'='*60}{C['x']}")
    print(f"  procedures processed: {done}/{len(procs)}")
    print(f"  new grade totals: {C['g']}reliable={totals['reliable']}{C['x']}  "
          f"{C['y']}marginal={totals['marginal']}{C['x']}  {C['r']}none={totals['none']}{C['x']}")
    if not args.apply:
        print(f"  {C['b']}dry-run — nothing written.{C['x']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
