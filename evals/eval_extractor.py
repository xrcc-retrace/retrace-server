"""Extractor quality eval — structural lint + closed-loop oracle consistency.

Two data sources:
  default        read procedures already in the DB (fast, no Gemini).
  --reextract    run process_expert_video on test_data/*.{mp4,mov,MOV},
                 exercising the LIVE extractor prompt end-to-end.

Checks per step:
  1. Structural lint (no LLM): verifiability in-enum; error_criteria <=3 &
     distinct & free of judgment/cognitive words; completion_criteria a single
     spatial sentence with no counting/temporal tokens on reliable/marginal.
  2. Closed-loop consistency: for each reliable/marginal step, the oracle is
     run on the step's OWN near-done frame with its OWN done-frame reference —
     it SHOULD confirm complete. A step graded reliable that the oracle can't
     confirm on its canonical done-frame is flagged (bad criterion or wrong
     grade) — this is exactly how legacy "what-changed" criteria surface.
  3. --judge (optional): a Flash LLM-judge rates criterion/grade/errors per step.

    python evals/eval_extractor.py [--reextract] [--judge] [--limit N]
"""

import argparse
import asyncio
import re
from pathlib import Path

import _common as c

from config import GEMINI_API_KEY
from routers.learner import _ORACLE_CONFIDENCE_FLOOR
from services.step_oracle import verify_step

# Judgment/cognitive words the extraction prompt forbids in error_criteria.
FORBIDDEN = [
    "careless", "sloppy", "wrong-ish", "seems confused", "confused",
    "distracted", "not paying attention", "technique is off", "something is off",
    "not quite right", "seems to", "appears confused",
]
# Counting/temporal tokens that make a criterion non-single-frame-verifiable.
TEMPORAL = ["times", "seconds", "minutes", "count", "number of", "for a while",
            "repeatedly", "video ends", "video begins"]


def lint_step(step: dict) -> list[str]:
    """Return a list of structural problems (empty = clean)."""
    problems = []
    v = step.get("visual_verifiability")
    if v not in ("reliable", "marginal", "none", None):
        problems.append(f"verifiability not in enum: {v!r}")

    errs = step.get("error_criteria") or []
    if len(errs) > 3:
        problems.append(f"error_criteria has {len(errs)} (>3)")
    lowered = [e.lower() for e in errs]
    if len(set(lowered)) < len(lowered):
        problems.append("error_criteria has duplicates")
    for e in errs:
        for w in FORBIDDEN:
            if w in e.lower():
                problems.append(f"error_criteria judgment word '{w}': {e[:50]}")

    cc = (step.get("completion_criteria") or "").strip()
    if v in ("reliable", "marginal", None):
        if not cc:
            problems.append("completion_criteria empty on a verifiable step")
        for w in TEMPORAL:
            if w in cc.lower():
                problems.append(f"completion_criteria temporal/counting token '{w}': {cc[:50]}")
        # crude single-sentence check: at most one terminal period mid-string
        if cc.count(".") > 1 or "\n" in cc:
            problems.append(f"completion_criteria not a single sentence: {cc[:50]}")
    return problems


# --- frame sourcing ----------------------------------------------------------

def _frames_from_clip(proc_id: str, step: dict) -> dict | None:
    clip = c.clip_path_for(proc_id, step["step_number"])
    if not clip.exists():
        return None
    f = c.harvest_step_frames(proc_id, step["step_number"], clip)
    return f if "done" in f and "near_done" in f else None


def _frames_from_source(video_path: Path, step: dict, tmp_dir: Path) -> dict | None:
    """Cut done/near_done frames straight from a source video via timestamps
    (used for --reextract, where clips haven't been generated yet)."""
    end = float(step.get("timestamp_end") or 0.0)
    start = float(step.get("timestamp_start") or 0.0)
    if end <= 0:
        return None
    near = max(start, end - 0.4)
    done_p = tmp_dir / f"step_{step['step_number']}_done.jpg"
    near_p = tmp_dir / f"step_{step['step_number']}_near.jpg"
    ok_done = c._extract_frame_at(video_path, max(0.0, end - 0.1), done_p)
    ok_near = c._extract_frame_at(video_path, near, near_p)
    if not (ok_done and ok_near):
        return None
    return {"done": done_p, "near_done": near_p}


# --- closed-loop consistency -------------------------------------------------

async def consistency_check(steps_with_frames: list[tuple], log: c.EventLog) -> dict:
    """steps_with_frames: list of (proc_id, step, frames_dict).

    Reports TWO sub-metrics per step (they diverge, and the gap is itself the
    signal):
      - text_ok: criterion coherence — the true done frame text-only against the
        criterion. Fails when the criterion is a hidden delta ("has been eaten")
        or a marginal detail no single frame shows.
      - img_ok: production confirmability — the ~90% frame vs the done frame as
        reference (exactly how the /verify endpoint grounds it).
    A step counts as a consistency miss when BOTH fail.
    """
    sem = asyncio.Semaphore(5)
    text_pass = img_pass = both_pass = checked = 0
    misses = []
    lock = asyncio.Lock()

    def _confident(v):
        return bool(v and v.complete and v.confidence >= _ORACLE_CONFIDENCE_FLOOR)

    async def _one(proc_id, step, frames):
        nonlocal text_pass, img_pass, both_pass, checked
        if step.get("visual_verifiability") == "none":
            return
        async with sem:
            v_text = await verify_step(frames["done"].read_bytes(), step, None)
            v_img = await verify_step(
                frames["near_done"].read_bytes(), step, frames["done"].read_bytes())
        t_ok, i_ok = _confident(v_text), _confident(v_img)
        async with lock:
            checked += 1
            text_pass += int(t_ok)
            img_pass += int(i_ok)
            both_pass += int(t_ok or i_ok)  # confirmable by at least one grounding
            if not (t_ok or i_ok):
                misses.append({
                    "proc": proc_id, "step": step["step_number"],
                    "verifiability": step.get("visual_verifiability"),
                    "criterion": (step.get("completion_criteria") or "")[:90],
                })
                log.write(consistency_miss=True, proc=proc_id, step=step["step_number"],
                          criterion=(step.get("completion_criteria") or "")[:120])

    await asyncio.gather(*(_one(pid, s, f) for pid, s, f in steps_with_frames))
    return {"checked": checked, "misses": misses,
            "text_rate": (text_pass / checked if checked else 0.0),
            "img_rate": (img_pass / checked if checked else 0.0),
            "rate": (both_pass / checked if checked else 0.0)}


# --- optional LLM judge ------------------------------------------------------

async def judge_steps(steps_with_frames: list[tuple]) -> dict:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class _Judge(BaseModel):
        criterion_is_spatial_endstate: bool
        grade_is_appropriate: bool
        errors_observable_and_distinct: bool
        note: str = ""

    client = genai.Client(api_key=GEMINI_API_KEY)
    sys = ("You audit a coaching step's metadata against a photo of the step's "
           "completed state. Judge strictly. criterion_is_spatial_endstate: is "
           "completion_criteria a single-frame spatial end-state visible in the "
           "DONE image (not an action/temporal/counting statement)? "
           "grade_is_appropriate: does visual_verifiability (reliable/marginal/none) "
           "match how detectable that end-state is? errors_observable_and_distinct: "
           "are error_criteria visible wrong-states, mutually distinct? Output JSON.")
    sem = asyncio.Semaphore(4)
    scores = {"criterion": 0, "grade": 0, "errors": 0, "n": 0}
    notes = []
    lock = asyncio.Lock()

    async def _one(proc_id, step, frames):
        text = (f"completion_criteria: {step.get('completion_criteria')!r}\n"
                f"visual_verifiability: {step.get('visual_verifiability')!r}\n"
                f"error_criteria: {step.get('error_criteria')}")
        parts = [types.Part.from_text(text="DONE image (step complete):"),
                 types.Part.from_bytes(data=frames["done"].read_bytes(), mime_type="image/jpeg"),
                 types.Part.from_text(text=text)]

        def _call():
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=sys, response_mime_type="application/json",
                    response_schema=_Judge,
                    http_options=types.HttpOptions(timeout=20_000)),
            )
        try:
            async with sem:
                resp = await asyncio.to_thread(_call)
            j = _Judge.model_validate_json(resp.text)
        except Exception:
            return
        async with lock:
            scores["n"] += 1
            scores["criterion"] += int(j.criterion_is_spatial_endstate)
            scores["grade"] += int(j.grade_is_appropriate)
            scores["errors"] += int(j.errors_observable_and_distinct)
            if not (j.criterion_is_spatial_endstate and j.grade_is_appropriate):
                notes.append(f"{proc_id[:8]} step {step['step_number']}: {j.note[:80]}")

    await asyncio.gather(*(_one(pid, s, f) for pid, s, f in steps_with_frames))
    return {"scores": scores, "notes": notes}


# --- data gathering ----------------------------------------------------------

async def _gather_db(limit) -> list[dict]:
    procs = c.procedures_with_clips()
    return procs[:limit] if limit else procs


async def _gather_reextract(limit) -> list[dict]:
    from services.gemini_processor import process_expert_video
    test_dir = c._BACKEND_ROOT / "test_data"
    vids = [p for p in sorted(test_dir.iterdir())
            if p.suffix.lower() in (".mp4", ".mov", ".m4v")]
    if limit:
        vids = vids[:limit]
    procs = []
    for v in vids:
        print(f"  extracting {v.name} ...")
        try:
            out = await process_expert_video(str(v), GEMINI_API_KEY)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        out["_source_video"] = str(v)
        out["id"] = v.stem  # synthetic id for reporting
        procs.append(out)
    return procs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true",
                    help="Run the live extractor on test_data instead of reading the DB")
    ap.add_argument("--judge", action="store_true", help="Add the Flash LLM-judge pass")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procedure", action="append",
                    help="Only these procedure id(s) (prefix ok, repeatable; DB mode only)")
    args = ap.parse_args()

    log = c.EventLog("eval_extractor")
    procs = await (_gather_reextract(args.limit) if args.reextract else _gather_db(args.limit))
    if args.procedure and not args.reextract:
        procs = [p for p in procs if any(p["id"].startswith(x) for x in args.procedure)]
    c.header(f"Extractor quality — {len(procs)} procedure(s) "
             f"({'re-extracted' if args.reextract else 'from DB'})")

    # 1) structural lint
    total_steps = clean_steps = 0
    lint_hits = []
    for p in procs:
        for s in p["steps"]:
            total_steps += 1
            probs = lint_step(s)
            if probs:
                lint_hits.append((p["id"], s["step_number"], probs))
                log.write(lint_fail=True, proc=p["id"], step=s["step_number"], problems=probs)
            else:
                clean_steps += 1
    c.print_gate("steps passing structural lint", clean_steps / total_steps if total_steps else 0.0,
                 f"{clean_steps}/{total_steps}")

    # 2) closed-loop consistency (needs frames)
    tmp = c.EVALS_DIR / "fixtures" / "reextract_frames"
    steps_with_frames = []
    for p in procs:
        if args.reextract:
            tmp_p = tmp / p["id"]
            tmp_p.mkdir(parents=True, exist_ok=True)
            src = Path(p["_source_video"])
            for s in p["steps"]:
                f = _frames_from_source(src, s, tmp_p)
                if f:
                    steps_with_frames.append((p["id"], s, f))
        else:
            for s in p["steps"]:
                f = _frames_from_clip(p["id"], s)
                if f:
                    steps_with_frames.append((p["id"], s, f))

    cons = await consistency_check(steps_with_frames, log)
    c.print_gate("steps oracle confirms (either grounding)", cons["rate"],
                 f"of {cons['checked']}")
    print(f"    criterion coherence (text-only)     : {cons['text_rate']*100:5.1f}%")
    print(f"    production confirmable (image+text) : {cons['img_rate']*100:5.1f}%")

    # 3) optional judge
    judge = None
    if args.judge and steps_with_frames:
        judge = await judge_steps(steps_with_frames)
        n = judge["scores"]["n"] or 1
        c.header("LLM-judge (Flash)")
        c.print_gate("criterion is a single-frame spatial end-state", judge["scores"]["criterion"] / n)
        c.print_gate("verifiability grade is appropriate", judge["scores"]["grade"] / n)
        c.print_gate("error_criteria observable & distinct", judge["scores"]["errors"] / n)

    # dumps — only the failures a human should read
    if lint_hits:
        c.header("LINT FAILURES")
        for pid, sn, probs in lint_hits:
            print(f"  {pid[:8]} step {sn}:")
            for pr in probs:
                print(f"      - {pr}")
    if cons["misses"]:
        c.header("CONSISTENCY MISSES (graded verifiable but oracle can't confirm — re-extract candidates)")
        for m in cons["misses"]:
            print(f"  {m['proc'][:8]} step {m['step']} [{m['verifiability']}] — {m['criterion']}")
    if judge and judge["notes"]:
        c.header("JUDGE NOTES")
        for nline in judge["notes"]:
            print(f"  {nline}")

    log.close()
    print(f"\n  log: {log.path}")


if __name__ == "__main__":
    asyncio.run(main())
