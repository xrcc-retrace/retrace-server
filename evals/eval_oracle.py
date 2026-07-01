"""Oracle accuracy eval — scores services.step_oracle.verify_step directly.

No server, no Gemini Live. Uses the self-supervised labels harvested from the
expert clips: a step's last frame IS "done" (positive), its first frame and the
prior step's last frame are "not done" (negative). We feed those to the oracle
and measure how often it fires "complete" correctly — with the headline metric
being the FALSE-POSITIVE (early-advance) rate, the expensive error.

Runs each case in two grounding modes: image+text (reference done-frame given,
the production path for video steps) and text-only (no reference, the
manual/web path). Also measures the interject false-positive rate: how often the
oracle flags an error on a frame where the learner is doing it correctly.

    python evals/eval_oracle.py [--limit N] [--grounding both|image|text] [--smoke]

Positives use the ~90% frame, never the exact reference JPEG, so the oracle
can't win by matching identical bytes.
"""

import argparse
import asyncio

import _common as c

from routers.learner import _ORACLE_CONFIDENCE_FLOOR
from services.step_oracle import verify_step

CONCURRENCY = 5


def _confident_complete(verdict) -> bool:
    """Mirror the endpoint: a 'complete' only advances above the floor."""
    return bool(verdict and verdict.complete and verdict.confidence >= _ORACLE_CONFIDENCE_FLOOR)


def _build_cases(procs: list[dict]) -> list[dict]:
    """One row per (step, frame-under-test) with its expected label."""
    cases = []
    for p in procs:
        steps = {s["step_number"]: s for s in p["steps"]}
        for s in p["steps"]:
            n = s["step_number"]
            if s.get("visual_verifiability") == "none":
                continue  # verbal-only steps are never oracle-advanced
            clip = c.clip_path_for(p["id"], n)
            if not clip.exists():
                continue
            frames = c.harvest_step_frames(p["id"], n, clip)
            if "done" not in frames or "near_done" not in frames:
                continue
            ref = frames["done"].read_bytes()
            # Positive: near-done frame should read as complete.
            cases.append(dict(proc=p["id"], step=s, expect=True, kind="pos",
                              frame=frames["near_done"], ref=ref))
            # Negative A: the start frame should NOT read as complete.
            if "start" in frames:
                cases.append(dict(proc=p["id"], step=s, expect=False, kind="neg_start",
                                  frame=frames["start"], ref=ref))
            # Negative B: the prior step's done frame (this step not yet begun).
            prev = steps.get(n - 1)
            if prev is not None:
                prev_frames = c.harvest_step_frames(p["id"], n - 1, c.clip_path_for(p["id"], n - 1))
                if "done" in prev_frames:
                    cases.append(dict(proc=p["id"], step=s, expect=False, kind="neg_prevdone",
                                      frame=prev_frames["done"], ref=ref))
    return cases


async def _run_mode(cases: list[dict], use_reference: bool, log: c.EventLog) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = []

    async def _one(case: dict):
        async with sem:
            current = case["frame"].read_bytes()
            ref = case["ref"] if use_reference else None
            verdict = await verify_step(current, case["step"], ref)
        predicted = _confident_complete(verdict)
        row = {
            "proc": case["proc"], "step": case["step"]["step_number"],
            "kind": case["kind"], "expect": case["expect"], "predicted": predicted,
            "confidence": (verdict.confidence if verdict else None),
            "error_index": (verdict.error_index if verdict else None),
            "verifiability": case["step"].get("visual_verifiability"),
            "frame": case["frame"].name,
            "criterion": (case["step"].get("completion_criteria") or "")[:80],
            "mode": "image+text" if use_reference else "text-only",
        }
        log.write(**row)
        results.append(row)

    await asyncio.gather(*(_one(c_) for c_ in cases))

    tp = sum(1 for r in results if r["expect"] and r["predicted"])
    fn = sum(1 for r in results if r["expect"] and not r["predicted"])
    fp = sum(1 for r in results if not r["expect"] and r["predicted"])
    tn = sum(1 for r in results if not r["expect"] and not r["predicted"])
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision,
            "recall": recall, "fp_rate": fp_rate, "rows": results}


async def _run_interject(procs: list[dict], log: c.EventLog) -> dict:
    """Run the oracle on correct-execution frames; any error flag is a FP."""
    sem = asyncio.Semaphore(CONCURRENCY)
    flagged = total = 0
    hits = []
    lock = asyncio.Lock()

    async def _one(proc_id, step, frame_path):
        nonlocal flagged, total
        async with sem:
            verdict = await verify_step(frame_path.read_bytes(), step, None)
        async with lock:
            total += 1
            if verdict and verdict.error_index is not None:
                flagged += 1
                errs = step.get("error_criteria") or []
                idx = verdict.error_index
                hits.append({"proc": proc_id, "step": step["step_number"],
                             "frame": frame_path.name,
                             "flagged_error": errs[idx] if 0 <= idx < len(errs) else f"idx{idx}"})
                log.write(interject_fp=True, proc=proc_id, step=step["step_number"],
                          frame=frame_path.name, error_index=idx)

    tasks = []
    for p in procs:
        for s in p["steps"]:
            if not (s.get("error_criteria") or []):
                continue
            clip = c.clip_path_for(p["id"], s["step_number"])
            if not clip.exists():
                continue
            frames = c.harvest_step_frames(p["id"], s["step_number"], clip)
            for label in ("start", "mid", "near_done"):
                if label in frames:
                    tasks.append(_one(p["id"], s, frames[label]))
    await asyncio.gather(*tasks)
    return {"flagged": flagged, "total": total,
            "fp_rate": (flagged / total if total else 0.0), "hits": hits}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="Cap number of procedures")
    ap.add_argument("--grounding", choices=["both", "image", "text"], default="both")
    ap.add_argument("--smoke", action="store_true", help="Just 2 procedures")
    ap.add_argument("--procedure", action="append",
                    help="Only these procedure id(s) (prefix ok, repeatable)")
    args = ap.parse_args()

    procs = c.procedures_with_clips()
    if args.procedure:
        procs = [p for p in procs if any(p["id"].startswith(x) for x in args.procedure)]
    if args.smoke:
        procs = procs[:2]
    elif args.limit:
        procs = procs[: args.limit]

    cases = _build_cases(procs)
    log = c.EventLog("eval_oracle")
    c.header(f"Oracle accuracy — {len(procs)} procedure(s), {len(cases)} labeled frame-cases")
    print(f"  confidence floor = {_ORACLE_CONFIDENCE_FLOOR}  (matches /verify)\n")

    modes = []
    if args.grounding in ("both", "image"):
        modes.append(("image+text", True))
    if args.grounding in ("both", "text"):
        modes.append(("text-only", False))

    summary = {}
    for name, use_ref in modes:
        stats = await _run_mode(cases, use_ref, log)
        summary[name] = stats
        print(f"\n  {c._C['bold']}{name}{c._C['reset']}  "
              f"(TP={stats['tp']} FP={stats['fp']} TN={stats['tn']} FN={stats['fn']})")
        c.print_gate(f"{name} precision (few false advances)", stats["precision"])
        c.print_gate(f"{name} recall (catches real completions)", stats["recall"])
        print(f"    false-positive (early-advance) rate: "
              f"{c._C['bold']}{stats['fp_rate']*100:.1f}%{c._C['reset']}")

    inter = await _run_interject(procs, log)
    print()
    c.header("Interject false-positive rate (errors flagged on correct frames)")
    # invert: we want a LOW fp_rate, so gate on (1 - fp_rate)
    c.print_gate("clean frames left un-flagged", 1.0 - inter["fp_rate"],
                 f"{inter['flagged']}/{inter['total']} frames wrongly flagged")

    # False-positive dump — the only things a human needs to eyeball.
    header_printed = False
    for name, stats in summary.items():
        fps = [r for r in stats["rows"] if not r["expect"] and r["predicted"]]
        if fps and not header_printed:
            c.header("FALSE POSITIVES (oracle said complete on a not-done frame)")
            header_printed = True
        for r in fps:
            print(f"  [{name}] {r['proc'][:8]} step {r['step']} {r['kind']} "
                  f"conf={r['confidence']} — {r['criterion']}  ({r['frame']})")
    if inter["hits"]:
        c.header("INTERJECT FALSE POSITIVES")
        for h in inter["hits"]:
            print(f"  {h['proc'][:8]} step {h['step']} ({h['frame']}) → {h['flagged_error']}")

    log.close()
    print(f"\n  log: {log.path}")


if __name__ == "__main__":
    asyncio.run(main())
