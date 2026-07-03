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
        [--no-parts] [--model MODEL] [--thinking-budget N]

By default the oracle receives each procedure's parts glossary (the production
path); `--no-parts` withholds it for the A/B. `--model` / `--thinking-budget`
override RETRACE_ORACLE_MODEL / 0 for candidate-model matrices; per-call
latency (p50/p95) is reported either way.

Positives use the ~90% frame, never the exact reference JPEG, so the oracle
can't win by matching identical bytes.
"""

import argparse
import asyncio
import statistics
import time

import _common as c

from routers.learner import _ORACLE_CONFIDENCE_FLOOR
from services.step_oracle import verify_step

CONCURRENCY = 5

# Set from CLI args in main(); threaded into every verify_step call so the
# whole eval runs under one candidate config.
ORACLE_KW: dict = {}


def _confident_complete(verdict) -> bool:
    """Mirror the endpoint: a 'complete' only advances above the floor."""
    return bool(verdict and verdict.complete and verdict.confidence >= _ORACLE_CONFIDENCE_FLOOR)


def _latency_line(samples_s: list[float]) -> str:
    if not samples_s:
        return "latency: n/a"
    p50 = statistics.median(samples_s)
    p95 = sorted(samples_s)[max(0, int(len(samples_s) * 0.95) - 1)]
    return f"latency p50={p50:.2f}s p95={p95:.2f}s (n={len(samples_s)})"


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
            glossary = p.get("parts") or None
            # Positive: near-done frame should read as complete.
            cases.append(dict(proc=p["id"], step=s, expect=True, kind="pos",
                              frame=frames["near_done"], ref=ref, glossary=glossary))
            # Negative A: the start frame should NOT read as complete.
            if "start" in frames:
                cases.append(dict(proc=p["id"], step=s, expect=False, kind="neg_start",
                                  frame=frames["start"], ref=ref, glossary=glossary))
            # Negative B: the prior step's done frame (this step not yet begun).
            prev = steps.get(n - 1)
            if prev is not None:
                prev_frames = c.harvest_step_frames(p["id"], n - 1, c.clip_path_for(p["id"], n - 1))
                if "done" in prev_frames:
                    cases.append(dict(proc=p["id"], step=s, expect=False, kind="neg_prevdone",
                                      frame=prev_frames["done"], ref=ref, glossary=glossary))
    return cases


async def _run_mode(cases: list[dict], use_reference: bool, log: c.EventLog) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[dict] = []

    latencies: list[float] = []

    async def _one(case: dict):
        async with sem:
            current = case["frame"].read_bytes()
            ref = case["ref"] if use_reference else None
            t0 = time.monotonic()
            verdict = await verify_step(current, case["step"], ref,
                                        case.get("glossary"), **ORACLE_KW)
            latencies.append(time.monotonic() - t0)
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
            "recall": recall, "fp_rate": fp_rate, "rows": results,
            "latencies": latencies}


async def _run_interject(procs: list[dict], log: c.EventLog) -> dict:
    """Run the oracle on correct-execution frames; any error flag is a FP."""
    sem = asyncio.Semaphore(CONCURRENCY)
    flagged = total = 0
    hits = []
    lock = asyncio.Lock()

    async def _one(proc_id, step, frame_path, glossary):
        nonlocal flagged, total
        async with sem:
            verdict = await verify_step(frame_path.read_bytes(), step, None,
                                        glossary, **ORACLE_KW)
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
                    tasks.append(_one(p["id"], s, frames[label], p.get("parts") or None))
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
    ap.add_argument("--no-parts", action="store_true",
                    help="Withhold the parts glossary from the oracle (A/B)")
    ap.add_argument("--model", help="Override RETRACE_ORACLE_MODEL for every call")
    ap.add_argument("--thinking-budget", type=int,
                    help="Override the oracle's thinking budget (default 0)")
    args = ap.parse_args()

    if args.model:
        ORACLE_KW["model"] = args.model
    if args.thinking_budget is not None:
        ORACLE_KW["thinking_budget"] = args.thinking_budget

    procs = c.procedures_with_clips()
    if args.no_parts:
        procs = [{**p, "parts": []} for p in procs]
    if args.procedure:
        procs = [p for p in procs if any(p["id"].startswith(x) for x in args.procedure)]
    if args.smoke:
        procs = procs[:2]
    elif args.limit:
        procs = procs[: args.limit]

    cases = _build_cases(procs)
    log = c.EventLog("eval_oracle")
    c.header(f"Oracle accuracy — {len(procs)} procedure(s), {len(cases)} labeled frame-cases")
    from config import RETRACE_ORACLE_MODEL as _default_model
    print(f"  model = {ORACLE_KW.get('model') or _default_model}"
          f"  thinking_budget = {ORACLE_KW.get('thinking_budget', 0)}"
          f"  parts glossary = {'OFF' if args.no_parts else 'on'}")
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
              f"{c._C['bold']}{stats['fp_rate']*100:.1f}%{c._C['reset']}"
              f"    {_latency_line(stats['latencies'])}")

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
