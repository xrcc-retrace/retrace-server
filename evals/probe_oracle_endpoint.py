"""Endpoint state-machine probe for POST /api/learner/session/{id}/verify.

Validates the gating/dwell/dedup logic that lives in the ROUTER (not in
verify_step): the toggle gate (skip when active_monitoring off / verbal-only
step), the negative→pending / positive→advanced transitions, and that a
confirmed completion actually advances the server-side session.

Needs a running server (`python main.py`). Oracle-dependent assertions target a
step the oracle actually confirms offline first, so the state-machine test is
deterministic even on a legacy corpus.

    python evals/probe_oracle_endpoint.py [--server http://localhost:8000]
"""

import argparse
import asyncio

import httpx

import _common as c

from routers.learner import _ORACLE_CONFIDENCE_FLOOR
from services.step_oracle import verify_step

PASS, FAIL = f"{c._C['pass']}PASS{c._C['reset']}", f"{c._C['fail']}FAIL{c._C['reset']}"
SKIP = f"{c._C['marginal']}SKIP{c._C['reset']}"


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {PASS if ok else FAIL}  {name}{('  — ' + detail) if detail else ''}")
    return ok


async def _find_confirmable_step() -> tuple | None:
    """Find (procedure, step, near_done_path) the oracle confirms offline."""
    for p in c.procedures_with_clips():
        for s in p["steps"]:
            if s.get("visual_verifiability") == "none":
                continue
            clip = c.clip_path_for(p["id"], s["step_number"])
            if not clip.exists():
                continue
            frames = c.harvest_step_frames(p["id"], s["step_number"], clip)
            if "near_done" not in frames or "done" not in frames:
                continue
            verdict = await verify_step(
                frames["near_done"].read_bytes(), s, frames["done"].read_bytes())
            if verdict and verdict.complete and verdict.confidence >= _ORACLE_CONFIDENCE_FLOOR:
                return p, s, frames
    return None


async def _start(client, base, procedure_id, **kw) -> str | None:
    body = {"procedure_id": procedure_id, **kw}
    r = await client.post(f"{base}/api/learner/session/start", json=body)
    if r.status_code != 200:
        return None
    return r.json()["session_id"]


async def _verify(client, base, sid, frame_path) -> dict:
    files = {"frame": ("frame.jpg", frame_path.read_bytes(), "image/jpeg")}
    r = await client.post(f"{base}/api/learner/session/{sid}/verify", files=files)
    return r.json()


async def _state(client, base, sid) -> dict:
    r = await client.get(f"{base}/api/learner/session/{sid}/state")
    return r.json()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    args = ap.parse_args()
    base = args.server.rstrip("/")

    c.header(f"Oracle endpoint probe — {base}")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            h = await client.get(f"{base}/health")
            assert h.status_code == 200
        except Exception:
            print(f"  {FAIL}  server not reachable at {base} — start it with `python main.py`")
            return

        found = await _find_confirmable_step()
        if not found:
            print(f"  {SKIP}  no oracle-confirmable step in the corpus "
                  "(legacy criteria) — run eval_extractor --reextract first for a clean target")
            return
        p, step, frames = found
        n = step["step_number"]
        print(f"  target: {p['id'][:8]} step {n} — {step['title'][:50]}\n")

        results = []

        # 1) master switch: active_monitoring off → skip regardless of frame
        sid = await _start(client, base, p["id"], active_monitoring=False,
                           auto_advance=False, starting_step=n)
        v = await _verify(client, base, sid, frames["near_done"])
        results.append(_check("active_monitoring=off → skip", v.get("status") == "skip",
                              f'got {v.get("status")}'))
        await client.delete(f"{base}/api/learner/session/{sid}")

        # 2) negative frame → not advanced (pending or error, never advanced)
        sid = await _start(client, base, p["id"], active_monitoring=True,
                           auto_advance=True, starting_step=n)
        neg = frames.get("start", frames["near_done"])
        v = await _verify(client, base, sid, neg)
        results.append(_check("start frame → not advanced",
                              v.get("status") in ("pending", "error"),
                              f'got {v.get("status")}'))

        # 3) positive frame → advanced (reliable dwell=1), session step increments
        before = (await _state(client, base, sid)).get("current_step")
        v = await _verify(client, base, sid, frames["near_done"])
        after = (await _state(client, base, sid)).get("current_step")
        advanced = v.get("status") in ("advanced", "completed")
        results.append(_check("near-done frame → advanced", advanced, f'got {v.get("status")}'))
        if v.get("status") == "advanced":
            results.append(_check("server current_step incremented",
                                  after == (before or n) + 1, f"{before}→{after}"))
        await client.delete(f"{base}/api/learner/session/{sid}")

    print()
    rate = sum(results) / len(results) if results else 0.0
    c.print_gate("endpoint state-machine checks", rate, f"{sum(results)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
