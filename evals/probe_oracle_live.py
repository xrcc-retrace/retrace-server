"""Live-behavior probe for the new injected-event protocol (spends tokens).

The oracle moved visual advance + interjection OUT of Gemini Live. Live now only
speaks on a real event the phone injects as an `[ADVANCE]` / `[CORRECTION]` text
turn — and, critically, stays SILENT otherwise (the regression that the old
`[check]` heartbeat hallucination is gone).

Uses the production ephemeral-token + Constrained-WebSocket path (same as
probe_tool_reliability) with the CURRENT build_system_prompt. Three checks:
  1. inject [ADVANCE] → Live speaks (congrats + coaches the named next step),
     without reading the bracket tag aloud.
  2. inject [CORRECTION] → Live speaks one short corrective line.
  3. idle (no event, no user speech) → Live stays SILENT.

    python evals/probe_oracle_live.py [--idle 15]
"""

import argparse
import asyncio
import json

import _common as c  # noqa: F401  (bootstraps sys.path + .env)

from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import GEMINI_CONFIG, TOOL_DEFINITIONS, build_system_prompt

PROCEDURE = {
    "id": "probe-live",
    "title": "Reassemble a water bottle lid",
    "description": "Put the lid back together and onto the bottle.",
    "steps": [
        {"step_number": 1, "title": "Seat the gasket in the lid",
         "description": "Press the rubber gasket into the groove on the underside of the lid.",
         "completion_criteria": "The gasket is seated flush in the lid groove.",
         "error_criteria": ["The gasket is lying next to the lid, not inserted."],
         "visual_verifiability": "reliable", "tips": [], "warnings": []},
        {"step_number": 2, "title": "Screw the lid onto the bottle",
         "description": "Thread the lid clockwise onto the bottle until snug.",
         "completion_criteria": "The lid is screwed fully onto the bottle.",
         "error_criteria": [], "visual_verifiability": "reliable", "tips": [], "warnings": []},
    ],
}


async def _connect_and(prompt: str):
    """Mint token, open WS, return the live ws context (caller drives)."""
    token = await create_ephemeral_token(
        system_instruction=prompt, tools=TOOL_DEFINITIONS,
        model=GEMINI_CONFIG["model"], voice="Puck",
    )
    import websockets
    ws = await websockets.connect(token["websocket_url"])
    await ws.send('{"setup":{}}')
    async with asyncio.timeout(6.0):
        async for raw in ws:
            text = raw.decode() if isinstance(raw, bytes) else raw
            if '"setupComplete"' in text:
                return ws
    await ws.close()
    return None


async def _collect(ws, timeout: float) -> str:
    """Accumulate output transcription until turnComplete or timeout."""
    parts = []
    try:
        async with asyncio.timeout(timeout):
            async for raw in ws:
                text = raw.decode() if isinstance(raw, bytes) else raw
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                sc = obj.get("serverContent", {})
                if isinstance(sc, dict):
                    ot = sc.get("outputTranscription", {})
                    if isinstance(ot, dict) and ot.get("text"):
                        parts.append(ot["text"])
                    if sc.get("turnComplete"):
                        break
    except asyncio.TimeoutError:
        pass
    return "".join(parts).strip()


async def _inject_check(prompt, event_text, label, expect_speech, idle=12.0) -> tuple[bool, str]:
    ws = await _connect_and(prompt)
    if ws is None:
        return False, "no setupComplete"
    try:
        if event_text is not None:
            await ws.send(json.dumps({"realtimeInput": {"text": event_text}}))
        transcript = await _collect(ws, idle)
    finally:
        await ws.close()

    spoke = bool(transcript)
    tag_leaked = ("[ADVANCE]" in transcript.upper()) or ("[CORRECTION]" in transcript.upper())
    if expect_speech:
        ok = spoke and not tag_leaked
    else:
        ok = not spoke
    return ok, transcript[:160].replace("\n", " ")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=15.0, help="Silence-test wait (s)")
    args = ap.parse_args()

    prompt = build_system_prompt(PROCEDURE, auto_advance=True, active_monitoring=True,
                                 current_step_number=1)
    c.header(f"Live injected-event probe — {GEMINI_CONFIG['model']}")

    checks = []

    ok, tx = await _inject_check(
        prompt,
        "[ADVANCE] The learner just finished this step. Next step: Screw the lid onto the "
        "bottle — thread it clockwise until snug. Congratulate briefly, then coach it.",
        "ADVANCE", expect_speech=True)
    checks.append(("[ADVANCE] → speaks, no tag leak", ok))
    print(f"  {'PASS' if ok else 'FAIL'}  [ADVANCE] → speaks; tag not read aloud")
    print(f"        said: {c._C['dim']}{tx or '(silence)'}{c._C['reset']}")

    ok, tx = await _inject_check(
        prompt,
        "[CORRECTION] The gasket is lying next to the lid, not inserted. Tell them concisely.",
        "CORRECTION", expect_speech=True)
    checks.append(("[CORRECTION] → speaks", ok))
    print(f"  {'PASS' if ok else 'FAIL'}  [CORRECTION] → short corrective line")
    print(f"        said: {c._C['dim']}{tx or '(silence)'}{c._C['reset']}")

    ok, tx = await _inject_check(prompt, None, "IDLE", expect_speech=False, idle=args.idle)
    checks.append(("idle → silence (no hallucination)", ok))
    print(f"  {'PASS' if ok else 'FAIL'}  idle {args.idle:.0f}s → SILENCE (heartbeat-hallucination gone)")
    if tx:
        print(f"        LEAKED: {c._C['fail']}{tx}{c._C['reset']}")

    print()
    rate = sum(1 for _, o in checks if o) / len(checks)
    c.print_gate("live injected-event behavior", rate,
                 f"{sum(1 for _, o in checks if o)}/{len(checks)}")


if __name__ == "__main__":
    asyncio.run(main())
