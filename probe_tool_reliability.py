"""Measure advance_step tool-call reliability on native-audio 12-2025.

STALE PREMISE (2026-07-02): advance_step was removed from Live's toolset —
step advancement is oracle-owned via the client-intercepted `check_step`
tool (claim="done"|"question"). This probe still runs (the token path and
prompt builder are alive) but it measures a tool the production prompt no
longer declares. Follow-up before trusting its numbers: retarget HIT
detection to `check_step` and score claim-correctness (done-phrases →
claim="done", question-phrases → claim="question").

Exercises the EXACT production path (ephemeral token minted via
services.ephemeral_token + WebSocket connect via Constrained endpoint) that
iOS uses. Each trial sends a short user turn that should trigger the tool,
waits up to TIMEOUT, and records HIT (tool-call fired) or MISS
(model replied verbally without dispatching the tool).

Reports a rate so we can decide:
  - HIT rate ≥ ~80%: tool calling is reliable enough for the hackathon
  - HIT rate ≤ ~20%: the documented native-audio race condition is biting
                     us; time for the out-of-band transcript fallback.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import GEMINI_CONFIG, TOOL_DEFINITIONS, build_system_prompt


# Mirror the EXACT production auto_advance=False prompt the iOS app actually
# sends. Fabricate a minimal 3-step procedure so build_system_prompt has
# enough structure; the phrasing around "call advance_step" is what we care
# about reproducing faithfully.
FAKE_PROCEDURE = {
    "title": "Remove the barbell collar",
    "description": "Take the collar off the left end of the bar.",
    "steps": [
        {
            "step_number": 1,
            "title": "Turn the collar counter-clockwise",
            "description": "Rotate the collar counter-clockwise until the threads disengage.",
            "completion_criteria": "The collar is completely free of the bar threads.",
            "tips": ["Use both hands if it's tight."],
            "warnings": [],
        },
        {"step_number": 2, "title": "Slide it off", "description": "Pull the collar off the end of the bar.",
         "completion_criteria": "Collar is no longer on the bar.", "tips": [], "warnings": []},
        {"step_number": 3, "title": "Set it aside", "description": "Place the collar on the rack.",
         "completion_criteria": "Collar is resting on the rack.", "tips": [], "warnings": []},
    ],
}

# Toggle between manual (auto_advance=False → model SHOULD fire on "done")
# and auto (auto_advance=True → prompt tells model NOT to fire on verbal cues
# and wait for visual confirmation instead). Flip via env var so one file
# covers both modes.
AUTO_ADVANCE = os.environ.get("PROBE_AUTO_ADVANCE", "false").lower() == "true"
SYSTEM_PROMPT = build_system_prompt(
    FAKE_PROCEDURE,
    auto_advance=AUTO_ADVANCE,
    current_step_number=1,
)

N_TRIALS = 10
TIMEOUT_SEC = 12.0

# Phrases that should all map to advance_step, cycled across trials so we don't
# just test one exact utterance.
UTTERANCES = [
    "done",
    "I'm finished with this step",
    "next",
    "I did it",
    "ok, moving on",
    "done with step one",
    "finished",
    "next please",
    "I'm done",
    "step complete",
]


async def run_trial(trial_n: int, utterance: str) -> dict:
    """One connect → mint token → open WS → send utterance → wait for either
    a toolCall or a turnComplete (meaning the model spoke without dispatching).
    Returns {"hit": bool, "reason": str, "transcript": str}.
    """
    # Mint a fresh ephemeral token via production code — same path iOS uses.
    token_data = await create_ephemeral_token(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        model=GEMINI_CONFIG["model"],
        voice="Puck",
    )

    import websockets
    try:
        async with websockets.connect(token_data["websocket_url"]) as ws:
            # Empty setup — config is baked into the token.
            await ws.send('{"setup":{}}')

            # Wait for setupComplete before sending the user turn. Without
            # this we'd sometimes send realtimeInput before the server is
            # ready and trip code 1007.
            got_setup = False
            try:
                async with asyncio.timeout(6.0):
                    async for raw in ws:
                        text = raw.decode() if isinstance(raw, bytes) else raw
                        if '"setupComplete"' in text:
                            got_setup = True
                            break
            except asyncio.TimeoutError:
                return {"hit": False, "reason": "timeout_before_setupComplete", "transcript": ""}
            if not got_setup:
                return {"hit": False, "reason": "no_setupComplete", "transcript": ""}

            # Send the learner's utterance as a user turn.
            # 3.1 Live: clientContent.turns is silently dropped for live input;
            # realtimeInput.text is the correct channel. Migrated 2026-04-18.
            user_turn = json.dumps({
                "realtimeInput": {"text": utterance}
            })
            await ws.send(user_turn)

            # Now wait up to TIMEOUT for either:
            #   (a) toolCall.functionCalls[...advance_step...] → HIT
            #   (b) serverContent.turnComplete == True without a toolCall → MISS
            transcript_parts = []
            try:
                async with asyncio.timeout(TIMEOUT_SEC):
                    async for raw in ws:
                        text = raw.decode() if isinstance(raw, bytes) else raw
                        # Try to parse JSON. Some frames will be raw PCM audio;
                        # those decode but won't have structure we care about.
                        try:
                            obj = json.loads(text)
                        except Exception:
                            continue

                        # Did the model dispatch a tool call?
                        if "toolCall" in obj:
                            fcs = obj["toolCall"].get("functionCalls", [])
                            names = [fc.get("name") for fc in fcs]
                            if "advance_step" in names:
                                return {
                                    "hit": True,
                                    "reason": "toolCall_advance_step",
                                    "transcript": "".join(transcript_parts),
                                }
                            else:
                                return {
                                    "hit": False,
                                    "reason": f"wrong_tool_called:{names}",
                                    "transcript": "".join(transcript_parts),
                                }

                        sc = obj.get("serverContent", {})
                        # Accumulate output transcription so we can see WHAT the
                        # model said when it failed to dispatch.
                        ot = sc.get("outputTranscription", {}) if isinstance(sc, dict) else {}
                        if isinstance(ot, dict) and ot.get("text"):
                            transcript_parts.append(ot["text"])

                        if isinstance(sc, dict) and sc.get("turnComplete"):
                            # Model finished speaking without a toolCall.
                            return {
                                "hit": False,
                                "reason": "turnComplete_without_toolCall",
                                "transcript": "".join(transcript_parts),
                            }
            except asyncio.TimeoutError:
                return {
                    "hit": False,
                    "reason": f"timeout_after_{TIMEOUT_SEC}s",
                    "transcript": "".join(transcript_parts),
                }

            return {"hit": False, "reason": "stream_ended", "transcript": "".join(transcript_parts)}
    except Exception as exc:
        return {"hit": False, "reason": f"exception:{type(exc).__name__}:{str(exc)[:120]}", "transcript": ""}


async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set")
        sys.exit(1)

    print(f"Model: {GEMINI_CONFIG['model']}")
    print(f"Running {N_TRIALS} trials, {TIMEOUT_SEC}s timeout each")
    print(f"Tools: {[t['name'] for t in TOOL_DEFINITIONS]}")
    print()

    hits = 0
    results = []
    for i in range(N_TRIALS):
        utterance = UTTERANCES[i % len(UTTERANCES)]
        print(f"[{i+1:2d}/{N_TRIALS}] '{utterance}' ...", end=" ", flush=True)
        r = await run_trial(i, utterance)
        results.append((utterance, r))
        if r["hit"]:
            hits += 1
            print("✅ HIT")
        else:
            verbal = r["transcript"][:80].replace("\n", " ")
            print(f"❌ MISS ({r['reason']}) verbal='{verbal}'")
        # Brief pause so we don't hammer the server.
        await asyncio.sleep(0.5)

    rate = hits / N_TRIALS * 100
    print()
    print("=" * 70)
    print(f"RESULT: {hits}/{N_TRIALS} = {rate:.0f}% HIT rate")
    print("=" * 70)
    if rate >= 80:
        print("→ Tool calling is RELIABLE enough. Keep native-audio, ship it.")
    elif rate >= 50:
        print("→ Tool calling is MARGINAL. Consider transcript-fallback safety net.")
    else:
        print("→ Tool calling is UNRELIABLE. Transcript-fallback is required.")

    # Surface MISS utterances so we can see if specific phrasings never work.
    misses = [(u, r) for u, r in results if not r["hit"]]
    if misses:
        print()
        print("MISSES:")
        for u, r in misses:
            verbal = r["transcript"][:120].replace("\n", " ")
            print(f"  - '{u}' → {r['reason']} | verbal='{verbal}'")


if __name__ == "__main__":
    asyncio.run(main())
