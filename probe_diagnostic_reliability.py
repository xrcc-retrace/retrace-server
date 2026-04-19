"""Measure tool-call reliability for the Troubleshooting diagnostic prompt.

Gate for iOS work: if Gemini doesn't reliably call identify_product when the
user names a broken product, the whole diagnostic flow stalls at phase 1.
The prompt also asks the model to call search_procedures after 2-3 symptom
exchanges and handoff_to_learner once the user confirms a match — this probe
covers all three with scenario-specific contexts.

Runs against the EXACT production path: ephemeral token minted via
services.ephemeral_token (v1alpha auth_tokens) + BidiGenerateContentConstrained
WebSocket. Same config baked into every iOS session.

Reports:
  - Per-tool HIT rate across N trials
  - MISS transcripts so you can see what the model said instead

Gate decision:
  >= 80% overall HIT rate: ship it
  50-79%: marginal; tune the prompt before iOS work
  <  50%: structural — either the prompt is wrong or tool calling doesn't
          work well in free-conversation mode; escalate to transcript fallback
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
from services.troubleshoot.prompt import (
    DIAGNOSTIC_TOOL_DEFINITIONS,
    GEMINI_CONFIG_DIAGNOSTIC,
    build_diagnostic_prompt,
)


SYSTEM_PROMPT = build_diagnostic_prompt(voice=GEMINI_CONFIG_DIAGNOSTIC["voice"])

TIMEOUT_SEC = 30.0


# A scenario is a list of user turns to replay against a fresh Live session,
# and the tool name we expect the model to call by the end. Multi-turn lets
# us test tools that only make sense after context is built up
# (search_procedures needs identify_product first; handoff_to_learner needs
# the model to think it has a matching procedure).
#
# For multi-turn scenarios we interleave synthetic tool responses so the
# model sees the same feedback it would see in production.
SCENARIOS: list[dict] = [
    # ===== identify_product trials (single-turn, vague-to-specific) =====
    {
        "name": "identify_levoit_by_model",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "Hey, my Levoit LV600HH humidifier isn't making any mist."},
        ],
    },
    {
        "name": "identify_levoit_casual",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "My humidifier stopped working. It's the Levoit one, the tall tower."},
        ],
    },
    {
        "name": "identify_instant_pot_by_error",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "My Instant Pot Duo is showing a C6 error on the display."},
        ],
    },
    {
        "name": "identify_macbook",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "I've got a MacBook Pro that won't turn on at all. No chime, no lights."},
        ],
    },
    {
        "name": "identify_keurig",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "My Keurig K-Elite keeps telling me to add water even though the reservoir is full."},
        ],
    },
    {
        "name": "identify_wmf_espresso",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "The WMF 1500S espresso machine has an error code on the display and won't brew."},
        ],
    },
    {
        "name": "identify_dyson",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "My Dyson V11 cordless vacuum isn't holding a charge — shuts off after thirty seconds."},
        ],
    },
    {
        "name": "identify_vague_then_clarify",
        "expected_tool": "identify_product",
        "turns": [
            {"user": "My humidifier is broken."},
            # Model probably asks for details — give it the brand
            {"user": "It's a Levoit, the big tower one, LV600 I think."},
        ],
    },
    # ===== search_procedures trials (multi-turn: identify first, then symptoms) =====
    {
        "name": "search_after_identify_levoit",
        "expected_tool": "search_procedures",
        "prior_tool_responses": {
            # Keyed by tool_name — the probe will auto-respond when the model
            # calls this tool, so the conversation can proceed to the next turn.
            "identify_product": {
                "product_name": "Levoit LV600HH",
                "confidence": "high",
                "category": "humidifier",
                "linked_procedures": [],
            },
        },
        "turns": [
            {"user": "My Levoit LV600HH humidifier isn't misting."},
            {"user": "Yeah, the tank is full and properly seated. The light is on."},
            {"user": "I can hear the fan humming but nothing is coming out. I haven't cleaned it in about six months."},
        ],
    },
    {
        "name": "search_after_identify_instantpot",
        "expected_tool": "search_procedures",
        "prior_tool_responses": {
            "identify_product": {
                "product_name": "Instant Pot Duo V5",
                "confidence": "high",
                "category": "pressure cooker",
                "linked_procedures": [],
            },
        },
        "turns": [
            {"user": "My Instant Pot Duo is showing C6 on the display."},
            {"user": "It started right when I was trying to pressure cook some rice."},
            {"user": "The lid was sealed properly, ring is in place. It just beeped and showed the code."},
        ],
    },
    # ===== handoff_to_learner trials (multi-turn culminating in user confirmation) =====
    {
        "name": "handoff_after_user_confirm",
        "expected_tool": "handoff_to_learner",
        "prior_tool_responses": {
            "identify_product": {
                "product_name": "Levoit LV600HH",
                "confidence": "high",
                "category": "humidifier",
                "linked_procedures": [],
            },
            "search_procedures": {
                "candidates": [
                    {
                        "procedure_id": "proc-123-descaling",
                        "title": "Levoit LV600HH — Transducer Descaling",
                        "match_reason": "no mist output with fan running, likely mineral buildup on ultrasonic transducer",
                    },
                ],
            },
        },
        "turns": [
            {"user": "My Levoit LV600HH isn't misting."},
            {"user": "Tank's full, fan is running, blue light is on, no cleaning in six months."},
            {"user": "The one about descaling the transducer — yeah, that's exactly the right one. Let's do it."},
        ],
    },
    {
        "name": "handoff_terse_confirm",
        "expected_tool": "handoff_to_learner",
        "prior_tool_responses": {
            "identify_product": {
                "product_name": "Instant Pot Duo V5",
                "confidence": "high",
                "category": "pressure cooker",
                "linked_procedures": [],
            },
            "search_procedures": {
                "candidates": [
                    {
                        "procedure_id": "proc-456-c6",
                        "title": "Instant Pot Duo — C6 Error Recovery",
                        "match_reason": "C6 indicates pressure build failure — usually steam release valve or sealing ring",
                    },
                ],
            },
        },
        "turns": [
            {"user": "Instant Pot showing C6."},
            {"user": "Ring's in, lid is sealed, tried cooking rice."},
            {"user": "Go ahead, walk me through it."},
        ],
    },
    # ===== fetch_manual trial (search returned empty) =====
    {
        "name": "fetch_manual_after_empty_search",
        "expected_tool": "fetch_manual",
        "prior_tool_responses": {
            "identify_product": {
                "product_name": "Unknown Brand Humidifier",
                "confidence": "low",
                "category": "humidifier",
                "linked_procedures": [],
            },
            "search_procedures": {"candidates": []},
        },
        "turns": [
            {"user": "Some random humidifier I got on Amazon isn't misting anymore."},
            {"user": "Tank is full, fan runs, no mist. Don't remember the brand."},
            {"user": "Yeah I've had it maybe a year, never cleaned it."},
        ],
    },
]


async def _run_scenario(scenario: dict) -> dict:
    """Run one scenario end-to-end. Returns a result dict with hit / reason / transcript / tool_sequence."""
    token_data = await create_ephemeral_token(
        system_instruction=SYSTEM_PROMPT,
        tools=DIAGNOSTIC_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_DIAGNOSTIC["model"],
        voice=GEMINI_CONFIG_DIAGNOSTIC["voice"],
    )

    expected = scenario["expected_tool"]
    prior_responses = scenario.get("prior_tool_responses", {})
    turns = scenario["turns"]

    import websockets
    try:
        async with websockets.connect(token_data["websocket_url"]) as ws:
            await ws.send('{"setup":{}}')

            # Wait for setupComplete
            try:
                async with asyncio.timeout(6.0):
                    async for raw in ws:
                        text = raw.decode() if isinstance(raw, bytes) else raw
                        if '"setupComplete"' in text:
                            break
            except asyncio.TimeoutError:
                return {"hit": False, "reason": "timeout_before_setupComplete", "transcript": "", "tool_sequence": []}

            transcript_parts: list[str] = []
            tool_sequence: list[str] = []  # names of tools actually called
            turn_idx = 0

            async def send_user_turn(text: str):
                # 3.1 Live: clientContent is silently dropped for live input;
                # realtimeInput.text is the correct channel.
                await ws.send(json.dumps({
                    "realtimeInput": {"text": text}
                }))

            async def send_tool_response(call_id: str, name: str, response: dict):
                await ws.send(json.dumps({
                    "toolResponse": {
                        "functionResponses": [{
                            "id": call_id,
                            "name": name,
                            "response": response,
                        }],
                    }
                }))

            # Send first turn
            await send_user_turn(turns[turn_idx]["user"])
            turn_idx += 1

            try:
                async with asyncio.timeout(TIMEOUT_SEC):
                    async for raw in ws:
                        text = raw.decode() if isinstance(raw, bytes) else raw
                        try:
                            obj = json.loads(text)
                        except Exception:
                            continue

                        # Tool call from model
                        if "toolCall" in obj:
                            fcs = obj["toolCall"].get("functionCalls", [])
                            for fc in fcs:
                                name = fc.get("name", "")
                                call_id = fc.get("id", "")
                                tool_sequence.append(name)

                                if name == expected:
                                    # HIT — this is the tool we were testing for
                                    return {
                                        "hit": True,
                                        "reason": f"called_{name}",
                                        "transcript": "".join(transcript_parts),
                                        "tool_sequence": tool_sequence,
                                        "tool_args": fc.get("args", {}),
                                    }

                                # Not the expected tool — but might be a prerequisite
                                # (e.g. identify_product before search_procedures).
                                # Auto-respond with the scripted prior response so
                                # the conversation can keep going.
                                if name in prior_responses:
                                    await send_tool_response(call_id, name, prior_responses[name])
                                else:
                                    # Tool was called that isn't expected and has
                                    # no prior response scripted. Return generic ack.
                                    await send_tool_response(call_id, name, {"status": "ok"})

                            continue

                        sc = obj.get("serverContent", {}) if isinstance(obj, dict) else {}

                        # Accumulate output transcription
                        ot = sc.get("outputTranscription", {}) if isinstance(sc, dict) else {}
                        if isinstance(ot, dict) and ot.get("text"):
                            transcript_parts.append(ot["text"])

                        # Turn complete — model stopped speaking. If there are
                        # more turns, send the next one. Otherwise we missed.
                        if isinstance(sc, dict) and sc.get("turnComplete"):
                            if turn_idx < len(turns):
                                await send_user_turn(turns[turn_idx]["user"])
                                turn_idx += 1
                                # reset per-turn transcript marker
                                transcript_parts.append(" || ")
                            else:
                                return {
                                    "hit": False,
                                    "reason": f"ran_out_of_turns_without_{expected}",
                                    "transcript": "".join(transcript_parts),
                                    "tool_sequence": tool_sequence,
                                }

            except asyncio.TimeoutError:
                return {
                    "hit": False,
                    "reason": f"timeout_after_{TIMEOUT_SEC}s",
                    "transcript": "".join(transcript_parts),
                    "tool_sequence": tool_sequence,
                }

            return {
                "hit": False,
                "reason": "stream_ended",
                "transcript": "".join(transcript_parts),
                "tool_sequence": tool_sequence,
            }
    except Exception as exc:
        return {
            "hit": False,
            "reason": f"exception:{type(exc).__name__}:{str(exc)[:150]}",
            "transcript": "",
            "tool_sequence": [],
        }


async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set")
        sys.exit(1)

    print(f"Model: {GEMINI_CONFIG_DIAGNOSTIC['model']}")
    print(f"Voice: {GEMINI_CONFIG_DIAGNOSTIC['voice']}")
    print(f"Tools: {[t['name'] for t in DIAGNOSTIC_TOOL_DEFINITIONS]}")
    print(f"Prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"Scenarios: {len(SCENARIOS)}")
    print()

    results = []
    for i, scenario in enumerate(SCENARIOS):
        label = f"[{i+1:2d}/{len(SCENARIOS)}] {scenario['name']:40s} expect={scenario['expected_tool']:24s}"
        print(label, end=" ... ", flush=True)
        r = await _run_scenario(scenario)
        results.append((scenario, r))
        if r["hit"]:
            print("HIT")
        else:
            verbal = r["transcript"][:80].replace("\n", " ")
            seq = r.get("tool_sequence", [])
            seq_str = f"seq={seq}" if seq else ""
            print(f"MISS ({r['reason']}) {seq_str} verbal='{verbal}'")
        # Brief pause so we don't hammer the API
        await asyncio.sleep(0.5)

    # Per-tool breakdown
    by_tool: dict[str, list] = {}
    for scenario, r in results:
        by_tool.setdefault(scenario["expected_tool"], []).append(r)

    print()
    print("=" * 78)
    print("Per-tool HIT rates:")
    for tool_name, rs in sorted(by_tool.items()):
        hits = sum(1 for r in rs if r["hit"])
        total = len(rs)
        pct = hits / total * 100 if total else 0
        print(f"  {tool_name:28s} {hits}/{total}  ({pct:.0f}%)")

    total_hits = sum(1 for _, r in results if r["hit"])
    overall_pct = total_hits / len(results) * 100 if results else 0
    print()
    print(f"OVERALL: {total_hits}/{len(results)} = {overall_pct:.0f}%")
    print("=" * 78)

    if overall_pct >= 80:
        print("GATE PASS: ship it. Tool calling is reliable enough.")
    elif overall_pct >= 50:
        print("GATE MARGINAL: tune the prompt before iOS work.")
    else:
        print("GATE FAIL: prompt or tool defs need rework, or fall back to transcript scanning.")

    # Dump misses for diagnosis
    misses = [(s, r) for s, r in results if not r["hit"]]
    if misses:
        print()
        print("MISSES:")
        for scenario, r in misses:
            verbal = r["transcript"][:200].replace("\n", " ")
            seq = r.get("tool_sequence", [])
            print(f"  - {scenario['name']} (expected {scenario['expected_tool']})")
            print(f"      reason: {r['reason']}")
            if seq:
                print(f"      tool_sequence: {seq}")
            print(f"      verbal: '{verbal}'")


if __name__ == "__main__":
    asyncio.run(main())
