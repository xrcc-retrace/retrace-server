"""Canary probe: verify Gemini Live honors the system_instruction baked into
ephemeral tokens.

Motivated by https://discuss.ai.google.dev/t/live-api-with-ephemeral-token-ignores-the-system-instruction/113346
— an open, unresolved report that BidiGenerateContentConstrained sessions can
silently ignore the system_instruction in live_connect_constraints.

Our coaching step-progression logic depends on the prompt being honored. The
existing probe_retrace_config.py only confirms the socket connects and
setupComplete round-trips. This probe additionally confirms the model's
behavior reflects the baked-in prompt.

Strategy: bake a distinctive canary rule into the system prompt, open a real
Live session via the production ephemeral-token path, ask a neutral question,
and assert the response contains the canary token.
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
from services.prompt_builder import GEMINI_CONFIG

CANARY = "CANARY_RETRACE_OK"
CANARY_PROMPT = (
    "You are a test assistant. You MUST begin every single response with the "
    f"exact literal token {CANARY} followed by a space. This prefix is a hard "
    "requirement with no exceptions — it applies to greetings, answers, "
    "acknowledgments, and every other utterance. After the prefix, keep "
    "replies to one short sentence."
)


async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set")
        sys.exit(1)

    print(f"Target model: {GEMINI_CONFIG['model']}")
    print(f"Canary token: {CANARY}")

    # 1. Mint an ephemeral token carrying the canary prompt via the exact
    #    production code path iOS uses.
    print("\n[1] Minting ephemeral token with canary system_instruction...")
    try:
        token_data = await create_ephemeral_token(
            system_instruction=CANARY_PROMPT,
            tools=[],
            model=GEMINI_CONFIG["model"],
            voice=GEMINI_CONFIG["voice"],
        )
        print(f"    Token minted (expires_at={token_data['expires_at']})")
    except Exception as exc:
        print(f"    FAILED to mint token: {type(exc).__name__}: {exc}")
        sys.exit(2)

    # 2. Open the WebSocket, send empty setup (config is in the token),
    #    wait for setupComplete, send a neutral user turn, collect the
    #    transcribed reply.
    print("\n[2] Opening Live session and asking a neutral question...")
    import websockets

    try:
        async with websockets.connect(token_data["websocket_url"]) as ws:
            await ws.send('{"setup":{}}')

            got_setup_complete = False
            output_transcript_parts: list[str] = []
            turn_complete = False

            async def read_frames():
                nonlocal got_setup_complete, turn_complete
                async for raw in ws:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    try:
                        msg = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    if "setupComplete" in msg:
                        got_setup_complete = True
                        await ws.send(json.dumps({
                            "clientContent": {
                                "turns": [{
                                    "role": "user",
                                    "parts": [{"text": "Hello, please reply with one short sentence."}],
                                }],
                                "turnComplete": True,
                            }
                        }))
                        continue

                    server_content = msg.get("serverContent")
                    if not server_content:
                        continue

                    # Collect the output-audio transcript chunks.
                    output_trans = server_content.get("outputTranscription")
                    if output_trans and "text" in output_trans:
                        output_transcript_parts.append(output_trans["text"])

                    if server_content.get("turnComplete"):
                        turn_complete = True
                        break

            try:
                async with asyncio.timeout(25.0):
                    await read_frames()
            except asyncio.TimeoutError:
                pass

            if not got_setup_complete:
                print("    FAILED: no setupComplete within 25s")
                sys.exit(3)

            transcript = "".join(output_transcript_parts).strip()
            print(f"    setupComplete=yes  turn_complete={turn_complete}")
            print(f"    Output transcript: {transcript!r}")

            # 3. Interpret the result.
            print("\n[3] Interpreting result...")
            if not transcript:
                print("    INCONCLUSIVE: no output transcript captured.")
                print("    The model may have chosen silence (proactive_audio=True) or")
                print("    output_audio_transcription didn't fire. Re-run; if persistent,")
                print("    switch to response_modalities=['TEXT'] for this probe.")
                sys.exit(4)

            if CANARY in transcript:
                print(f"    PASS: canary '{CANARY}' present in response.")
                print("    system_instruction IS honored with ephemeral tokens.")
                print("    Architecture verdict confirmed — step data can stay in the")
                print("    system prompt, current Retrace design is sound.")
                sys.exit(0)
            else:
                print(f"    FAIL: canary '{CANARY}' NOT present in response.")
                print("    The forum-thread bug is active on this API key: the")
                print("    system_instruction baked into live_connect_constraints is")
                print("    being silently ignored. Coaching behavior cannot be relied on.")
                print("    Next steps: (a) file a reproducer upstream, (b) review")
                print("    alternative grounding paths before the next device test.")
                sys.exit(5)

    except Exception as exc:
        print(f"    WebSocket probe FAILED: {type(exc).__name__}: {exc}")
        sys.exit(6)


if __name__ == "__main__":
    asyncio.run(main())
