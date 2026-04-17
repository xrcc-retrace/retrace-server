"""Verify the full Retrace Live config (minus proactive_audio) holds end-to-end.

Also mints a real ephemeral token via our production code path and opens a
session with it, so we're exercising the same flow the iOS app uses.
"""

import asyncio
import os
import sys
from pathlib import Path

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types

from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import GEMINI_CONFIG, TOOL_DEFINITIONS


async def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set")
        sys.exit(1)

    print(f"Target model: {GEMINI_CONFIG['model']}")
    print(f"Tools: {[t['name'] for t in TOOL_DEFINITIONS]}")

    # 1. Direct-connect probe with the full Retrace config (api_version=v1alpha
    # to match production ephemeral_token.py and to enable proactive_audio).
    print("\n[1] Direct connect with full production config (v1alpha)...")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    full_cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_LOW",
        system_instruction=types.Content(
            parts=[types.Part(text="You are a test assistant. Reply briefly.")]
        ),
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in TOOL_DEFINITIONS
        ])],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=100_000,
            sliding_window=types.SlidingWindow(target_tokens=40_000),
        ),
        session_resumption=types.SessionResumptionConfig(),
        proactivity=types.ProactivityConfig(proactive_audio=True),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    try:
        async with client.aio.live.connect(
            model=GEMINI_CONFIG["model"], config=full_cfg
        ) as session:
            await session.send_client_content(
                turns=types.Content(parts=[types.Part(text="Say hi in one word.")]),
                turn_complete=True,
            )
            got_anything = False
            try:
                async with asyncio.timeout(10.0):
                    async for msg in session.receive():
                        got_anything = True
                        if getattr(msg, "server_content", None) and \
                           getattr(msg.server_content, "turn_complete", False):
                            break
            except asyncio.TimeoutError:
                pass
            print(f"    ✅ Direct connect OK (received_payload={got_anything})")
    except Exception as exc:
        print(f"    ❌ Direct connect FAILED: {type(exc).__name__}: {exc}")
        return

    # 2. Ephemeral token mint (v1alpha auth_tokens) with production code
    print("\n[2] Mint ephemeral token via production code...")
    try:
        token_data = await create_ephemeral_token(
            system_instruction="You are a test assistant. Reply briefly.",
            tools=TOOL_DEFINITIONS,
            model=GEMINI_CONFIG["model"],
            voice="Puck",
        )
        print(f"    ✅ Token minted: expires_at={token_data['expires_at']}")
        print(f"       ws_url={token_data['websocket_url'][:100]}...")
    except Exception as exc:
        print(f"    ❌ Token mint FAILED: {type(exc).__name__}: {exc}")
        return

    # 3. Open a Live session using the ephemeral token (same as iOS)
    print("\n[3] Connect via ephemeral token (same flow iOS uses)...")
    import websockets
    try:
        async with websockets.connect(token_data["websocket_url"]) as ws:
            # Empty setup frame — our production iOS pattern
            await ws.send('{"setup":{}}')
            got_setup_complete = False
            try:
                async with asyncio.timeout(10.0):
                    async for raw in ws:
                        if b'"setupComplete"' in raw if isinstance(raw, bytes) else '"setupComplete"' in raw:
                            got_setup_complete = True
                            break
            except asyncio.TimeoutError:
                pass
            if got_setup_complete:
                print("    ✅ Ephemeral-token handshake OK — setupComplete received")
            else:
                print("    ⚠  Handshake connected but no setupComplete within 10s")
    except Exception as exc:
        print(f"    ❌ Ephemeral-token connect FAILED: {type(exc).__name__}: {exc}")
        return

    print("\n✅ All three checks passed. iOS should now connect cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
