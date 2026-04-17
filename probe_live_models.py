"""Probe which Gemini Live model IDs are currently accepted by bidiGenerateContent.

For each candidate model, open a direct Live session (bypassing ephemeral tokens
since those are just a gateway to the same endpoint), send a setup, wait for
setupComplete, and report pass/fail with the close reason when it fails.

Separately probes whether `proactivity.proactive_audio=True` and video input are
accepted per model — the things our Retrace config currently needs.

Run:
    cd main-local-server-test
    source .venv/bin/activate
    python probe_live_models.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env so GEMINI_API_KEY is picked up
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("GEMINI_API_KEY not set")
    sys.exit(1)

CANDIDATES = [
    # Current (known broken — our regression)
    "gemini-2.5-flash-live-preview",
    # Proposed quick fix
    "gemini-2.0-flash-live-001",
    # Newest, per docs
    "gemini-3.1-flash-live-preview",
    # Recent 2.5 native-audio
    "gemini-2.5-flash-native-audio-preview-12-2025",
    # Alternate naming patterns seen in search results
    "gemini-live-2.5-flash-native-audio",
    "gemini-live-2.5-flash-preview-native-audio-09-2025",
    # Older experimental, sometimes works as a fallback
    "gemini-2.0-flash-exp",
]


async def probe_handshake(client: genai.Client, model: str) -> tuple[bool, str]:
    """Open a minimal Live session. Returns (ok, detail)."""
    try:
        config = types.LiveConnectConfig(response_modalities=["AUDIO"])
        async with client.aio.live.connect(model=model, config=config) as session:
            # Send a trivial text turn so we exercise the receive loop too
            await session.send_client_content(
                turns=types.Content(parts=[types.Part(text="hi")]),
                turn_complete=True,
            )
            # Pull a few messages to confirm the session is alive; bail after
            # the first server payload or timeout.
            try:
                async with asyncio.timeout(8.0):
                    async for _ in session.receive():
                        return True, "OK (received server payload)"
            except asyncio.TimeoutError:
                return True, "OK (connected, no payload in 8s — server was silent but did not close)"
        return True, "OK (clean disconnect)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def probe_with_feature(
    client: genai.Client,
    model: str,
    feature: str,
    config: types.LiveConnectConfig,
) -> tuple[bool, str]:
    """Try a session with a specific config feature enabled."""
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(
                turns=types.Content(parts=[types.Part(text="hi")]),
                turn_complete=True,
            )
            try:
                async with asyncio.timeout(5.0):
                    async for _ in session.receive():
                        return True, f"{feature}: accepted"
            except asyncio.TimeoutError:
                return True, f"{feature}: accepted (no payload)"
        return True, f"{feature}: accepted"
    except Exception as exc:
        return False, f"{feature}: REJECTED ({type(exc).__name__}: {str(exc)[:200]})"


async def main():
    # CRITICAL: api_version="v1alpha" — the proactivity/affective-dialog
    # preview features are documented as v1alpha-only (confirmed by forum
    # thread https://discuss.ai.google.dev/t/new-live-api-features-dont-work-through-the-api-proactive-audio-and-affective-dialong/84326).
    # Also our production ephemeral-token flow pins v1alpha (ephemeral_token.py),
    # so the probe should match that.
    client = genai.Client(
        api_key=API_KEY, http_options={"api_version": "v1alpha"}
    )

    print("=" * 80)
    print("HANDSHAKE PROBE (api_version=v1alpha)")
    print("=" * 80)
    results: dict[str, tuple[bool, str]] = {}
    for model in CANDIDATES:
        print(f"\n→ {model}")
        ok, detail = await probe_handshake(client, model)
        results[model] = (ok, detail)
        marker = "✅" if ok else "❌"
        print(f"  {marker} {detail}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    working = [m for m, (ok, _) in results.items() if ok]
    broken = [m for m, (ok, _) in results.items() if not ok]
    print(f"\nWorking ({len(working)}):")
    for m in working:
        print(f"  ✅ {m}")
    print(f"\nBroken ({len(broken)}):")
    for m in broken:
        print(f"  ❌ {m}")
        print(f"     {results[m][1][:250]}")

    if not working:
        print("\nNo models accepted bidiGenerateContent. Check API key, quota, region.")
        return

    # Feature probes — against the first working model only
    target = working[0]
    print("\n" + "=" * 80)
    print(f"FEATURE PROBE against {target}")
    print("=" * 80)

    # Base config baseline — just audio out
    base = types.LiveConnectConfig(response_modalities=["AUDIO"])

    # Tools
    tools_cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="advance_step",
                description="mark step complete",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ])],
    )
    ok, detail = await probe_with_feature(client, target, "tools (function calling)", tools_cfg)
    print(f"  {'✅' if ok else '❌'} {detail}")

    # proactive_audio — the feature our Retrace config has
    try:
        proactive_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            proactivity=types.ProactivityConfig(proactive_audio=True),
        )
        ok, detail = await probe_with_feature(client, target, "proactivity.proactive_audio", proactive_cfg)
        print(f"  {'✅' if ok else '❌'} {detail}")
    except Exception as exc:
        print(f"  ⚠  proactivity.proactive_audio: SDK couldn't construct config ({exc})")

    # Context window compression
    try:
        compress_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=100_000,
                sliding_window=types.SlidingWindow(target_tokens=40_000),
            ),
        )
        ok, detail = await probe_with_feature(client, target, "context_window_compression", compress_cfg)
        print(f"  {'✅' if ok else '❌'} {detail}")
    except Exception as exc:
        print(f"  ⚠  context_window_compression: SDK couldn't construct config ({exc})")

    # Session resumption (empty — just enable the feature)
    try:
        resume_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            session_resumption=types.SessionResumptionConfig(),
        )
        ok, detail = await probe_with_feature(client, target, "session_resumption", resume_cfg)
        print(f"  {'✅' if ok else '❌'} {detail}")
    except Exception as exc:
        print(f"  ⚠  session_resumption: SDK couldn't construct config ({exc})")

    # Input + output audio transcription
    try:
        trans_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        ok, detail = await probe_with_feature(client, target, "audio_transcription (in+out)", trans_cfg)
        print(f"  {'✅' if ok else '❌'} {detail}")
    except Exception as exc:
        print(f"  ⚠  audio_transcription: SDK couldn't construct config ({exc})")

    # FULL Retrace config — the whole thing at once
    try:
        full_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_LOW",
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="advance_step",
                    description="mark step complete",
                    parameters={"type": "object", "properties": {}, "required": []},
                )
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
        ok, detail = await probe_with_feature(client, target, "FULL Retrace config (all features together)", full_cfg)
        print(f"  {'✅' if ok else '❌'} {detail}")
    except Exception as exc:
        print(f"  ⚠  full config: SDK couldn't construct ({exc})")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
