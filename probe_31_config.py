"""Diagnose why gemini-3.1-flash-live-preview rejects our baseline config.

`models.list()` confirms it exists on this API key. probe_live_models.py got
1007 "invalid argument" with the minimal `response_modalities=["AUDIO"]` probe.
Step through the config options one at a time to find the rejected field.
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

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.1-flash-live-preview"


async def try_config(client: genai.Client, label: str, config: types.LiveConnectConfig):
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await session.send_client_content(
                turns=types.Content(parts=[types.Part(text="hi")]),
                turn_complete=True,
            )
            try:
                async with asyncio.timeout(5.0):
                    async for _ in session.receive():
                        print(f"  ✅ {label}")
                        return True
            except asyncio.TimeoutError:
                print(f"  ⚠  {label}: connected, no payload in 5s")
                return True
    except Exception as exc:
        msg = str(exc)[:220]
        print(f"  ❌ {label}: {type(exc).__name__}: {msg}")
        return False


async def main():
    if not API_KEY:
        print("GEMINI_API_KEY not set")
        sys.exit(1)

    for version in ("v1alpha", "v1beta"):
        print("=" * 80)
        print(f"API version: {version}")
        print("=" * 80)
        client = genai.Client(api_key=API_KEY, http_options={"api_version": version})

        # Progressively test configs to find what 3.1 accepts.
        await try_config(
            client,
            "response_modalities=[AUDIO] only",
            types.LiveConnectConfig(response_modalities=["AUDIO"]),
        )
        await try_config(
            client,
            "response_modalities=[TEXT] only",
            types.LiveConnectConfig(response_modalities=["TEXT"]),
        )
        await try_config(
            client,
            "no response_modalities set",
            types.LiveConnectConfig(),
        )
        await try_config(
            client,
            "AUDIO + system_instruction",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=types.Content(parts=[types.Part(text="Be brief.")]),
            ),
        )
        await try_config(
            client,
            "AUDIO + speech_config(Puck)",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                    )
                ),
            ),
        )
        await try_config(
            client,
            "AUDIO + tools (function calling)",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name="advance_step",
                        description="advance to next step",
                        parameters={"type": "object", "properties": {}, "required": []},
                    )
                ])],
            ),
        )
        await try_config(
            client,
            "AUDIO + context_window_compression",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                context_window_compression=types.ContextWindowCompressionConfig(
                    trigger_tokens=100_000,
                    sliding_window=types.SlidingWindow(target_tokens=40_000),
                ),
            ),
        )
        await try_config(
            client,
            "AUDIO + session_resumption",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                session_resumption=types.SessionResumptionConfig(),
            ),
        )
        await try_config(
            client,
            "AUDIO + input+output transcription",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
            ),
        )
        await try_config(
            client,
            "AUDIO + proactivity (v1alpha-only feature)",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                proactivity=types.ProactivityConfig(proactive_audio=True),
            ),
        )
        await try_config(
            client,
            "AUDIO + media_resolution=LOW",
            types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                media_resolution="MEDIA_RESOLUTION_LOW",
            ),
        )
        print()


if __name__ == "__main__":
    asyncio.run(main())
