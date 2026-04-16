"""Generate voice preview WAV files for the curated Gemini voices.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python scripts/generate_voice_previews.py

Requires GEMINI_API_KEY in .env or environment.
"""

import os
import struct
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the server root
SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env")

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

VOICES = [
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Zephyr",
    "Enceladus",
    "Leda",
]

PREVIEW_TEXT = "Hi, I'll be your coaching guide today. Let's get started."
OUTPUT_DIR = SERVER_ROOT / "static" / "voice_previews"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Gemini TTS outputs PCM 16-bit signed LE, mono, 24 kHz
SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Wrap raw PCM data in a WAV header."""
    data_size = len(pcm_data)
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8
    block_align = NUM_CHANNELS * BITS_PER_SAMPLE // 8

    header = struct.pack(
        "<4sI4s"    # RIFF header
        "4sIHHIIHH"  # fmt chunk
        "4sI",       # data chunk header
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,              # fmt chunk size
        1,               # PCM format
        NUM_CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        BITS_PER_SAMPLE,
        b"data",
        data_size,
    )
    return header + pcm_data


def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    for voice_name in VOICES:
        print(f"Generating preview for {voice_name}...", end=" ", flush=True)

        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=PREVIEW_TEXT,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )

        # Extract raw PCM audio from response
        audio_part = response.candidates[0].content.parts[0]
        pcm_data = audio_part.inline_data.data

        wav_data = pcm_to_wav(pcm_data)
        out_path = OUTPUT_DIR / f"{voice_name}.wav"
        out_path.write_bytes(wav_data)

        size_kb = len(wav_data) / 1024
        print(f"OK ({size_kb:.1f} KB)")

    print(f"\nDone! {len(VOICES)} preview files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
