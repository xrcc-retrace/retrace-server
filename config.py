from pathlib import Path
import os
import shutil

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
MANUAL_IMAGES_DIR = DATA_DIR / "manual_images"
DB_PATH = DATA_DIR / "retrace.db"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-pro"

# Live model id baked into every ephemeral token. Override via
# RETRACE_LIVE_MODEL=<id> for fast rollback without a code change, e.g.
#   RETRACE_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025 python main.py
RETRACE_LIVE_MODEL = os.environ.get(
    "RETRACE_LIVE_MODEL",
    "gemini-3.1-flash-live-preview",
)

# Out-of-band step-verification oracle. A dedicated Flash model compares the
# learner's current camera frame against the step's spatial end-state (and
# error states) so Gemini Live never has to do visual verification itself.
# Override via RETRACE_ORACLE_MODEL for a fast swap without a code change.
RETRACE_ORACLE_MODEL = os.environ.get(
    "RETRACE_ORACLE_MODEL",
    "gemini-2.5-flash",
)

# Find ffmpeg binary
FFMPEG_PATH = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# Ensure directories exist
for d in [UPLOAD_DIR, DATA_DIR, CLIPS_DIR, MANUAL_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
