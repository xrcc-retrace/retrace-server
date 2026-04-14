from pathlib import Path
import os
import shutil

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CLIPS_DIR = DATA_DIR / "clips"
DB_PATH = DATA_DIR / "retrace.db"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-pro"

# Find ffmpeg binary
FFMPEG_PATH = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# Ensure directories exist
for d in [UPLOAD_DIR, DATA_DIR, CLIPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
