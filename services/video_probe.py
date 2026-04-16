"""ffprobe helper — authoritative video duration on upload."""

import asyncio
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

FFPROBE_PATH = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


async def probe_duration(video_path: str) -> float:
    """Return the video duration in seconds via ffprobe.

    Raises RuntimeError if ffprobe exits non-zero or the output isn't a float.
    """
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed ({result.returncode}): {result.stderr[-500:]}"
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration: {result.stdout!r}"
        ) from exc

    logger.info("Probed video duration: %.2fs (%s)", duration, video_path)
    return duration
