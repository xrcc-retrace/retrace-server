import logging
import subprocess
from pathlib import Path

from config import FFMPEG_PATH

logger = logging.getLogger(__name__)


async def extract_clips(
    video_path: str,
    procedure_id: str,
    steps: list[dict],
    output_dir: Path,
) -> dict[int, str]:
    """Extract one MP4 clip per step using the step's timestamp range.

    Each clip is re-encoded to H.264+AAC for broad playback compatibility
    (including iOS/glasses). Uses -ss before -i for fast seeking.

    Returns a mapping of step_number -> clip file path.
    """
    procedure_dir = output_dir / procedure_id
    procedure_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: dict[int, str] = {}

    for step in steps:
        step_num = step["step_number"]
        start = step["timestamp_start"]
        end = step["timestamp_end"]
        duration = end - start

        if duration <= 0:
            logger.warning("Step %d has zero/negative duration (%.2f-%.2f), skipping", step_num, start, end)
            continue

        output_path = procedure_dir / f"step_{step_num}.mp4"

        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",  # enable streaming playback
            str(output_path),
        ]

        logger.info(
            "Extracting clip for step %d: %.2fs - %.2fs (%.1fs)",
            step_num, start, end, duration,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("ffmpeg failed for step %d: %s", step_num, result.stderr[-500:])
            continue

        file_size_kb = output_path.stat().st_size / 1024
        clip_paths[step_num] = str(output_path)
        logger.info("Clip saved: %s (%.0f KB)", output_path.name, file_size_kb)

    return clip_paths
