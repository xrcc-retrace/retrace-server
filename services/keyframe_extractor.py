import logging
import subprocess
from pathlib import Path

from config import FFMPEG_PATH

logger = logging.getLogger(__name__)


async def extract_keyframes(
    video_path: str,
    procedure_id: str,
    steps: list[dict],
    output_dir: Path,
) -> dict[int, str]:
    """Extract one JPEG keyframe per step at the midpoint timestamp.

    Returns a mapping of step_number -> keyframe file path.
    """
    procedure_dir = output_dir / procedure_id
    procedure_dir.mkdir(parents=True, exist_ok=True)

    keyframe_paths: dict[int, str] = {}

    for step in steps:
        step_num = step["step_number"]
        midpoint = (step["timestamp_start"] + step["timestamp_end"]) / 2
        output_path = procedure_dir / f"step_{step_num}.jpg"

        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", f"{midpoint:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]

        logger.info("Extracting keyframe for step %d at %.2fs", step_num, midpoint)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(
                "ffmpeg failed for step %d: %s", step_num, result.stderr
            )
            continue

        keyframe_paths[step_num] = str(output_path)
        logger.info("Keyframe saved: %s", output_path)

    return keyframe_paths
