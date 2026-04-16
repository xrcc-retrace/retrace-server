import asyncio
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
) -> tuple[dict[int, str], list[int]]:
    """Extract one MP4 clip per step using the step's timestamp range.

    Each clip is re-encoded to H.264+AAC for broad playback compatibility
    (including iOS/glasses). Uses -ss before -i for fast seeking.

    Returns (success_map, failed_step_numbers). A step fails when duration
    is non-positive or ffmpeg exits non-zero.
    """
    procedure_dir = output_dir / procedure_id
    procedure_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: dict[int, str] = {}
    failed: list[int] = []

    for step in steps:
        step_num = step["step_number"]
        start = step["timestamp_start"]
        end = step["timestamp_end"]
        duration = end - start

        if duration <= 0:
            logger.warning(
                "Step %d has zero/negative duration (%.2f-%.2f), skipping",
                step_num, start, end,
            )
            failed.append(step_num)
            continue

        output_path = procedure_dir / f"step_{step_num}.mp4"

        cmd = [
            FFMPEG_PATH, "-y",
            # Output-seek: `-ss` AFTER `-i` decodes from the start and discards frames
            # before `start`. Slower than input-seek but produces monotonic PTS/DTS from 0.
            "-i", video_path,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            # iOS HD playback rejects clips with negative DTS. `-bf 0` disables B-frames
            # so PTS == DTS always. Small quality hit; negligible at CRF 23.
            "-bf", "0",
            "-pix_fmt", "yuv420p",
            # Container-level color tags (colr atom).
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            # Bitstream-level SEI tags — libx264 ignores the ffmpeg flags above for this,
            # needs x264opts. Without these, `color_primaries=unknown` in the stream and
            # iOS VideoToolbox reports `verify_image_parameters` errors on HD portrait.
            "-x264opts", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",  # enable streaming playback
            str(output_path),
        ]

        logger.info(
            "Extracting clip for step %d: %.2fs - %.2fs (%.1fs)",
            step_num, start, end, duration,
        )
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.error("ffmpeg failed for step %d: %s", step_num, result.stderr[-500:])
            failed.append(step_num)
            continue

        file_size_kb = output_path.stat().st_size / 1024
        clip_paths[step_num] = str(output_path)
        logger.info("Clip saved: %s (%.0f KB)", output_path.name, file_size_kb)

    return clip_paths, failed
