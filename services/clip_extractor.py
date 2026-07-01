import asyncio
import logging
import subprocess
from pathlib import Path

from config import FFMPEG_PATH

logger = logging.getLogger(__name__)

# Suffix for the lens-optimized variant served to the glasses at ?profile=lens.
LENS_VARIANT_SUFFIX = "_lens"


def lens_variant_path(clip_path: Path) -> Path:
    """The lens-variant path that sits beside a full clip (step_N.mp4 → step_N_lens.mp4)."""
    return clip_path.with_name(f"{clip_path.stem}{LENS_VARIANT_SUFFIX}{clip_path.suffix}")


def done_frame_path(clip_path: Path) -> Path:
    """The 'done state' poster beside a clip (step_N.mp4 → step_N_done.jpg).

    This is the LAST frame of the step clip — the canonical picture of what
    the scene looks like once the step is complete. The verification oracle
    compares the learner's live frame against it (image-to-image grounding).
    Distinct from the first-frame poster step_N.jpg used as a thumbnail.
    """
    return clip_path.with_name(f"{clip_path.stem}_done.jpg")


def extract_done_frame(clip_path: Path) -> Path | None:
    """Extract + disk-cache the last frame of a clip as a JPEG.

    Returns the cached path, or None if extraction failed / the clip is
    missing. `-sseof -1.0` decodes only the final ~1s of the clip and
    `-update 1` overwrites the output for each decoded frame, so the file
    left behind is the true last frame — cheap and robust for short clips.
    """
    if not clip_path.exists():
        return None
    dst = done_frame_path(clip_path)
    if dst.exists():
        return dst
    try:
        subprocess.run(
            [
                FFMPEG_PATH, "-y",
                "-sseof", "-1.0",
                "-i", str(clip_path),
                "-update", "1",
                "-q:v", "2",
                str(dst),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        logger.error("Done-frame extraction failed for %s: %s", clip_path.name, stderr[-300:])
        return None
    return dst if dst.exists() else None


def upload_lens_variant_path(src: Path) -> Path:
    """The lens-variant path for a full source upload ({uuid}.ext → {uuid}_lens.mp4).

    Full uploads may be .mov; transcode_lens_variant always emits H.264/AAC+faststart,
    so force a .mp4 container for the variant (the lens VideoPlayer uses codec: .mp4).
    """
    return src.with_name(f"{src.stem}{LENS_VARIANT_SUFFIX}.mp4")


def transcode_lens_variant(src_mp4: Path, dst_mp4: Path) -> bool:
    """Transcode a full-res step clip into a small, bitrate-capped lens variant.

    The native glasses lens is a 600x600 viewport fed over the Bluetooth Display
    link, so bytes-to-first-frame dominates playback latency. This downscales to
    fit 600x600 (aspect preserved, dims kept even for libx264) and caps the
    bitrate, while keeping the same iOS/VideoToolbox-compatibility flags as
    `extract_clips` (-bf 0, yuv420p, bt709 color tags + SEI, +faststart) so the
    variant plays everywhere the full clip does.

    Returns True on success, False (logged) on any ffmpeg failure — callers treat
    a False as "fall back to the full clip" rather than an error.
    """
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(src_mp4),
        # Fit within 600x600, preserve aspect, force even dimensions (libx264).
        "-vf", "scale=600:600:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",
        # Hard bitrate ceiling — the real lever for BT-link transfer time.
        # Capped at 350k to keep the peak stream under what the Bluetooth
        # Display link sustains; above this, busy-scene bitrate spikes starve
        # the on-glasses buffer and the clip stutters/freezes mid-playback.
        "-maxrate", "350k",
        "-bufsize", "700k",
        "-r", "15",
        # Same iOS-compat flags as extract_clips (negative-DTS / color SEI).
        "-bf", "0",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        "-x264opts", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-c:a", "aac",
        "-b:a", "64k",
        "-movflags", "+faststart",
        str(dst_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, "stderr", "") or ""
        logger.error("Lens-variant transcode failed for %s: %s", src_mp4.name, stderr[-500:])
        return False
    logger.info(
        "Lens variant saved: %s (%.0f KB)",
        dst_mp4.name, dst_mp4.stat().st_size / 1024,
    )
    return True


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

        # Pre-generate the lens-optimized variant so the first on-glasses view is
        # already fast. Non-fatal: the ?profile=lens serving path regenerates on
        # demand if this is missing, and the full clip is always the fallback.
        await asyncio.to_thread(
            transcode_lens_variant, output_path, lens_variant_path(output_path)
        )

        # Pre-generate the "done state" last-frame poster the verification
        # oracle grounds against. Non-fatal: extract_done_frame regenerates on
        # demand at /verify time if this is missing.
        await asyncio.to_thread(extract_done_frame, output_path)

    return clip_paths, failed
