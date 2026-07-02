"""Lens-variant refresh — keeps every `_lens.mp4` in sync with the encode ladder.

The glasses lens streams the small `?profile=lens` variants that
`services.clip_extractor.transcode_lens_variant` produces. When the encode
ladder changes (bump `LENS_ENCODE_VERSION` in clip_extractor.py), variants
already on disk are stale — including on the deployed Lightsail box, whose
data/ never leaves the box. This module owns regenerating them:

- `refresh_all(force)` walks CLIPS_DIR step clips + UPLOAD_DIR source videos
  and transcodes each missing (or, with force, every) variant.
- `refresh_if_stale()` compares a marker file in DATA_DIR against
  LENS_ENCODE_VERSION and force-regenerates on mismatch, writing the marker
  when the pass completes cleanly. Ordinary deploys (no version bump) pay
  nothing beyond one file read.
- `schedule_lens_refresh()` runs that check in the background from main.py's
  lifespan (mirrors schedule_fixtures_load / schedule_reextract), so a deploy
  + restart is all it takes to retrigger the clips on the server.

Regeneration is sequential on purpose — one ffmpeg at a time keeps the small
deploy box responsive — and each transcode writes atomically (temp + rename in
transcode_lens_variant), so requests mid-refresh serve the old variant until
the new one lands.
"""

import asyncio
import logging
from pathlib import Path

from config import CLIPS_DIR, DATA_DIR, UPLOAD_DIR
from services.clip_extractor import (
    LENS_ENCODE_VERSION,
    LENS_VARIANT_SUFFIX,
    lens_variant_path,
    transcode_lens_variant,
    upload_lens_variant_path,
)

logger = logging.getLogger(__name__)

MARKER_PATH = DATA_DIR / "lens_encode_version"

_UPLOAD_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def _sources() -> list[tuple[Path, Path]]:
    """All (source, variant_destination) pairs: per-step clips + full uploads."""
    pairs: list[tuple[Path, Path]] = []
    if CLIPS_DIR.exists():
        pairs.extend(
            (p, lens_variant_path(p))
            for p in sorted(CLIPS_DIR.glob("*/step_*.mp4"))
            if not p.stem.endswith(LENS_VARIANT_SUFFIX)
        )
    if UPLOAD_DIR.exists():
        pairs.extend(
            (p, upload_lens_variant_path(p))
            for p in sorted(UPLOAD_DIR.iterdir())
            if p.is_file()
            and p.suffix.lower() in _UPLOAD_VIDEO_EXTS
            and not p.stem.endswith(LENS_VARIANT_SUFFIX)
        )
    return pairs


def refresh_all(force: bool = False, dry_run: bool = False) -> tuple[int, int, int]:
    """Transcode a lens variant for each source. Returns (generated, skipped, failed).

    Synchronous (one ffmpeg at a time) — callers that must not block the event
    loop wrap it in asyncio.to_thread. `dry_run` counts what would be generated
    without running ffmpeg.
    """
    generated = skipped = failed = 0
    for src, dst in _sources():
        if dst.exists() and not force:
            skipped += 1
            continue
        if dry_run:
            logger.info("[would generate] %s → %s", src.name, dst.name)
            generated += 1
            continue
        if transcode_lens_variant(src, dst):
            generated += 1
        else:
            failed += 1
    return generated, skipped, failed


def read_marker() -> int | None:
    try:
        return int(MARKER_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_marker() -> None:
    MARKER_PATH.write_text(f"{LENS_ENCODE_VERSION}\n")


async def refresh_if_stale() -> None:
    """Force-regenerate every variant when the on-disk marker is stale.

    The marker is written only after a clean pass (no failures), so a crash or
    partial failure retries on the next startup rather than silently leaving a
    mix of old- and new-ladder variants behind a current marker.
    """
    marker = read_marker()
    if marker == LENS_ENCODE_VERSION:
        logger.info("Lens variants current (encode v%d) — no refresh needed", marker)
        return

    logger.info(
        "Lens variants stale (marker=%s, current=v%d) — regenerating all in background",
        marker, LENS_ENCODE_VERSION,
    )
    generated, skipped, failed = await asyncio.to_thread(refresh_all, True)
    if failed:
        logger.warning(
            "Lens refresh finished with failures: %d generated, %d failed — "
            "marker NOT updated, will retry next startup",
            generated, failed,
        )
        return
    write_marker()
    logger.info(
        "Lens refresh complete: %d variant(s) regenerated at encode v%d",
        generated, LENS_ENCODE_VERSION,
    )


def schedule_lens_refresh() -> asyncio.Task:
    """Fire-and-forget wrapper for the lifespan (mirrors schedule_fixtures_load).

    Returns the Task so the caller can cancel it on shutdown.
    """
    return asyncio.create_task(refresh_if_stale())
