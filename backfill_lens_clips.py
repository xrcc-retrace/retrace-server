#!/usr/bin/env python3
"""Backfill the lens-optimized variants for existing per-step clips AND full uploads.

The native glasses lens streams a small, bitrate-capped variant over the
Bluetooth link — for per-step clips (GET /api/clips/{id}/step_{n}.mp4?profile=lens)
and for the full source recording behind the "Watch full recording" button
(GET /api/uploads/{filename}?profile=lens). New procedures generate both at
processing time; this one-shot fills them in for procedures created before the
variants existed, so the first on-glasses view is already fast instead of paying
an on-demand transcode.

Walks CLIPS_DIR/*/step_*.mp4 and UPLOAD_DIR/* (skipping the *_lens outputs) and
transcodes a variant beside each via services.clip_extractor.transcode_lens_variant.
Idempotent — already-present variants are skipped unless --force.

Usage:
    python backfill_lens_clips.py            # generate any missing variants
    python backfill_lens_clips.py --force    # re-transcode ALL variants
    python backfill_lens_clips.py --dry-run  # list what would be generated
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import CLIPS_DIR, UPLOAD_DIR
from services.clip_extractor import (
    LENS_VARIANT_SUFFIX,
    lens_variant_path,
    transcode_lens_variant,
    upload_lens_variant_path,
)

_UPLOAD_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_lens_clips")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-transcode variants even when they already exist.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated but do not run ffmpeg.",
    )
    return p.parse_args()


def _backfill(sources, dst_for, label_for, args) -> tuple[int, int, int]:
    """Transcode a lens variant for each source. Returns (generated, skipped, failed)."""
    generated = skipped = failed = 0
    for src in sources:
        dst = dst_for(src)
        if dst.exists() and not args.force:
            skipped += 1
            continue
        if args.dry_run:
            logger.info("[would generate] %s → %s", label_for(src), dst.name)
            generated += 1
            continue
        if transcode_lens_variant(src, dst):
            generated += 1
        else:
            failed += 1
    return generated, skipped, failed


def main() -> int:
    args = parse_args()

    generated = skipped = failed = 0

    # Per-step clips — skip the *_lens.mp4 outputs themselves.
    if CLIPS_DIR.exists():
        clips = sorted(
            p for p in CLIPS_DIR.glob("*/step_*.mp4")
            if not p.stem.endswith(LENS_VARIANT_SUFFIX)
        )
        g, s, f = _backfill(
            clips,
            dst_for=lens_variant_path,
            label_for=lambda p: f"{p.parent.name}/{p.name}",
            args=args,
        )
        generated += g; skipped += s; failed += f
    else:
        logger.warning("Clips directory not found at %s — skipping clips.", CLIPS_DIR)

    # Full source recordings (for the "Watch full recording" lens button).
    if UPLOAD_DIR.exists():
        uploads = sorted(
            p for p in UPLOAD_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in _UPLOAD_VIDEO_EXTS
            and not p.stem.endswith(LENS_VARIANT_SUFFIX)
        )
        g, s, f = _backfill(
            uploads,
            dst_for=upload_lens_variant_path,
            label_for=lambda p: p.name,
            args=args,
        )
        generated += g; skipped += s; failed += f
    else:
        logger.warning("Upload directory not found at %s — skipping uploads.", UPLOAD_DIR)

    if args.dry_run:
        logger.info("Dry run complete. Would generate %d variant(s); %d already present.",
                    generated, skipped)
    else:
        logger.info("Done. Generated %d, skipped %d (already present), failed %d.",
                    generated, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
