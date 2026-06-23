#!/usr/bin/env python3
"""Backfill the lens-optimized clip variant (step_N_lens.mp4) for existing clips.

The native glasses lens streams a small, bitrate-capped variant of each step
clip (served at GET /api/clips/{id}/step_{n}.mp4?profile=lens). New procedures
generate it at extraction time; this one-shot fills it in for clips created
before the variant existed, so the first on-glasses view of any clip is already
fast instead of paying an on-demand transcode.

Walks CLIPS_DIR/*/step_*.mp4 (skipping the *_lens.mp4 outputs) and transcodes a
variant beside each one via services.clip_extractor.transcode_lens_variant.
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

from config import CLIPS_DIR
from services.clip_extractor import (
    LENS_VARIANT_SUFFIX,
    lens_variant_path,
    transcode_lens_variant,
)

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


def main() -> int:
    args = parse_args()

    if not CLIPS_DIR.exists():
        logger.error("Clips directory not found at %s", CLIPS_DIR)
        return 2

    # Full clips only — skip the *_lens.mp4 outputs themselves.
    clips = sorted(
        p for p in CLIPS_DIR.glob("*/step_*.mp4")
        if not p.stem.endswith(LENS_VARIANT_SUFFIX)
    )
    if not clips:
        logger.info("No clips found under %s — nothing to backfill.", CLIPS_DIR)
        return 0

    generated = skipped = failed = 0
    for clip in clips:
        dst = lens_variant_path(clip)
        rel = f"{clip.parent.name}/{clip.name}"

        if dst.exists() and not args.force:
            skipped += 1
            continue

        if args.dry_run:
            logger.info("[would generate] %s → %s", rel, dst.name)
            generated += 1
            continue

        if transcode_lens_variant(clip, dst):
            generated += 1
        else:
            failed += 1

    if args.dry_run:
        logger.info("Dry run complete. Would generate %d variant(s); %d already present.",
                    generated, skipped)
    else:
        logger.info("Done. Generated %d, skipped %d (already present), failed %d.",
                    generated, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
