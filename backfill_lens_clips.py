#!/usr/bin/env python3
"""Backfill the lens-optimized variants for existing per-step clips AND full uploads.

The native glasses lens streams a small, bitrate-capped variant over the
Bluetooth link — for per-step clips (GET /api/clips/{id}/step_{n}.mp4?profile=lens)
and for the full source recording behind the "Watch full recording" button
(GET /api/uploads/{filename}?profile=lens). New procedures generate both at
processing time, and the server's startup refresh (services/lens_variants.py)
force-regenerates everything when LENS_ENCODE_VERSION bumps — this CLI is the
manual driver for the same pass (e.g. to pre-warm before a restart, or to
re-encode after hand-tweaking ffmpeg flags without a version bump).

Delegates to services.lens_variants.refresh_all so the CLI, the startup
refresh, and extraction-time pre-generation can never drift. A clean --force
run also writes the version marker, so the next startup won't redo the work.

Usage:
    python backfill_lens_clips.py            # generate any missing variants
    python backfill_lens_clips.py --force    # re-transcode ALL variants
    python backfill_lens_clips.py --dry-run  # list what would be generated
"""

from __future__ import annotations

import argparse
import logging
import sys

from services.lens_variants import refresh_all, write_marker

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
    generated, skipped, failed = refresh_all(force=args.force, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry run complete. Would generate %d variant(s); %d already present.",
                    generated, skipped)
    else:
        logger.info("Done. Generated %d, skipped %d (already present), failed %d.",
                    generated, skipped, failed)
        # A clean forced pass leaves every variant on the current encode
        # ladder — stamp the marker so the startup refresh doesn't redo it.
        if args.force and not failed:
            write_marker()
            logger.info("Version marker updated.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
