"""Iconify icon → cached white PNG renderer.

Given an Iconify ID like ``lucide:refrigerator``, fetch the SVG (white
color baked in via Iconify's ``?color=#ffffff`` query param), rasterize
it to PNG with a transparent background, and cache on disk. Same icon
ID resolves to the same cache file regardless of which procedure
references it, so two procedures that share an icon don't double-render.

Uses ``cairosvg`` for rasterization — its only system dep is ``cairo``
itself, which is brought in via the ``pycairo`` wheel that ships
prebuilt for macOS (no Homebrew needed). Critically, cairosvg writes
PNGs with a real alpha channel, so the white icon stays visible against
the dark UI surface; reportlab's renderPM doesn't and was tried first.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cairosvg
import httpx

from config import DATA_DIR

logger = logging.getLogger(__name__)

ICONS_DIR = DATA_DIR / "icons"
ICONS_DIR.mkdir(parents=True, exist_ok=True)

# Iconify identifiers are lowercase letters, digits, and dashes. Both halves
# of the prefix:name pair share the same character class.
_NAME_RE = re.compile(r"^[a-z0-9-]+$")

ICONIFY_BASE = "https://api.iconify.design"
RENDER_PX = 96  # 3x the 40-pt badge — looks crisp on retina with headroom.

# Branded fallback when Gemini hallucinates an icon name or the picked
# Iconify ID 404s. Always white, same render path as picked icons, so
# the badge stays visually consistent. The router serves this PNG with
# `X-Icon-Fallback: true` so misses are debuggable.
FALLBACK_PREFIX = "lucide"
FALLBACK_NAME = "wrench"

# White is baked into the SVG at fetch time so the cached PNG is already
# ready to render against the dark UI without client-side tinting.
_WHITE_COLOR_PARAM = "%23ffffff"  # URL-encoded "#ffffff"


class IconNotFound(Exception):
    """Iconify CDN returned 404 for the requested {prefix}/{name}."""


class IconRenderError(Exception):
    """Anything else that goes wrong during render (network, parse, draw)."""


def cache_path(prefix: str, name: str) -> Path:
    return ICONS_DIR / f"{prefix}_{name}.png"


def render_iconify(prefix: str, name: str) -> Path:
    """Return the on-disk PNG path for an Iconify icon, rendering it if
    the cache is cold. Raises IconNotFound on a CDN 404 so the caller
    can map that to an HTTP 404 response.
    """
    if not _NAME_RE.match(prefix):
        raise IconRenderError(f"Invalid Iconify prefix: {prefix!r}")
    if not _NAME_RE.match(name):
        raise IconRenderError(f"Invalid Iconify name: {name!r}")

    out_path = cache_path(prefix, name)
    if out_path.exists():
        return out_path

    url = f"{ICONIFY_BASE}/{prefix}/{name}.svg?color={_WHITE_COLOR_PARAM}"
    try:
        resp = httpx.get(url, timeout=10.0)
    except httpx.HTTPError as e:
        raise IconRenderError(f"Iconify fetch failed: {e}") from e

    if resp.status_code == 404:
        raise IconNotFound(f"{prefix}:{name}")
    if resp.status_code != 200:
        raise IconRenderError(
            f"Iconify returned {resp.status_code} for {prefix}:{name}"
        )

    # Iconify returns "404" as a 200-status SVG body when the icon name is
    # bogus but well-formed (a quirk of their CDN). Treat that body as a miss.
    body = resp.content
    if b"<text" in body and b"404" in body:
        raise IconNotFound(f"{prefix}:{name}")

    tmp_png = out_path.with_suffix(".png.tmp")
    try:
        cairosvg.svg2png(
            bytestring=body,
            write_to=str(tmp_png),
            output_width=RENDER_PX,
            output_height=RENDER_PX,
        )
        os.replace(tmp_png, out_path)
        logger.info("Rendered Iconify icon %s:%s → %s", prefix, name, out_path.name)
        return out_path
    except Exception as e:
        if tmp_png.exists():
            try:
                tmp_png.unlink()
            except OSError:
                pass
        raise IconRenderError(
            f"Failed to rasterize {prefix}:{name}: {e}"
        ) from e
