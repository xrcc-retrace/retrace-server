"""Iconify icon proxy + on-disk PNG cache.

GET /api/icons/{prefix}/{name}.png
  → 200 image/png  — white-on-transparent rasterized icon
  → 200 image/png  — branded fallback icon (lucide:wrench) when the
                     requested ID 404s on Iconify or fails to render.
                     Sent with `X-Icon-Fallback: true` for debugging.
  → 400            — malformed prefix / name (rejected by the renderer)

Falling back transparently keeps the iOS card's icon badge always
white/on-brand. Emojis aren't tintable on iOS (Apple Color Emoji is a
bitmap font), so a colored emoji in that slot would break the white-only
design.

Cache lives on disk (see services.icon_renderer.ICONS_DIR). Two procedures
referencing the same Iconify ID share the cache entry. The fallback wrench
is cached once at first miss.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.icon_renderer import (
    FALLBACK_NAME,
    FALLBACK_PREFIX,
    IconNotFound,
    IconRenderError,
    render_iconify,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/icons", tags=["icons"])


@router.get("/{prefix}/{name}.png")
async def get_icon(prefix: str, name: str):
    is_fallback = False
    try:
        path = render_iconify(prefix, name)
    except IconNotFound:
        logger.info("Icon miss %s:%s — serving fallback", prefix, name)
        path = _fallback_path()
        is_fallback = True
    except IconRenderError as e:
        # Malformed prefix/name — caller bug, surface as 400.
        if "Invalid Iconify" in str(e):
            raise HTTPException(400, str(e))
        # Other render errors (bad SVG, rasterizer failure) — degrade
        # to the fallback rather than surfacing a 5xx, since the iOS
        # card has no graceful recovery for that.
        logger.warning("Icon render error for %s:%s — %s; serving fallback",
                       prefix, name, e)
        path = _fallback_path()
        is_fallback = True
    headers = {"X-Icon-Fallback": "true"} if is_fallback else None
    return FileResponse(str(path), media_type="image/png", headers=headers)


def _fallback_path():
    """Return the cached fallback wrench PNG, rendering it on first call."""
    try:
        return render_iconify(FALLBACK_PREFIX, FALLBACK_NAME)
    except (IconNotFound, IconRenderError) as e:
        # If the fallback itself fails (network down on first ever request),
        # we have no graceful answer left. 502 the request — iOS will fall
        # through to its step-count branch.
        logger.error("Icon fallback (%s:%s) failed: %s",
                     FALLBACK_PREFIX, FALLBACK_NAME, e)
        raise HTTPException(502, "Icon render failed and fallback unavailable")
