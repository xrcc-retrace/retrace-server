import asyncio
import logging
import re
import subprocess
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import CLIPS_DIR, FFMPEG_PATH
from services.clip_extractor import (
    extract_done_frame,
    lens_variant_path,
    transcode_lens_variant,
)
from services.database import (
    get_procedure,
    list_procedures,
    delete_procedure,
    update_procedure_fields,
    update_step_fields,
)

# Procedure ids are server-generated UUIDs; this allowlist keeps a hostile
# `procedure_id` path segment from escaping CLIPS_DIR when building clip paths.
_PROCEDURE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def _validate_procedure_id(procedure_id: str) -> None:
    if not _PROCEDURE_ID_RE.fullmatch(procedure_id):
        raise HTTPException(400, "Invalid procedure id")


class ProcedureUpdateBody(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    step_order: Optional[list[int]] = None


class StepUpdateBody(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tips: Optional[list[str]] = None
    warnings: Optional[list[str]] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    error_criteria: Optional[list[str]] = None
    visual_verifiability: Optional[str] = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["procedures"])


# iOS's URLCache silently returns stale data under the
# `.returnCacheDataElseLoad` cache policy that ProcedureAPIService uses by
# default — see dat-ios-test ProcedureAPIService.swift:24. New uploads
# wouldn't appear in the workflow list until a forceRefresh fired.
# `no-store` on the response prevents URLCache from holding the entry at
# all, so every fetch hits the network. Cheap on the server side
# (procedures list is just a SQLite SELECT) and removes the staleness
# class of bugs without depending on iOS pull-to-refresh hygiene.
_NO_STORE = {"Cache-Control": "no-store, no-cache, must-revalidate"}


@router.get("/procedures/{procedure_id}")
async def get_procedure_endpoint(procedure_id: str):
    """Retrieve a generated procedure by ID."""
    proc = get_procedure(procedure_id)
    if not proc:
        raise HTTPException(404, "Procedure not found")
    return JSONResponse(content=proc, headers=_NO_STORE)


@router.get("/procedures")
async def list_procedures_endpoint():
    """List all generated procedures."""
    return JSONResponse(content=list_procedures(), headers=_NO_STORE)


@router.delete("/procedures/{procedure_id}")
async def delete_procedure_endpoint(procedure_id: str):
    """Delete a procedure and its steps."""
    if not delete_procedure(procedure_id):
        raise HTTPException(404, "Procedure not found")
    return {"deleted": True}


@router.put("/procedures/{procedure_id}")
async def update_procedure_endpoint(procedure_id: str, body: ProcedureUpdateBody):
    """Update procedure title, description, or step order."""
    if not update_procedure_fields(
        procedure_id,
        title=body.title,
        description=body.description,
        step_order=body.step_order,
    ):
        raise HTTPException(404, "Procedure not found")
    return get_procedure(procedure_id)


@router.put("/procedures/{procedure_id}/steps/{step_number}")
async def update_step_endpoint(procedure_id: str, step_number: int, body: StepUpdateBody):
    """Update a single step's fields."""
    if not update_step_fields(
        procedure_id,
        step_number,
        title=body.title,
        description=body.description,
        tips=body.tips,
        warnings=body.warnings,
        timestamp_start=body.timestamp_start,
        timestamp_end=body.timestamp_end,
        error_criteria=body.error_criteria,
        visual_verifiability=body.visual_verifiability,
    ):
        raise HTTPException(404, "Step not found")
    return get_procedure(procedure_id)


@router.get("/clips/{procedure_id}/step_{step_number}.mp4")
async def get_clip(procedure_id: str, step_number: int, profile: Optional[str] = None):
    """Serve a video clip for a specific procedure step.

    `?profile=lens` serves a small, bitrate-capped variant for the glasses lens
    (transcoded from the full clip on first request and cached beside it). Any
    other / no profile serves the full-resolution clip. Unknown profiles and a
    failed transcode both fall back to the full clip — the param is purely an
    optimization, never a hard dependency, so the iOS client can request it even
    against an older server.
    """
    _validate_procedure_id(procedure_id)
    clip_path = CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")

    if profile == "lens":
        lens_path = lens_variant_path(clip_path)
        # Variants are pre-generated at extraction and refreshed at startup
        # (services/lens_variants.py), so this is normally a cache hit. The
        # on-demand transcode is a fallback — run it off the event loop so a
        # long encode doesn't freeze every other request (incl. /verify polls).
        if lens_path.exists() or await asyncio.to_thread(
            transcode_lens_variant, clip_path, lens_path
        ):
            return FileResponse(str(lens_path), media_type="video/mp4")
        # Transcode failed — degrade to the full clip rather than erroring.
        logger.warning("Lens variant unavailable for %s step %s; serving full clip",
                       procedure_id, step_number)

    return FileResponse(str(clip_path), media_type="video/mp4")


@router.get("/clips/{procedure_id}/step_{step_number}.jpg")
async def get_clip_poster(procedure_id: str, step_number: int):
    """Serve a single-frame JPEG poster for a step's clip.

    Used as the lens-side thumbnail in the Coaching `.referenceExpanded`
    surface — `MWDATDisplay.Image(uri:)` only accepts HTTPS images, not
    embedded video frames. Extracts the first frame on demand via ffmpeg
    and caches the JPEG on disk next to the source clip. Subsequent
    requests serve the cached file directly.
    """
    _validate_procedure_id(procedure_id)
    clip_path = CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")

    poster_path = clip_path.with_suffix(".jpg")
    if not poster_path.exists():
        # First-frame extraction. -ss before -i is the "fast" seek (decodes
        # nothing before the keyframe) which is fine at t=0. -frames:v 1
        # writes exactly one frame; -q:v 2 is the higher-quality end of the
        # JPEG range (1-31, lower is better). -y overwrites stale partial
        # files in case of a prior crash.
        try:
            subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-ss", "0",
                    "-i", str(clip_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(poster_path),
                ],
                check=True,
                capture_output=True,
                timeout=15,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("Poster extraction failed for %s step %s: %s",
                         procedure_id, step_number, e)
            raise HTTPException(500, "Failed to extract poster") from e

    return FileResponse(str(poster_path), media_type="image/jpeg")


@router.get("/clips/{procedure_id}/step_{step_number}_done.jpg")
async def get_clip_done_frame(procedure_id: str, step_number: int):
    """Serve the last-frame ('done state') JPEG for a step's clip.

    This is the canonical picture of what the scene looks like once the step
    is complete. The verification oracle uses it for image-to-image grounding.
    Extracted on demand (last frame of the clip) and cached beside the clip.
    """
    _validate_procedure_id(procedure_id)
    clip_path = CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")

    done_path = await asyncio.to_thread(extract_done_frame, clip_path)
    if done_path is None:
        raise HTTPException(500, "Failed to extract done frame")
    return FileResponse(str(done_path), media_type="image/jpeg")
