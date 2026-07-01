import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import GEMINI_API_KEY, UPLOAD_DIR, REFERENCE_IMAGES_DIR
from services.database import (
    create_procedure_placeholder,
    get_procedure,
    set_procedure_reference_image,
    update_procedure_status,
)
from services.reprocess import process_video_into_procedure
from services.video_probe import probe_duration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/expert", tags=["expert"])


@router.post("/upload", status_code=202)
async def upload_expert_video(video: UploadFile = File(...)):
    """Receive an expert recording and start background processing."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")

    procedure_id = str(uuid.uuid4())

    # Save uploaded file, preserving original extension
    ext = Path(video.filename).suffix if video.filename else ".mp4"
    video_path = UPLOAD_DIR / f"{procedure_id}{ext}"
    logger.info("Saving uploaded video to %s", video_path)
    with open(video_path, "wb") as f:
        await asyncio.to_thread(shutil.copyfileobj, video.file, f)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info("Video saved: %.1f MB", file_size_mb)

    # Authoritative duration — if ffprobe can't read it, the upload is bad
    try:
        actual_duration = await probe_duration(str(video_path))
    except Exception as exc:
        logger.error("ffprobe failed on upload %s: %s", video_path, exc)
        raise HTTPException(400, f"Could not probe uploaded video: {exc}") from exc

    # millisecond precision so iOS's ISO8601DateFormatter.withFractionalSeconds
    # can parse this — Python's default 6-digit microsecond output silently
    # fails to parse on iOS and the procedure won't appear in workflows.
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    create_procedure_placeholder(procedure_id, str(video_path), created_at)

    # Fire-and-forget background processing
    asyncio.create_task(_process_video(procedure_id, video_path, actual_duration))

    return {"id": procedure_id, "status": "processing"}


@router.post("/procedures/{procedure_id}/reference-image")
async def upload_reference_image(procedure_id: str, file: UploadFile = File(...)):
    """Attach a reference image (e.g. a diagram) to a procedure.

    The image is shown on the learner's phone during coaching. Saved as
    {procedure_id}.{ext} and served from /api/reference_images/.
    """
    if get_procedure(procedure_id) is None:
        raise HTTPException(status_code=404, detail="Procedure not found")

    ext = Path(file.filename).suffix.lstrip(".") if file.filename else ""
    if ext.lower() not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    image_path = REFERENCE_IMAGES_DIR / f"{procedure_id}.{ext}"
    logger.info("Saving reference image to %s", image_path)
    with open(image_path, "wb") as out:
        await asyncio.to_thread(shutil.copyfileobj, file.file, out)

    # Reference images always live at the same path ({id}.{ext}), so a replace
    # produces a byte-identical URL. iOS (and any URL-keyed cache) would then
    # serve the stale image. Append a version token from the file's mtime so
    # each upload yields a unique URL → guaranteed cache miss on replace. The
    # file-serving route ignores the query string.
    version = int(image_path.stat().st_mtime)
    url = f"/api/reference_images/{procedure_id}.{ext}?v={version}"
    set_procedure_reference_image(procedure_id, url)

    return {"reference_image_url": url}


@router.delete("/procedures/{procedure_id}/reference-image")
async def delete_reference_image(procedure_id: str):
    """Remove a procedure's reference image (clears the DB field + deletes the file)."""
    proc = get_procedure(procedure_id)
    if proc is None:
        raise HTTPException(status_code=404, detail="Procedure not found")

    url = proc.get("reference_image_url")
    if url:
        try:
            # Strip any "?v=" cache-busting query before deriving the filename.
            filename = Path(urlparse(url).path).name
            (REFERENCE_IMAGES_DIR / filename).unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete reference image file for %s", procedure_id)

    set_procedure_reference_image(procedure_id, None)
    return {"reference_image_url": None}


async def _process_video(procedure_id: str, video_path: Path, actual_duration: float):
    """Background coroutine: Gemini analysis + clip extraction.

    Delegates the extraction pipeline to services.reprocess so a fresh upload and
    an in-place re-extraction run identical code. On failure the upload path marks
    the procedure failed and deletes the orphaned source video.
    """
    try:
        await process_video_into_procedure(procedure_id, video_path, actual_duration)
    except Exception as e:
        logger.error("Background processing failed for %s: %s", procedure_id, e)
        update_procedure_status(procedure_id, "failed", str(e))
        try:
            video_path.unlink(missing_ok=True)
            logger.info("Deleted failed upload source: %s", video_path)
        except OSError as unlink_err:
            logger.warning("Failed to delete %s: %s", video_path, unlink_err)
