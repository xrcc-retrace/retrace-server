import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import GEMINI_API_KEY, CLIPS_DIR, UPLOAD_DIR, REFERENCE_IMAGES_DIR
from models.procedure import ProcedureStep
from services.gemini_processor import process_expert_video
from services.clip_extractor import extract_clips
from services.database import (
    create_procedure_placeholder,
    get_procedure,
    save_procedure,
    set_procedure_reference_image,
    update_procedure_status,
)
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
    """Background coroutine: Gemini analysis + clip extraction."""
    try:
        gemini_output = await process_expert_video(str(video_path), GEMINI_API_KEY)

        # Clamp Gemini timestamps to the true video bounds. process_expert_video
        # now returns a dict (not a Pydantic instance) so its top-level fields
        # are dict keys; per-step entries are themselves dicts.
        clamped_steps: list[dict] = []
        for s in gemini_output["steps"]:
            raw_start, raw_end = s["timestamp_start"], s["timestamp_end"]
            start = max(0.0, min(raw_start, actual_duration))
            end = max(start, min(raw_end, actual_duration))
            if (start, end) != (raw_start, raw_end):
                logger.warning(
                    "Step %d timestamps clamped: (%.2f, %.2f) -> (%.2f, %.2f)",
                    s["step_number"], raw_start, raw_end, start, end,
                )
            step_dict = dict(s)
            step_dict["timestamp_start"] = start
            step_dict["timestamp_end"] = end
            clamped_steps.append(step_dict)

        # Extract clips
        try:
            clip_paths, failed_steps = await extract_clips(
                str(video_path), procedure_id, clamped_steps, CLIPS_DIR
            )
        except Exception as e:
            logger.error("Clip extraction failed entirely: %s", e)
            clip_paths = {}
            failed_steps = [s["step_number"] for s in clamped_steps]

        # Build step data with clip URLs + per-step status
        failed_set = set(failed_steps)
        steps: list[ProcedureStep] = []
        for s in clamped_steps:
            step_num = s["step_number"]
            if step_num in failed_set:
                clip_url = None
                clip_status = "failed"
            else:
                clip_url = f"/api/clips/{procedure_id}/step_{step_num}.mp4"
                clip_status = "ready"
            steps.append(ProcedureStep(
                step_number=step_num,
                title=s["title"],
                description=s["description"],
                timestamp_start=s["timestamp_start"],
                timestamp_end=s["timestamp_end"],
                tips=s.get("tips", []),
                warnings=s.get("warnings", []),
                completion_criteria=s.get("completion_criteria", ""),
                error_criteria=s.get("error_criteria", []),
                clip_url=clip_url,
                clip_status=clip_status,
            ))

        # Save completed procedure to database
        procedure_dict = {
            "id": procedure_id,
            "title": gemini_output["title"],
            "description": gemini_output["description"],
            "total_duration": actual_duration,
            "status": "completed_partial" if failed_steps else "completed",
            "icon_symbol": gemini_output["icon_symbol"],
            "icon_emoji": gemini_output["icon_emoji"],
            "category": gemini_output["category"],
        }
        steps_dicts = [s.model_dump() for s in steps]
        save_procedure(procedure_dict, steps_dicts)

        logger.info(
            "Procedure created: %s (%s, %d steps, status=%s)",
            procedure_id, gemini_output["title"], len(steps), procedure_dict["status"],
        )
    except Exception as e:
        logger.error("Background processing failed for %s: %s", procedure_id, e)
        update_procedure_status(procedure_id, "failed", str(e))
        try:
            video_path.unlink(missing_ok=True)
            logger.info("Deleted failed upload source: %s", video_path)
        except OSError as unlink_err:
            logger.warning("Failed to delete %s: %s", video_path, unlink_err)
