import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import GEMINI_API_KEY, CLIPS_DIR, UPLOAD_DIR
from models.procedure import ProcedureStep
from services.gemini_processor import process_expert_video
from services.clip_extractor import extract_clips
from services.database import (
    create_procedure_placeholder,
    save_procedure,
    update_procedure_status,
)

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
        shutil.copyfileobj(video.file, f)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info("Video saved: %.1f MB", file_size_mb)

    created_at = datetime.now(timezone.utc).isoformat()
    create_procedure_placeholder(procedure_id, str(video_path), created_at)

    # Fire-and-forget background processing
    asyncio.create_task(_process_video(procedure_id, video_path))

    return {"id": procedure_id, "status": "processing"}


async def _process_video(procedure_id: str, video_path: Path):
    """Background coroutine: Gemini analysis + clip extraction."""
    try:
        gemini_output = await process_expert_video(str(video_path), GEMINI_API_KEY)

        # Extract clips for each step
        steps_dicts = [s.model_dump() for s in gemini_output.steps]
        try:
            clip_paths = await extract_clips(
                str(video_path), procedure_id, steps_dicts, CLIPS_DIR
            )
        except Exception as e:
            logger.error("Clip extraction failed: %s", e)
            clip_paths = {}

        # Build step data with clip URLs
        steps = []
        for s in gemini_output.steps:
            clip_url = None
            if s.step_number in clip_paths:
                clip_url = f"/api/clips/{procedure_id}/step_{s.step_number}.mp4"
            steps.append(ProcedureStep(
                step_number=s.step_number,
                title=s.title,
                description=s.description,
                timestamp_start=s.timestamp_start,
                timestamp_end=s.timestamp_end,
                tips=s.tips,
                warnings=s.warnings,
                completion_criteria=s.completion_criteria,
                clip_url=clip_url,
            ))

        # Save completed procedure to database
        procedure_dict = {
            "id": procedure_id,
            "title": gemini_output.title,
            "description": gemini_output.description,
            "total_duration": gemini_output.total_duration,
        }
        steps_dicts = [s.model_dump() for s in steps]
        save_procedure(procedure_dict, steps_dicts)

        logger.info(
            "Procedure created: %s (%s, %d steps)",
            procedure_id, gemini_output.title, len(steps),
        )
    except Exception as e:
        logger.error("Background processing failed for %s: %s", procedure_id, e)
        update_procedure_status(procedure_id, "failed", str(e))
