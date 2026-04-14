import logging
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import GEMINI_API_KEY, CLIPS_DIR, UPLOAD_DIR
from models.procedure import ProcedureStep
from services.gemini_processor import process_expert_video
from services.clip_extractor import extract_clips
from services.database import save_procedure

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/expert", tags=["expert"])


@router.post("/upload")
async def upload_expert_video(video: UploadFile = File(...)):
    """Receive an expert recording, process with Gemini, extract step clips."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")

    procedure_id = str(uuid.uuid4())

    # Save uploaded file
    video_path = UPLOAD_DIR / f"{procedure_id}.mp4"
    logger.info("Saving uploaded video to %s", video_path)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info("Video saved: %.1f MB", file_size_mb)

    # Process with Gemini
    try:
        gemini_output = await process_expert_video(str(video_path), GEMINI_API_KEY)
    except Exception as e:
        logger.error("Gemini processing failed: %s", e)
        raise HTTPException(500, f"Video processing failed: {e}")

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
            clip_url=clip_url,
        ))

    created_at = datetime.now(timezone.utc).isoformat()

    # Save to database
    procedure_dict = {
        "id": procedure_id,
        "title": gemini_output.title,
        "description": gemini_output.description,
        "total_duration": gemini_output.total_duration,
        "source_video": str(video_path),
        "created_at": created_at,
    }
    steps_dicts = [s.model_dump() for s in steps]
    save_procedure(procedure_dict, steps_dicts)

    logger.info(
        "Procedure created: %s (%s, %d steps)",
        procedure_id, gemini_output.title, len(steps),
    )

    # Return full procedure for the client
    return {**procedure_dict, "steps": steps_dicts}
