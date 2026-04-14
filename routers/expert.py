import json
import logging
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import GEMINI_API_KEY, KEYFRAMES_DIR, PROCEDURES_DIR, UPLOAD_DIR
from models.procedure import Procedure, ProcedureStep
from services.gemini_processor import process_expert_video
from services.keyframe_extractor import extract_keyframes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/expert", tags=["expert"])


@router.post("/upload")
async def upload_expert_video(video: UploadFile = File(...)):
    """Receive an expert recording, process with Gemini, extract keyframes."""
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

    # Extract keyframes
    steps_dicts = [s.model_dump() for s in gemini_output.steps]
    try:
        keyframe_paths = await extract_keyframes(
            str(video_path), procedure_id, steps_dicts, KEYFRAMES_DIR
        )
    except Exception as e:
        logger.error("Keyframe extraction failed: %s", e)
        keyframe_paths = {}

    # Build procedure with keyframe URLs
    steps = []
    for s in gemini_output.steps:
        keyframe_url = None
        if s.step_number in keyframe_paths:
            keyframe_url = f"/api/keyframes/{procedure_id}/step_{s.step_number}.jpg"
        steps.append(ProcedureStep(
            step_number=s.step_number,
            title=s.title,
            description=s.description,
            timestamp_start=s.timestamp_start,
            timestamp_end=s.timestamp_end,
            tips=s.tips,
            warnings=s.warnings,
            keyframe_url=keyframe_url,
        ))

    procedure = Procedure(
        id=procedure_id,
        title=gemini_output.title,
        description=gemini_output.description,
        steps=steps,
        total_duration=gemini_output.total_duration,
        source_video=str(video_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Save procedure JSON
    procedure_file = PROCEDURES_DIR / f"{procedure_id}.json"
    with open(procedure_file, "w") as f:
        json.dump(procedure.model_dump(), f, indent=2)

    logger.info(
        "Procedure created: %s (%s, %d steps)",
        procedure_id, procedure.title, len(steps),
    )

    return procedure.model_dump()
