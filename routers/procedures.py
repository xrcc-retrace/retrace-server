import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import CLIPS_DIR
from services.database import get_procedure, list_procedures, delete_procedure

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["procedures"])


@router.get("/procedures/{procedure_id}")
async def get_procedure_endpoint(procedure_id: str):
    """Retrieve a generated procedure by ID."""
    proc = get_procedure(procedure_id)
    if not proc:
        raise HTTPException(404, "Procedure not found")
    return proc


@router.get("/procedures")
async def list_procedures_endpoint():
    """List all generated procedures."""
    return list_procedures()


@router.delete("/procedures/{procedure_id}")
async def delete_procedure_endpoint(procedure_id: str):
    """Delete a procedure and its steps."""
    if not delete_procedure(procedure_id):
        raise HTTPException(404, "Procedure not found")
    return {"deleted": True}


@router.get("/clips/{procedure_id}/step_{step_number}.mp4")
async def get_clip(procedure_id: str, step_number: int):
    """Serve a video clip for a specific procedure step."""
    clip_path = CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4")
