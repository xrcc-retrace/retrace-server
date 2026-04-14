import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import KEYFRAMES_DIR, PROCEDURES_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["procedures"])


@router.get("/procedures/{procedure_id}")
async def get_procedure(procedure_id: str):
    """Retrieve a generated procedure by ID."""
    procedure_file = PROCEDURES_DIR / f"{procedure_id}.json"
    if not procedure_file.exists():
        raise HTTPException(404, "Procedure not found")
    with open(procedure_file) as f:
        return json.load(f)


@router.get("/procedures")
async def list_procedures():
    """List all generated procedures (id + title only)."""
    procedures = []
    for f in sorted(PROCEDURES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(f) as fh:
            data = json.load(fh)
            procedures.append({
                "id": data["id"],
                "title": data["title"],
                "description": data["description"],
                "total_duration": data["total_duration"],
                "step_count": len(data["steps"]),
                "created_at": data["created_at"],
            })
    return procedures


@router.get("/keyframes/{procedure_id}/step_{step_number}.jpg")
async def get_keyframe(procedure_id: str, step_number: int):
    """Serve a keyframe image for a specific procedure step."""
    keyframe_path = KEYFRAMES_DIR / procedure_id / f"step_{step_number}.jpg"
    if not keyframe_path.exists():
        raise HTTPException(404, "Keyframe not found")
    return FileResponse(str(keyframe_path), media_type="image/jpeg")
