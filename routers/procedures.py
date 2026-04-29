import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import CLIPS_DIR
from services.database import (
    get_procedure,
    list_procedures,
    delete_procedure,
    update_procedure_fields,
    update_step_fields,
)


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
    ):
        raise HTTPException(404, "Step not found")
    return get_procedure(procedure_id)


@router.get("/clips/{procedure_id}/step_{step_number}.mp4")
async def get_clip(procedure_id: str, step_number: int):
    """Serve a video clip for a specific procedure step."""
    clip_path = CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4")
