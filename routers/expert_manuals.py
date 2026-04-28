"""Expert-mode PDF manual upload + ingestion + image serving.

Mirrors the video upload pattern in routers/expert.py but for PDFs:

    POST /api/expert/manuals/upload  → 202 + manual_id, kicks off background ingest
    GET  /api/expert/manuals/{id}    → poll ingest status
    GET  /api/manual_images/{pid}/step_{n}.png → serve a rendered PDF page

The actual ingest pipeline (Gemini Files API + structured-output extraction
+ per-step page rendering) lives in services.manual_ingest and is shared
with troubleshoot mode. This router just wraps it for the expert flow.
"""

import asyncio
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import GEMINI_API_KEY, MANUAL_IMAGES_DIR
from models.troubleshoot import ManualStatusResponse, ManualUploadResponse
from services import manual_ingest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["expert-manuals"])

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB — keeps Gemini Files API uploads sane


@router.post("/api/expert/manuals/upload", status_code=202, response_model=ManualUploadResponse)
async def upload_expert_manual(
    pdf: UploadFile = File(...),
    product_name: str = Form(...),
):
    """Receive a PDF manual and start background ingestion.

    Returns 202 immediately with `manual_id`; the phone polls
    `GET /api/expert/manuals/{manual_id}` to learn when it flips to
    `ready` (success — `procedure_id` populated) or `failed`.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")
    if pdf.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, f"Expected PDF, got {pdf.content_type}")
    product_name = product_name.strip()
    if not product_name:
        raise HTTPException(400, "product_name is required")

    data = await pdf.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(
            400,
            f"PDF too large: {len(data)} bytes (max {MAX_PDF_BYTES})",
        )

    manual_id = manual_ingest.save_uploaded_manual(data, pdf.filename or "upload.pdf")
    entry = manual_ingest.get_manual_status(manual_id)

    # Fire-and-forget background ingest
    asyncio.create_task(
        manual_ingest.ingest_manual(
            manual_id=manual_id,
            mode="expert",
            product_name=product_name,
        )
    )

    return ManualUploadResponse(
        manual_id=manual_id,
        status=entry["status"],
        filename=entry["filename"],
    )


@router.get("/api/expert/manuals/{manual_id}", response_model=ManualStatusResponse)
async def expert_manual_status(manual_id: str):
    """Poll an expert manual ingest. When `status == 'ready'`,
    `procedure_id` points at the freshly-extracted procedure.
    """
    entry = manual_ingest.get_manual_status(manual_id)
    if not entry:
        raise HTTPException(404, "Manual not found")
    return ManualStatusResponse(
        manual_id=entry["manual_id"],
        status=entry["status"],
        procedure_id=entry.get("procedure_id"),
        error=entry.get("error"),
        filename=entry["filename"],
    )


@router.get("/api/manual_images/{procedure_id}/step_{step_number}.png")
async def get_manual_image(procedure_id: str, step_number: int):
    """Serve a rendered PDF page for a manual-derived step."""
    file_path = MANUAL_IMAGES_DIR / procedure_id / f"step_{step_number}.png"
    if not file_path.exists():
        raise HTTPException(404, "Reference image not found")
    return FileResponse(str(file_path), media_type="image/png")
