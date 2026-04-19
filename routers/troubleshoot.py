"""Troubleshoot-mode API surface — parallel to routers/learner.py.

Importing this module imports `services.troubleshoot`, which triggers
tool-handler registration with the shared session dispatcher.
"""

import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.procedure import Procedure
from models.session import (
    EphemeralTokenResponse,
    GeminiConfig,
    GeminiToolDefinition,
    SessionStartResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from models.troubleshoot import (
    HandoffRequest,
    ManualStatusResponse,
    ManualUploadResponse,
    TroubleshootSessionStartRequest,
    TroubleshootSessionStartResponse,
    TroubleshootSessionState,
)
from services import session as shared_session
from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import (
    GEMINI_CONFIG as LEARNER_GEMINI_CONFIG,
    TOOL_DEFINITIONS as LEARNER_TOOL_DEFINITIONS,
    build_system_prompt,
)
from services.troubleshoot import manual_ingest  # noqa: F401 — also triggers package import
from services.troubleshoot import session as ts_session
from services.troubleshoot.prompt import (
    DIAGNOSTIC_TOOL_DEFINITIONS,
    GEMINI_CONFIG_DIAGNOSTIC,
    build_diagnostic_prompt,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/troubleshoot", tags=["troubleshoot"])


# ---- Session lifecycle ------------------------------------------------------


@router.post("/session/start", response_model=TroubleshootSessionStartResponse)
async def start_session(req: TroubleshootSessionStartRequest):
    """Start a new diagnostic session and return everything the phone needs
    to open a Gemini Live WebSocket — system prompt, tool definitions,
    Gemini config, ephemeral token.
    """
    session = ts_session.create_diagnostic_session(voice=req.voice)
    system_prompt = build_diagnostic_prompt(voice=req.voice)
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=DIAGNOSTIC_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_DIAGNOSTIC["model"],
        voice=req.voice,
    )
    return TroubleshootSessionStartResponse(
        session_id=session["session_id"],
        system_prompt=system_prompt,
        tools=[GeminiToolDefinition(**t) for t in DIAGNOSTIC_TOOL_DEFINITIONS],
        gemini_config=GeminiConfig(model=GEMINI_CONFIG_DIAGNOSTIC["model"], voice=req.voice),
        ephemeral_token=EphemeralTokenResponse(**token_data),
    )


@router.post("/session/{session_id}/token", response_model=EphemeralTokenResponse)
async def refresh_token(session_id: str, handle: str | None = None):
    """Mint a fresh ephemeral token for a diagnostic session.

    Pass `?handle=<resumption_handle>` from the latest
    `sessionResumptionUpdate.new_handle` captured by the phone to continue
    the existing Gemini Live session with its compressed context intact.
    """
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != "troubleshoot":
        raise HTTPException(404, "Diagnostic session not found")

    voice = session.get("voice", GEMINI_CONFIG_DIAGNOSTIC["voice"])
    system_prompt = build_diagnostic_prompt(voice=voice)
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=DIAGNOSTIC_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_DIAGNOSTIC["model"],
        voice=voice,
        resumption_handle=handle,
    )
    return EphemeralTokenResponse(**token_data)


@router.post("/session/{session_id}/tool-call", response_model=ToolCallResponse)
async def tool_call(session_id: str, req: ToolCallRequest):
    """Handle a diagnostic-mode tool call forwarded from Gemini Live."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != "troubleshoot":
        raise HTTPException(404, "Diagnostic session not found")

    # Dispatch runs synchronously but some handlers kick off background work
    # (Gemini ranking, PDF ingest). The shared dispatcher key is
    # (session_type, tool_name), so this routes to the troubleshoot handlers.
    result = shared_session.handle_tool_call(session_id, req.tool_name, req.arguments)
    return ToolCallResponse(result=result)


@router.get("/session/{session_id}/state", response_model=TroubleshootSessionState)
async def session_state(session_id: str):
    """Current diagnostic session state — phase, identified product,
    candidate procedures, tool-call log.
    """
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != "troubleshoot":
        raise HTTPException(404, "Diagnostic session not found")

    return TroubleshootSessionState(
        session_id=session["session_id"],
        phase=session.get("phase", ts_session.PHASE_DISCOVERING),
        status=session["status"],
        started_at=session["started_at"],
        identified_product=session.get("identified_product"),
        candidate_procedures=session.get("candidate_procedures", []),
        generated_procedure_id=session.get("generated_procedure_id"),
        handoff_target=session.get("handoff_target"),
        handoff_learner_session_id=session.get("handoff_learner_session_id"),
        diagnostic_log=session.get("diagnostic_log", []),
    )


@router.post("/session/{session_id}/handoff", response_model=SessionStartResponse)
async def handoff(session_id: str, req: HandoffRequest):
    """Conclude the diagnostic session and open a Learner session on the
    chosen procedure. Returns the same shape as /api/learner/session/start
    so the phone can swap its WebSocket atomically.

    Idempotent: calling twice with the same target procedure returns the
    already-created learner session instead of minting a new one.
    """
    diag = shared_session.get_session(session_id)
    if not diag or diag.get("session_type") != "troubleshoot":
        raise HTTPException(404, "Diagnostic session not found")

    # Idempotency: if handoff was already executed to this same procedure,
    # just re-mint a learner token against the existing learner session
    # instead of creating a second one.
    existing_learner_id = diag.get("handoff_learner_session_id")
    if existing_learner_id and diag.get("handoff_target") == req.procedure_id:
        learner = shared_session.get_session(existing_learner_id)
        if learner:
            return await _build_learner_start_response(learner, req.voice)

    promoted = ts_session.promote_to_learner(
        session_id=session_id,
        procedure_id=req.procedure_id,
        auto_advance=req.auto_advance,
        voice=req.voice,
    )
    if not promoted:
        raise HTTPException(404, "Procedure not found")

    return await _build_learner_start_response(promoted["learner"], req.voice)


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End and remove a diagnostic session."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != "troubleshoot":
        raise HTTPException(404, "Diagnostic session not found")
    shared_session.delete_session(session_id)
    return {"ended": True}


# ---- Dashboard ------------------------------------------------------------


@router.get("/sessions")
async def list_sessions():
    """List all active + recent diagnostic sessions for the dashboard."""
    return ts_session.list_active_sessions()


# ---- Manual upload / ingest -----------------------------------------------


@router.post("/manuals/upload", response_model=ManualUploadResponse)
async def upload_manual(pdf: UploadFile = File(...)):
    """Upload a PDF manual. Returns the manual_id; actual ingest runs when
    Gemini calls generate_sop_from_manual with this id.
    """
    if pdf.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, f"Expected PDF, got {pdf.content_type}")
    data = await pdf.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    manual_id = manual_ingest.save_uploaded_manual(data, pdf.filename or "upload.pdf")
    entry = manual_ingest.get_manual_status(manual_id)
    return ManualUploadResponse(
        manual_id=manual_id,
        status=entry["status"],
        filename=entry["filename"],
    )


@router.get("/manuals/{manual_id}", response_model=ManualStatusResponse)
async def manual_status(manual_id: str):
    """Poll a manual's ingest status. When `status == 'ready'`, `procedure_id`
    is the newly-generated procedure."""
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


# ---- Helpers --------------------------------------------------------------


async def _build_learner_start_response(learner_session: dict, voice: str) -> SessionStartResponse:
    proc = learner_session["procedure"]
    system_prompt = build_system_prompt(
        proc,
        auto_advance=learner_session["auto_advance"],
        current_step_number=learner_session["current_step"],
    )
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=LEARNER_TOOL_DEFINITIONS,
        model=LEARNER_GEMINI_CONFIG["model"],
        voice=voice,
    )
    return SessionStartResponse(
        session_id=learner_session["session_id"],
        procedure=Procedure(**proc),
        system_prompt=system_prompt,
        tools=[GeminiToolDefinition(**t) for t in LEARNER_TOOL_DEFINITIONS],
        gemini_config=GeminiConfig(model=LEARNER_GEMINI_CONFIG["model"], voice=voice),
        ephemeral_token=EphemeralTokenResponse(**token_data),
    )
