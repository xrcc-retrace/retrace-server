"""Ask-AI mode API surface — a lightweight, audio-only learner voice search.

Mode-parallel to routers/troubleshoot.py but much smaller: one search tool, no
identify/diagnose phases. Importing this module imports services.ask_ai, which
registers the search_procedures handler with the shared dispatcher.

Flow: start → (Gemini calls search_procedures) → lens shows top matches → the
learner taps one → POST /session/{id}/handoff promotes it into a normal learner
coaching session (same SessionStartResponse as /api/learner/session/start), so
the phone can swap its WebSocket atomically.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from models.ask_ai import (
    AskAIHandoffRequest,
    AskAISessionStartRequest,
    AskAISessionStartResponse,
    AskAISessionState,
)
from models.procedure import Procedure
from models.session import (
    EphemeralTokenResponse,
    GeminiConfig,
    GeminiToolDefinition,
    SessionStartResponse,
    ToolCallRequest,
    ToolCallResponse,
)
from services import session as shared_session
from services.ask_ai import session as ask_session  # triggers handler registration via package import
from services.ask_ai.prompt import (
    ASK_AI_TOOL_DEFINITIONS,
    GEMINI_CONFIG_ASK_AI,
    build_ask_ai_prompt,
)
from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import (
    GEMINI_CONFIG as LEARNER_GEMINI_CONFIG,
    TOOL_DEFINITIONS as LEARNER_TOOL_DEFINITIONS,
    build_system_prompt,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ask_ai", tags=["ask_ai"])


# ---- Session lifecycle ------------------------------------------------------


@router.post("/session/start", response_model=AskAISessionStartResponse)
async def start_session(req: AskAISessionStartRequest):
    """Start a new Ask-AI session and return everything the phone needs to open
    a Gemini Live WebSocket. Audio-only — the phone simply never streams camera
    frames for this session.
    """
    session = ask_session.create_ask_ai_session(voice=req.voice)
    system_prompt = build_ask_ai_prompt(voice=req.voice)
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=ASK_AI_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_ASK_AI["model"],
        voice=req.voice,
    )
    return AskAISessionStartResponse(
        session_id=session["session_id"],
        system_prompt=system_prompt,
        tools=[GeminiToolDefinition(**t) for t in ASK_AI_TOOL_DEFINITIONS],
        gemini_config=GeminiConfig(model=GEMINI_CONFIG_ASK_AI["model"], voice=req.voice),
        ephemeral_token=EphemeralTokenResponse(**token_data),
    )


@router.post("/session/{session_id}/token", response_model=EphemeralTokenResponse)
async def refresh_token(session_id: str, handle: str | None = None):
    """Mint a fresh ephemeral token for an Ask-AI session (resumption-aware)."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != ask_session.SESSION_TYPE:
        raise HTTPException(404, "Ask-AI session not found")

    voice = session.get("voice", GEMINI_CONFIG_ASK_AI["voice"])
    system_prompt = build_ask_ai_prompt(voice=voice)
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=ASK_AI_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_ASK_AI["model"],
        voice=voice,
        resumption_handle=handle,
    )
    return EphemeralTokenResponse(**token_data)


@router.post("/session/{session_id}/tool-call", response_model=ToolCallResponse)
async def tool_call(session_id: str, req: ToolCallRequest):
    """Handle an Ask-AI tool call forwarded from Gemini Live."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != ask_session.SESSION_TYPE:
        raise HTTPException(404, "Ask-AI session not found")

    # The shared dispatcher key is (session_type, tool_name); run on a thread so
    # the Gemini ranking call doesn't block the event loop.
    result = await asyncio.to_thread(
        shared_session.handle_tool_call, session_id, req.tool_name, req.arguments
    )
    return ToolCallResponse(result=result)


@router.get("/session/{session_id}/state", response_model=AskAISessionState)
async def session_state(session_id: str):
    """Current Ask-AI session state — status + the latest search candidates."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != ask_session.SESSION_TYPE:
        raise HTTPException(404, "Ask-AI session not found")

    return AskAISessionState(
        session_id=session["session_id"],
        status=session["status"],
        started_at=session["started_at"],
        candidate_procedures=session.get("candidate_procedures", []),
        handoff_target=session.get("handoff_target"),
        handoff_learner_session_id=session.get("handoff_learner_session_id"),
    )


@router.post("/session/{session_id}/handoff", response_model=SessionStartResponse)
async def handoff(session_id: str, req: AskAIHandoffRequest):
    """Promote the picked procedure into a Learner session. Returns the same
    shape as /api/learner/session/start so the phone can swap its WebSocket.

    Idempotent: calling twice with the same target returns the already-created
    learner session instead of minting a second one.
    """
    ask = shared_session.get_session(session_id)
    if not ask or ask.get("session_type") != ask_session.SESSION_TYPE:
        raise HTTPException(404, "Ask-AI session not found")

    existing_learner_id = ask.get("handoff_learner_session_id")
    if existing_learner_id and ask.get("handoff_target") == req.procedure_id:
        learner = shared_session.get_session(existing_learner_id)
        if learner:
            return await _build_learner_start_response(learner, req.voice)

    promoted = ask_session.promote_to_learner(
        session_id=session_id,
        procedure_id=req.procedure_id,
        auto_advance=req.auto_advance,
        active_monitoring=req.active_monitoring,
        voice=req.voice,
    )
    if not promoted:
        raise HTTPException(404, "Procedure not found")

    return await _build_learner_start_response(promoted["learner"], req.voice)


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End and remove an Ask-AI session."""
    session = shared_session.get_session(session_id)
    if not session or session.get("session_type") != ask_session.SESSION_TYPE:
        raise HTTPException(404, "Ask-AI session not found")
    shared_session.delete_session(session_id)
    return {"ended": True}


# ---- Helpers --------------------------------------------------------------


async def _build_learner_start_response(learner_session: dict, voice: str) -> SessionStartResponse:
    proc = learner_session["procedure"]
    system_prompt = build_system_prompt(
        proc,
        auto_advance=learner_session["auto_advance"],
        active_monitoring=learner_session["active_monitoring"],
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
