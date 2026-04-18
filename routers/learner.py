import logging

from fastapi import APIRouter, HTTPException

from models.procedure import Procedure
from models.session import (
    EphemeralTokenResponse,
    GeminiConfig,
    GeminiToolDefinition,
    SessionStartRequest,
    SessionStartResponse,
    SessionState,
    StepCompletion,
    ToolCallRequest,
    ToolCallResponse,
)
from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import build_system_prompt, GEMINI_CONFIG, TOOL_DEFINITIONS
from services.session import (
    create_session,
    delete_session,
    get_context_summary,
    get_session,
    handle_tool_call,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learner", tags=["learner"])

# Server-authoritative voice list — iOS fetches this instead of hardcoding
VOICE_OPTIONS = [
    {"name": "Puck", "description": "Upbeat and energetic", "preview_url": "/static/voice_previews/Puck.wav"},
    {"name": "Charon", "description": "Calm and informative", "preview_url": "/static/voice_previews/Charon.wav"},
    {"name": "Kore", "description": "Clear and firm", "preview_url": "/static/voice_previews/Kore.wav"},
    {"name": "Fenrir", "description": "Warm and encouraging", "preview_url": "/static/voice_previews/Fenrir.wav"},
    {"name": "Aoede", "description": "Bright and expressive", "preview_url": "/static/voice_previews/Aoede.wav"},
    {"name": "Zephyr", "description": "Bright and breezy", "preview_url": "/static/voice_previews/Zephyr.wav"},
    {"name": "Enceladus", "description": "Soft and breathy", "preview_url": "/static/voice_previews/Enceladus.wav"},
    {"name": "Leda", "description": "Steady and neutral", "preview_url": "/static/voice_previews/Leda.wav"},
]


@router.get("/voices")
async def list_voices():
    """Return available voice options with preview URLs."""
    return VOICE_OPTIONS


@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(req: SessionStartRequest):
    """Start a learning session for a procedure.

    Returns everything the phone needs to open a Gemini Live WebSocket:
    the curated system prompt, tool definitions, and Gemini config.
    """
    try:
        session = create_session(
            req.procedure_id,
            auto_advance=req.auto_advance,
            voice=req.voice,
            starting_step=req.starting_step,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not session:
        raise HTTPException(404, "Procedure not found")

    proc = session["procedure"]
    voice = session["voice"]
    system_prompt = build_system_prompt(
        proc,
        auto_advance=session["auto_advance"],
        current_step_number=session["current_step"],
    )

    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=TOOL_DEFINITIONS,
        model=GEMINI_CONFIG["model"],
        voice=voice,
    )

    return SessionStartResponse(
        session_id=session["session_id"],
        procedure=Procedure(**proc),
        system_prompt=system_prompt,
        tools=[GeminiToolDefinition(**t) for t in TOOL_DEFINITIONS],
        gemini_config=GeminiConfig(model=GEMINI_CONFIG["model"], voice=voice),
        ephemeral_token=EphemeralTokenResponse(**token_data),
    )


@router.post("/session/{session_id}/token", response_model=EphemeralTokenResponse)
async def refresh_token(session_id: str, handle: str | None = None):
    """Issue a fresh ephemeral token for an existing session.

    Pass `?handle=<resumption_handle>` (the most recent
    `sessionResumptionUpdate.new_handle` captured by the phone) to mint a
    token that reopens the prior Gemini Live session with compressed context
    preserved. Omit for a brand-new session.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    proc = session["procedure"]
    # Anchor the regenerated system prompt on the step the learner is actually
    # on. Otherwise a resumed WebSocket reopens with step-1 guidance baked
    # into the token's locked system_instruction, and if the client-side
    # context-summary inject fails we silently desync.
    system_prompt = build_system_prompt(
        proc,
        auto_advance=session["auto_advance"],
        current_step_number=session["current_step"],
    )

    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=TOOL_DEFINITIONS,
        model=GEMINI_CONFIG["model"],
        voice=session.get("voice", GEMINI_CONFIG["voice"]),
        resumption_handle=handle,
    )
    return EphemeralTokenResponse(**token_data)


@router.post("/session/{session_id}/tool-call", response_model=ToolCallResponse)
async def tool_call(session_id: str, req: ToolCallRequest):
    """Handle a tool call forwarded from Gemini Live via the phone.

    The phone intercepts Gemini's function calls, injects the session_id,
    and POSTs here. The result goes back to Gemini Live.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    result = handle_tool_call(session_id, req.tool_name, req.arguments)
    return ToolCallResponse(result=result)


@router.get("/session/{session_id}/state", response_model=SessionState)
async def session_state(session_id: str):
    """Get current session state (step progress, history)."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    return SessionState(
        session_id=session["session_id"],
        procedure_id=session["procedure_id"],
        current_step=session["current_step"],
        total_steps=session["total_steps"],
        status=session["status"],
        started_at=session["started_at"],
        step_history=[StepCompletion(**sh) for sh in session["step_history"]],
    )


@router.get("/session/{session_id}/context-summary")
async def context_summary(session_id: str):
    """Get a compact text summary of session state.

    The phone injects this as a user message after context window
    compression or session reconnection to re-orient the model.
    """
    summary = get_context_summary(session_id)
    if summary is None:
        raise HTTPException(404, "Session not found")
    return {"summary": summary}


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End and remove a learning session."""
    if not delete_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"ended": True}
