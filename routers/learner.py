import asyncio
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import CLIPS_DIR
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
from services.clip_extractor import extract_done_frame
from services.ephemeral_token import create_ephemeral_token
from services.prompt_builder import build_system_prompt, GEMINI_CONFIG, TOOL_DEFINITIONS
from services.session import (
    advance_step,
    create_session,
    delete_session,
    get_context_summary,
    get_current_step_detail,
    get_session,
    handle_tool_call,
)
from services.step_oracle import verify_step

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learner", tags=["learner"])

# Verification-oracle tuning. A completion is only accepted when the oracle
# reports `complete=true` with confidence at/above this floor, sustained for
# the number of consecutive polls the step's verifiability level requires
# (dwell — filters single-frame false positives; see design principle #2).
_ORACLE_CONFIDENCE_FLOOR = 0.6
_DWELL_REQUIRED = {"reliable": 1, "marginal": 2}

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
            active_monitoring=req.active_monitoring,
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
        active_monitoring=session["active_monitoring"],
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
        active_monitoring=session["active_monitoring"],
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


@router.post("/session/{session_id}/verify")
async def verify_current_step(
    session_id: str,
    frame: UploadFile = File(...),
    mode: str = Form("poll"),
    claim: str | None = Form(None),
):
    """Out-of-band visual verification of the current step.

    Two modes share the endpoint (and the frame upload):

    `mode="poll"` (default) — the unprompted ~4s Active-monitoring poll.
    The server runs the Flash oracle against the step's spatial end-state
    (grounded on the expert's 'done' frame) and its error states, applies the
    dwell + toggle gates, and — on a confirmed completion — advances the
    session server-side and returns the same shape a tool-call `advance_step`
    would. Gemini Live is never involved; the phone injects the outcome into
    the Live socket as an `[ADVANCE]` / `[CORRECTION]` turn.

    `mode="on_demand"` — the learner-initiated `check_step` tool. The phone
    intercepts the Live tool call, attaches its latest frame, and sets
    `claim="done"` (learner says they finished) or `claim="question"`
    (learner asks if it's right). Learner-initiated, so it bypasses the
    active_monitoring/auto_advance toggles, dwell, and the error dedup latch —
    see `_verify_on_demand`.

    Poll response `status`:
      - `skip`      — step is verbal-only (visual_verifiability=none) or the
                      session isn't actively auto-advancing. Phone does nothing.
      - `pending`   — polled, nothing confirmed yet (or oracle unavailable).
      - `error`     — a known error is visible; `error_text` is the correction.
      - `advanced` / `completed` — the step advanced; carries next-step fields.

    On-demand adds: `not_confirmed` (+`reason`, optional `error_text`),
    `unverifiable` (+`reason`), `check_failed` (+`reason`),
    `already_completed`; `advanced`/`completed` carry `method`
    ("visual"|"verbal") and an optional `note`.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if mode == "on_demand":
        if claim not in ("done", "question"):
            raise HTTPException(400, "claim must be 'done' or 'question' for on_demand mode")
        return await _verify_on_demand(session_id, session, frame, claim)

    if session.get("status") == "completed" or not session.get("procedure"):
        return {"status": "skip"}

    # Master switch: with active monitoring off the coach is fully reactive —
    # the phone shouldn't be polling at all, but skip defensively.
    if not session.get("active_monitoring"):
        return {"status": "skip"}

    step = get_current_step_detail(session_id)
    if not step:
        return {"status": "skip"}

    verifiability = step.get("visual_verifiability", "reliable")
    errors = step.get("error_criteria") or []
    # Advance is allowed only on a visually-verifiable step in an auto-advancing
    # session (mirrors _dispatch_advance_step). Error interjection stays alive
    # regardless, so a verbal-only step can still be corrected mid-work.
    advance_allowed = verifiability != "none" and bool(session.get("auto_advance"))
    if not advance_allowed and not errors:
        return {"status": "skip"}

    current_jpg = await frame.read()
    if not current_jpg:
        return {"status": "pending"}

    # Image-to-image grounding when a real clip exists (video procedures).
    # Manual/web steps have no video 'done' frame → text-only comparison.
    reference_jpg: bytes | None = None
    clip_path = CLIPS_DIR / session["procedure_id"] / f"step_{step['step_number']}.mp4"
    if clip_path.exists():
        done_path = await asyncio.to_thread(extract_done_frame, clip_path)
        if done_path is not None:
            reference_jpg = await asyncio.to_thread(done_path.read_bytes)

    polled_step = step["step_number"]
    verdict = await verify_step(
        current_jpg, step, reference_jpg,
        glossary=session["procedure"].get("parts"),
    )
    if verdict is None:
        return {"status": "pending"}

    # The oracle call is awaited — a verbal advance_step could have landed
    # while it was in flight, moving the session off the step we verified.
    # If so, drop this verdict rather than advancing a step that already moved.
    current = get_current_step_detail(session_id)
    if not current or current["step_number"] != polled_step:
        return {"status": "pending"}

    # Error interjection takes precedence — an error means the step isn't done.
    # Dedup server-side: emit each correction once until a clean poll clears it,
    # so the coach doesn't nag every tick while the mistake persists.
    if verdict.error_index is not None and 0 <= verdict.error_index < len(errors):
        session["_verify_streak"] = 0
        error_text = errors[verdict.error_index]
        if session.get("_last_error") == error_text:
            return {"status": "pending"}
        session["_last_error"] = error_text
        return {"status": "error", "error_text": error_text}

    # Clean frame — clear the last-error latch so a recurrence re-fires.
    session["_last_error"] = None

    if not advance_allowed:
        return {"status": "pending"}

    # Completion path with dwell: require N consecutive confident positives.
    confident = verdict.complete and verdict.confidence >= _ORACLE_CONFIDENCE_FLOOR
    if not confident:
        session["_verify_streak"] = 0
        return {"status": "pending"}

    required = _DWELL_REQUIRED.get(verifiability, 1)
    streak = session.get("_verify_streak", 0) + 1
    session["_verify_streak"] = streak
    if streak < required:
        return {"status": "pending", "confirmations": streak, "required": required}

    # Confirmed — advance server-side (tagged visual) and hand the phone the
    # next-step payload so it can inject an [ADVANCE] turn into Gemini Live.
    result = advance_step(session_id, method="visual")
    logger.info(
        "Session %s: oracle advanced step via visual verification (%s)",
        session_id, verifiability,
    )
    return result


async def _verify_on_demand(
    session_id: str, session: dict, frame: UploadFile, claim: str
) -> dict:
    """Learner-initiated oracle check backing the Live `check_step` tool.

    Differs from the poll path deliberately:
      - Bypasses active_monitoring/auto_advance (the learner asked — works in
        Passive mode too; those toggles gate only the unprompted poll).
      - No dwell and no `_verify_streak` writes — one confident verdict decides.
      - No error dedup skip — an explicit question deserves an explicit answer
        (but a matched error still SETS `_last_error` so the next poll doesn't
        immediately re-nag the mistake Live is about to voice).
      - Trust-on-failure: on claim="done", an unavailable oracle or an
        unverifiable step advances on the learner's word (method="verbal")
        rather than blocking the session.
      - Insist-once: the first oracle rejection of a done-claim is reported;
        a second consecutive done-claim on the same step advances verbally.
    """
    if session.get("status") == "completed" or not session.get("procedure"):
        return {"status": "already_completed"}

    step = get_current_step_detail(session_id)
    if not step:
        return {"status": "already_completed"}

    verifiability = step.get("visual_verifiability", "reliable")
    errors = step.get("error_criteria") or []

    def _advance_verbal(note: str) -> dict:
        result = advance_step(session_id, method="verbal")
        if result.get("status") in ("advanced", "completed"):
            result["method"] = "verbal"
            result["note"] = note
        return result

    # Steps with no single-frame visual end-state (counting/timing/torque):
    # the learner's word is the only signal there is.
    if verifiability == "none":
        if claim == "done":
            logger.info(
                "Session %s: check_step done-claim on none-verifiability step — verbal advance",
                session_id,
            )
            return _advance_verbal("This step has no visible end-state; advanced on the learner's word.")
        return {
            "status": "unverifiable",
            "reason": "This step has no visible end-state to check from the camera.",
        }

    current_jpg = await frame.read()
    reference_jpg: bytes | None = None
    if current_jpg:
        clip_path = CLIPS_DIR / session["procedure_id"] / f"step_{step['step_number']}.mp4"
        if clip_path.exists():
            done_path = await asyncio.to_thread(extract_done_frame, clip_path)
            if done_path is not None:
                reference_jpg = await asyncio.to_thread(done_path.read_bytes)

    checked_step = step["step_number"]
    verdict = None
    if current_jpg:
        verdict = await verify_step(
            current_jpg, step, reference_jpg,
            glossary=session["procedure"].get("parts"),
        )

    if verdict is None:
        # Oracle unavailable (or no usable frame). Never block the learner.
        logger.warning(
            "Session %s: check_step oracle unavailable (claim=%s)", session_id, claim
        )
        if claim == "done":
            return _advance_verbal("Visual check unavailable — advanced on the learner's word.")
        return {
            "status": "check_failed",
            "reason": "Couldn't check the camera right now.",
        }

    # The oracle call is awaited — the step could have moved while in flight
    # (poll advance, HUD swipe, go_to_step). Drop the stale verdict.
    current = get_current_step_detail(session_id)
    if not current or current["step_number"] != checked_step:
        if session.get("status") == "completed":
            return {"status": "already_completed"}
        return {
            "status": "not_confirmed",
            "reason": "The step changed while checking — coach the current step.",
        }

    # A clearly visible known error trumps everything — the step isn't done.
    if verdict.error_index is not None and 0 <= verdict.error_index < len(errors):
        error_text = errors[verdict.error_index]
        session["_last_error"] = error_text
        logger.info(
            "Session %s: check_step matched error %d (claim=%s)",
            session_id, verdict.error_index, claim,
        )
        return {
            "status": "not_confirmed",
            "reason": verdict.reason or error_text,
            "error_text": error_text,
        }

    if verdict.complete and verdict.confidence >= _ORACLE_CONFIDENCE_FLOOR:
        result = advance_step(session_id, method="visual")
        if result.get("status") in ("advanced", "completed"):
            result["method"] = "visual"
        logger.info(
            "Session %s: check_step visually confirmed step %d (claim=%s, conf=%.2f)",
            session_id, checked_step, claim, verdict.confidence,
        )
        return result

    # Not complete (or not confident enough).
    if claim == "done":
        if session.get("_done_claim_step") == checked_step:
            # Insist-once: the oracle already rejected a done-claim on this
            # step. The learner is the final authority — advance on their word.
            logger.info(
                "Session %s: check_step insist-once override on step %d",
                session_id, checked_step,
            )
            return _advance_verbal("The learner insisted after a failed visual check.")
        session["_done_claim_step"] = checked_step

    if verdict.reason:
        reason = f"The step doesn't look finished yet: {verdict.reason}"
    elif step.get("completion_criteria"):
        reason = (
            "The step doesn't look finished yet. What finished looks like: "
            f"{step['completion_criteria']}"
        )
    else:
        reason = "The step doesn't look finished yet from the camera."
    return {"status": "not_confirmed", "reason": reason}


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
