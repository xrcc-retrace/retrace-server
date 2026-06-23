import logging
import uuid
from datetime import datetime, timezone

from services.database import get_procedure

logger = logging.getLogger(__name__)

# In-memory session store — fine for hackathon scope
_sessions: dict[str, dict] = {}


def create_session(
    procedure_id: str | None = None,
    auto_advance: bool = True,
    active_monitoring: bool = True,
    voice: str = "Puck",
    starting_step: int | None = None,
    session_type: str = "learner",
) -> dict | None:
    """Create a new session in the in-memory store.

    `session_type="learner"` (default) binds the session to a procedure, seeds
    `current_step`, and populates `procedure`. `session_type="troubleshoot"`
    creates a procedure-less diagnostic session — the troubleshoot package
    augments it with phase/product/log fields.

    `starting_step` (1-indexed, learner only) lets the caller resume at an
    arbitrary step — the returned session's `current_step` is seeded there,
    and downstream the system prompt is anchored on it so Gemini Live opens
    at step N from the first frame instead of step 1.
    """
    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    if procedure_id is not None:
        proc = get_procedure(procedure_id)
        if not proc:
            return None

        total_steps = len(proc["steps"])
        start_at = starting_step if starting_step is not None else 1
        if start_at < 1 or start_at > total_steps:
            raise ValueError(
                f"starting_step {start_at} out of range (1..{total_steps})"
            )

        session = {
            "session_id": session_id,
            "session_type": session_type,
            "procedure_id": procedure_id,
            "current_step": start_at,
            "total_steps": total_steps,
            "status": "in_progress",
            # Millisecond precision — stays parseable by iOS
            # ISO8601DateFormatter.withFractionalSeconds (which caps at 3 digits).
            # Matches the format used by ephemeral_token.create_ephemeral_token.
            "started_at": started_at,
            "step_history": [],
            "procedure": proc,
            "auto_advance": auto_advance,
            "active_monitoring": active_monitoring,
            "voice": voice,
        }
        logger.info(
            "Created %s session %s for procedure %s (%d steps)",
            session_type, session_id, procedure_id, session["total_steps"],
        )
    else:
        # Procedure-less session (e.g. troubleshoot). Mode-specific fields
        # are added by the mode's own session helper after this returns.
        session = {
            "session_id": session_id,
            "session_type": session_type,
            "procedure_id": None,
            "status": "in_progress",
            "started_at": started_at,
            "voice": voice,
        }
        logger.info(
            "Created %s session %s (no procedure)",
            session_type, session_id,
        )

    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> dict | None:
    return _sessions.get(session_id)


def get_current_step_detail(session_id: str) -> dict | None:
    """Get full details of the current step."""
    session = _sessions.get(session_id)
    if not session:
        return None
    steps = session["procedure"]["steps"]
    for step in steps:
        if step["step_number"] == session["current_step"]:
            return step
    return None


def advance_step(session_id: str, method: str = "verbal") -> dict:
    """Mark current step complete and move to next. Returns result dict.

    `method` is recorded in step_history and should reflect how the advance
    was triggered: "visual" for auto-advance (Gemini's camera-based detection)
    or "verbal" for manual mode (learner said "done").
    """
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}

    if session["status"] == "completed":
        return {"status": "already_completed", "message": "All steps are already done."}

    # Record completion of current step
    session["step_history"].append({
        "step_number": session["current_step"],
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "method": method,
    })

    logger.info("Session %s: step %d completed", session_id, session["current_step"])

    # Check if that was the last step
    if session["current_step"] >= session["total_steps"]:
        session["status"] = "completed"
        return {
            "status": "completed",
            "message": f"All {session['total_steps']} steps completed! Great job.",
        }

    # Advance to next step
    session["current_step"] += 1
    next_step = get_current_step_detail(session_id)
    return {
        "status": "advanced",
        "new_step": session["current_step"],
        "step_title": next_step["title"],
        "step_description": next_step["description"],
        "tips": next_step.get("tips", []),
        "warnings": next_step.get("warnings", []),
        "completion_criteria": next_step.get("completion_criteria", ""),
    }


def get_reference_clip(session_id: str) -> dict:
    """Get the clip URL for the current step."""
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}

    step = get_current_step_detail(session_id)
    if not step:
        return {"error": "No current step"}

    return {
        "clip_url": step.get("clip_url"),
        "step_number": step["step_number"],
        "step_title": step["title"],
    }


def go_to_step(session_id: str, step_number: int) -> dict:
    """Jump to a specific step."""
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}

    if step_number < 1 or step_number > session["total_steps"]:
        return {
            "error": f"Invalid step number. Must be between 1 and {session['total_steps']}.",
        }

    session["current_step"] = step_number
    session["status"] = "in_progress"

    step = get_current_step_detail(session_id)
    logger.info("Session %s: jumped to step %d", session_id, step_number)
    return {
        "status": "jumped",
        "current_step": step_number,
        "step_title": step["title"],
        "step_description": step["description"],
        "tips": step.get("tips", []),
        "warnings": step.get("warnings", []),
        "completion_criteria": step.get("completion_criteria", ""),
    }


def get_context_summary(session_id: str) -> str | None:
    """Build a compact text summary of session state.

    Used to re-orient the model after context window compression
    discards old conversation turns, or after session reconnection.
    """
    session = _sessions.get(session_id)
    if not session:
        return None

    proc = session["procedure"]
    current = session["current_step"]
    total = session["total_steps"]
    # step_history is chronological and may repeat step numbers if the learner
    # used go_to_step to jump back and re-complete a step. For the model's
    # re-orientation we want a numerically-sorted unique set — "completed
    # steps 1, 2, 3" is useful context; "completed steps 3, 1, 2, 1" is not.
    completed_nums = sorted({sh["step_number"] for sh in session["step_history"]})

    step = get_current_step_detail(session_id)
    if not step:
        return None

    # Build completed steps summary
    if completed_nums:
        completed_str = f"The learner has completed steps {', '.join(str(n) for n in completed_nums)}."
    else:
        completed_str = "The learner has not completed any steps yet."

    # Build remaining steps count — do NOT leak future step titles, per the
    # system prompt's "you only know the current step" invariant. Counts are
    # fine so the model knows how close it is to the end; titles would let it
    # spoil the upcoming work verbally.
    remaining_count = sum(1 for s in proc["steps"] if s["step_number"] > current)
    if remaining_count == 0:
        remaining_str = "This is the final step."
    elif remaining_count == 1:
        remaining_str = (
            "One more step remains after this one. Do not describe or hint at it "
            "before calling advance_step."
        )
    else:
        remaining_str = (
            f"{remaining_count} more steps remain after this one. Do not describe "
            "or hint at them before calling advance_step."
        )

    # Current step details
    tips_str = ""
    if step.get("tips"):
        tips_str = " Tips: " + "; ".join(step["tips"]) + "."
    warnings_str = ""
    if step.get("warnings"):
        warnings_str = " Warnings: " + "; ".join(step["warnings"]) + "."

    return (
        f'You are coaching "{proc["title"]}". '
        f"{completed_str} "
        f"They are currently on Step {current} of {total}: {step['title']}. "
        f"{step['description']}{tips_str}{warnings_str} "
        f"{remaining_str}"
    )


def delete_session(session_id: str) -> bool:
    """End and remove a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info("Deleted session %s", session_id)
        return True
    return False


def _dispatch_advance_step(session_id: str, args: dict) -> dict:
    """Dispatcher wrapper — enforce the auto-advance / active-monitoring toggles.

    Gemini supplies `trigger` ("verbal" if the learner said they're done,
    "visual" if it detected completion from the camera). A visual advance is
    only honored when the session is BOTH actively monitoring AND auto-advancing
    — otherwise the model must wait for the learner to say so. We refuse the
    advance rather than silently downgrading it, so a model that ignores the
    prompt still can't move the step. The completion `method` is tagged from the
    actual trigger (not inferred from the mode), fixing prior mis-tagging.
    """
    session = _sessions.get(session_id)
    trigger = (args or {}).get("trigger") or "verbal"
    visual_ok = bool(
        session and session.get("active_monitoring") and session.get("auto_advance")
    )
    if trigger == "visual" and not visual_ok:
        logger.info(
            "Session %s: refused visual advance_step (active=%s, auto_advance=%s)",
            session_id,
            session.get("active_monitoring") if session else None,
            session.get("auto_advance") if session else None,
        )
        return {
            "status": "ignored",
            "message": "Auto-advance is off — only advance when the learner says they're done.",
        }
    return advance_step(session_id, method=trigger)


# Tool-call dispatcher, keyed by (session_type, tool_name). Per-mode packages
# register their own handlers via register_tool_handler() at import time;
# the learner handlers below are the bootstrapped defaults.
_TOOL_HANDLERS: dict[tuple[str, str], callable] = {
    ("learner", "advance_step"): _dispatch_advance_step,
    ("learner", "get_reference_clip"): lambda sid, args: get_reference_clip(sid),
    ("learner", "go_to_step"): lambda sid, args: go_to_step(sid, args.get("step_number", 0)),
}


def register_tool_handler(session_type: str, tool_name: str, handler) -> None:
    """Register a tool-call handler for a given session type.

    Per-mode packages (e.g. services.troubleshoot.handlers) call this on
    import so the shared dispatcher can route Gemini Live tool calls without
    knowing about each mode's specifics.
    """
    _TOOL_HANDLERS[(session_type, tool_name)] = handler
    logger.info("Registered tool handler %s/%s", session_type, tool_name)


def handle_tool_call(session_id: str, tool_name: str, arguments: dict) -> dict:
    """Dispatch a Gemini Live tool call to the right handler.

    Looked up by (session_type, tool_name) so a given tool name can have
    different implementations per mode if needed.
    """
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    session_type = session.get("session_type", "learner")
    handler = _TOOL_HANDLERS.get((session_type, tool_name))
    if not handler:
        return {"error": f"Unknown tool for {session_type} session: {tool_name}"}
    return handler(session_id, arguments)
