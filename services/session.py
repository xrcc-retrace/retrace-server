import logging
import uuid
from datetime import datetime, timezone

from services.database import get_procedure

logger = logging.getLogger(__name__)

# In-memory session store — fine for hackathon scope
_sessions: dict[str, dict] = {}


def create_session(procedure_id: str, auto_advance: bool = True, voice: str = "Puck") -> dict:
    """Create a new learning session for a procedure."""
    proc = get_procedure(procedure_id)
    if not proc:
        return None

    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "procedure_id": procedure_id,
        "current_step": 1,
        "total_steps": len(proc["steps"]),
        "status": "in_progress",
        # Millisecond precision — stays parseable by iOS
        # ISO8601DateFormatter.withFractionalSeconds (which caps at 3 digits).
        # Matches the format used by ephemeral_token.create_ephemeral_token.
        "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "step_history": [],
        "procedure": proc,
        "auto_advance": auto_advance,
        "voice": voice,
    }
    _sessions[session_id] = session
    logger.info(
        "Created session %s for procedure %s (%d steps)",
        session_id, procedure_id, session["total_steps"],
    )
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

    # Build remaining steps list
    remaining = [s for s in proc["steps"] if s["step_number"] > current]
    if remaining:
        remaining_str = "Steps remaining: " + ", ".join(
            f"{s['step_number']} ({s['title']})" for s in remaining
        ) + "."
    else:
        remaining_str = "This is the final step."

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


def _dispatch_advance_step(session_id: str, _args: dict) -> dict:
    """Dispatcher wrapper — tag the completion method based on session mode."""
    session = _sessions.get(session_id)
    method = "visual" if session and session.get("auto_advance") else "verbal"
    return advance_step(session_id, method=method)


# Tool call dispatcher
_TOOL_HANDLERS = {
    "advance_step": _dispatch_advance_step,
    "get_reference_clip": lambda sid, args: get_reference_clip(sid),
    "go_to_step": lambda sid, args: go_to_step(sid, args.get("step_number", 0)),
}


def handle_tool_call(session_id: str, tool_name: str, arguments: dict) -> dict:
    """Dispatch a Gemini Live tool call to the right handler."""
    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}
    return handler(session_id, arguments)
