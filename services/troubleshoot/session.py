"""Diagnostic session lifecycle — augments a shared session with
troubleshoot-specific state (phase, identified product, candidates, log).

Low-level session storage (in-memory dict) lives in services.session. This
module only adds domain fields and higher-level operations that are unique
to the troubleshoot mode.
"""

import logging
from datetime import datetime, timezone

from services import session as shared_session

logger = logging.getLogger(__name__)


# Phase progression for a diagnostic session. Informational — the tool set
# does not change between phases; the prompt is what steers the model.
PHASE_DISCOVERING = "discovering"
PHASE_DIAGNOSING = "diagnosing"
PHASE_RESOLVING = "resolving"
PHASE_RESOLVED = "resolved"


def create_diagnostic_session(voice: str = "Charon") -> dict:
    """Create a new procedure-less diagnostic session.

    Adds troubleshoot-specific fields on top of the shared session shape so
    the dashboard + handlers can track phase progression, identified product,
    candidate procedures, and the full tool-call log.
    """
    s = shared_session.create_session(
        procedure_id=None,
        session_type="troubleshoot",
        voice=voice,
    )
    # Augment with diagnostic-specific fields. Mutations land in the shared
    # in-memory store via dict reference — no extra write needed.
    s["phase"] = PHASE_DISCOVERING
    s["identified_product"] = None          # {name, category, confidence}
    s["candidate_procedures"] = []           # list from search_procedures
    s["uploaded_manual_id"] = None           # present if user uploaded a PDF
    s["pending_manual_ingest"] = None        # job id while PDF → SOP runs
    s["generated_procedure_id"] = None       # set once manual ingestion finishes
    s["handoff_target"] = None               # procedure_id once Gemini commits
    s["handoff_learner_session_id"] = None   # once /handoff spins up learner
    s["diagnostic_log"] = []                 # [{ts, tool, args, result}]
    s["web_search_progress"] = None          # see init_/update_web_search_progress
    return s


def transition_phase(session_id: str, new_phase: str) -> bool:
    """Move a diagnostic session to a new phase and log it."""
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return False
    prev = s.get("phase")
    s["phase"] = new_phase
    s["diagnostic_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": "phase_transition",
        "from": prev,
        "to": new_phase,
    })
    logger.info("Troubleshoot session %s: %s → %s", session_id, prev, new_phase)
    return True


def log_tool_call(session_id: str, tool_name: str, args: dict, result: dict) -> None:
    """Append a tool-call entry to the diagnostic log."""
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return
    s["diagnostic_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": "tool_call",
        "tool": tool_name,
        "args": args,
        "result": result,
    })


def set_identified_product(session_id: str, product: dict) -> None:
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return
    s["identified_product"] = product


def set_candidate_procedures(session_id: str, candidates: list[dict]) -> None:
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return
    s["candidate_procedures"] = candidates


def init_web_search_progress(session_id: str) -> dict | None:
    """Initialize the per-session web-search progress block.

    Called by `_handle_web_search_for_fix` at the top of the handler so
    the iOS client can immediately see `phase=searching_web` on its next
    poll of /state. Returns the progress dict for direct use by the
    handler's update closure (one less get_session round-trip per
    update).
    """
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return None
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    s["web_search_progress"] = {
        "phase": "searching_web",
        "source_count": None,
        "procedure_id": None,
        "error_message": None,
        "started_at": now,
        "updated_at": now,
    }
    logger.info(
        "Troubleshoot session %s: web_search_progress phase=searching_web",
        session_id,
    )
    return s["web_search_progress"]


def update_web_search_progress(session_id: str, phase: str, **extra) -> None:
    """Update the progress block. Caller passes any of:
        source_count: int
        procedure_id: str
        error_message: str
    Unknown keys are ignored. No-op if the session has no progress
    block (handler always seeds one with init_, but tolerate races).
    """
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return
    progress = s.get("web_search_progress")
    if progress is None:
        # Handler should have called init_web_search_progress first;
        # build a minimal block on the fly so we never drop a signal.
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        progress = {
            "phase": phase,
            "source_count": None,
            "procedure_id": None,
            "error_message": None,
            "started_at": now,
            "updated_at": now,
        }
        s["web_search_progress"] = progress
    progress["phase"] = phase
    for key in ("source_count", "procedure_id", "error_message"):
        if key in extra:
            progress[key] = extra[key]
    progress["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    logger.info(
        "Troubleshoot session %s: web_search_progress phase=%s extra=%s",
        session_id, phase, {k: v for k, v in extra.items() if v is not None},
    )


def set_handoff_target(session_id: str, procedure_id: str, summary: str) -> None:
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != "troubleshoot":
        return
    s["handoff_target"] = procedure_id
    s["handoff_summary"] = summary


def promote_to_learner(
    session_id: str,
    procedure_id: str,
    auto_advance: bool = True,
    active_monitoring: bool = True,
    voice: str = "Puck",
) -> dict | None:
    """Spin up a Learner session on the chosen procedure, mark the
    diagnostic session resolved, and return both session dicts.

    The diagnostic session stays alive for 60s (effectively: until explicit
    DELETE or server restart) so the phone can fall back if the learner
    socket fails to open — keeping this here makes the retry idempotent.
    """
    diag = shared_session.get_session(session_id)
    if not diag or diag.get("session_type") != "troubleshoot":
        return None

    learner = shared_session.create_session(
        procedure_id=procedure_id,
        session_type="learner",
        auto_advance=auto_advance,
        active_monitoring=active_monitoring,
        voice=voice,
    )
    if not learner:
        return None

    diag["status"] = "resolved"
    diag["phase"] = PHASE_RESOLVED
    diag["handoff_learner_session_id"] = learner["session_id"]
    diag["diagnostic_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": "handoff_executed",
        "learner_session_id": learner["session_id"],
        "procedure_id": procedure_id,
    })
    logger.info(
        "Troubleshoot session %s promoted to learner session %s (procedure %s)",
        session_id, learner["session_id"], procedure_id,
    )
    return {"diagnostic": diag, "learner": learner}


def list_active_sessions() -> list[dict]:
    """Enumerate diagnostic sessions for the dashboard."""
    # Reach into the shared store directly — an O(n) walk is fine at hackathon scale.
    return [
        {
            "session_id": s["session_id"],
            "phase": s.get("phase"),
            "status": s["status"],
            "started_at": s["started_at"],
            "identified_product": s.get("identified_product"),
            "candidate_procedures": s.get("candidate_procedures", []),
            "generated_procedure_id": s.get("generated_procedure_id"),
            "handoff_target": s.get("handoff_target"),
            "handoff_learner_session_id": s.get("handoff_learner_session_id"),
            "diagnostic_log": s.get("diagnostic_log", []),
        }
        for s in shared_session._sessions.values()
        if s.get("session_type") == "troubleshoot"
    ]
