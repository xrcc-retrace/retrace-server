"""Ask-AI session lifecycle — a thin layer over the shared session store.

Mirrors services.troubleshoot.session but far smaller: an Ask-AI session is
procedure-less, holds the latest search candidates, and can promote a chosen
procedure into a normal learner session. Low-level storage lives in
services.session; this module only adds the Ask-AI domain fields + operations.
"""

import logging
from datetime import datetime, timezone

from services import session as shared_session

logger = logging.getLogger(__name__)

SESSION_TYPE = "ask_ai"


def create_ask_ai_session(voice: str = "Aoede") -> dict:
    """Create a new procedure-less Ask-AI session."""
    s = shared_session.create_session(
        procedure_id=None,
        session_type=SESSION_TYPE,
        voice=voice,
    )
    # Augment with Ask-AI-specific fields (mutations land in the shared store
    # via dict reference).
    s["candidate_procedures"] = []          # last search_procedures result
    s["handoff_target"] = None              # chosen procedure_id once picked
    s["handoff_learner_session_id"] = None  # once /handoff spins up a learner
    return s


def set_candidates(session_id: str, candidates: list[dict]) -> None:
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != SESSION_TYPE:
        return
    s["candidate_procedures"] = candidates


def promote_to_learner(
    session_id: str,
    procedure_id: str,
    auto_advance: bool = True,
    active_monitoring: bool = True,
    voice: str = "Puck",
) -> dict | None:
    """Spin up a Learner session on the chosen procedure and return both
    session dicts. Mirrors troubleshoot.promote_to_learner; idempotency is
    handled at the router layer.
    """
    ask = shared_session.get_session(session_id)
    if not ask or ask.get("session_type") != SESSION_TYPE:
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

    ask["status"] = "resolved"
    ask["handoff_target"] = procedure_id
    ask["handoff_learner_session_id"] = learner["session_id"]
    logger.info(
        "Ask-AI session %s promoted to learner session %s (procedure %s)",
        session_id, learner["session_id"], procedure_id,
    )
    return {"ask_ai": ask, "learner": learner}
