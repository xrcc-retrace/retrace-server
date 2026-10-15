"""Expert-interview session lifecycle — augments a shared session with the
planner's gap ledger, the Q/A log, the narration transcript, and the live-ask
budget.

Low-level storage (in-memory dict) lives in services.session. The server-side
session is a *planner cache*: the phone owns the canonical InterviewLog and
uploads it with the video (routers/expert.py). If this process dies between
Stop and Upload, nothing is lost that extraction needs.
"""

import logging
import time
from datetime import datetime, timezone

from models.interview import Ask, InterviewLog, QAEntry
from services import session as shared_session
from services.expert_interview import policy

logger = logging.getLogger(__name__)

SESSION_TYPE = "expert_interview"

PHASE_PREROLL = "preroll"
PHASE_RECORDING = "recording"
PHASE_DEBRIEF = "debrief"
PHASE_FINALIZED = "finalized"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def create_interview_session(voice: str, transport: str, task_hint: str) -> dict:
    s = shared_session.create_session(
        procedure_id=None,
        session_type=SESSION_TYPE,
        voice=voice,
    )
    s["phase"] = PHASE_PREROLL
    s["transport"] = transport
    s["briefing"] = task_hint.strip()
    s["quiet_mode"] = False
    s["gaps"] = {}                 # topic_key → gap record (see policy.py)
    s["qa"] = []                   # QAEntry dicts, in ask order
    s["transcript"] = []           # [{t, text}] planner-transcribed narration
    s["budget"] = {"live_asks": 0, "last_ask_at": None, "last_window_closed_at": None}
    s["pending_ask"] = None        # question_id delivered, awaiting /answer
    s["_observe_inflight"] = False
    s["_planner_failures"] = 0
    s["_last_frame_digest"] = None
    s["events"] = []               # [{ts, event, ...}] diagnostics
    return s


def get(session_id: str) -> dict | None:
    s = shared_session.get_session(session_id)
    if not s or s.get("session_type") != SESSION_TYPE:
        return None
    return s


def log_event(s: dict, event: str, **fields) -> None:
    s["events"].append({"timestamp": _now_iso(), "event": event, **fields})


def set_phase(s: dict, phase: str) -> None:
    if s["phase"] != phase:
        log_event(s, "phase_transition", from_=s["phase"], to=phase)
        s["phase"] = phase


def record_ask(s: dict, ask: Ask, phase: str, t_ask: float | None) -> None:
    s["pending_ask"] = ask.question_id
    s["qa"].append(QAEntry(
        question_id=ask.question_id,
        topic_key=ask.topic_key,
        priority=ask.priority,
        question=ask.question,
        outcome="asked",
        phase=phase,
        t_ask=t_ask,
    ).model_dump())


def record_answer(
    s: dict,
    *,
    question_id: str,
    transcript: str,
    outcome: str,
    t_ask: float | None,
    t_close: float | None,
) -> bool:
    entry = next((q for q in s["qa"] if q["question_id"] == question_id), None)
    if entry is None:
        return False
    entry["answer"] = transcript.strip() or None
    entry["outcome"] = outcome
    if t_ask is not None:
        entry["t_ask"] = t_ask
    entry["t_close"] = t_close
    policy.apply_answer(s["gaps"], topic_key=entry["topic_key"], outcome=outcome, transcript=transcript)
    if s.get("pending_ask") == question_id:
        s["pending_ask"] = None
    # The min-interval clock starts when the window CLOSES, so a long answer
    # doesn't eat into the quiet period that follows it.
    s["budget"]["last_window_closed_at"] = time.monotonic()
    if transcript.strip():
        s["transcript"].append({"t": t_close, "text": transcript.strip(), "source": "answer"})
    return True


def append_transcript(s: dict, text: str, video_t: float | None) -> None:
    text = (text or "").strip()
    if text:
        s["transcript"].append({"t": video_t, "text": text, "source": "narration"})


def to_log(s: dict, facts: list[dict] | None = None) -> InterviewLog:
    """Server-side view of the interview, for /finalize. The phone merges this
    with its own copy (speech windows live only on the phone)."""
    return InterviewLog(
        version=1,
        session_id=s["session_id"],
        transport=s.get("transport", "iphone"),
        briefing=s.get("briefing", ""),
        qa=[QAEntry(**q) for q in s["qa"]],
        facts=facts or [],
        transcript=list(s["transcript"]),
        diagnostics={
            "live_asks": s["budget"]["live_asks"],
            "quiet_mode": s["quiet_mode"],
            "planner_failures": s["_planner_failures"],
            "gaps": {k: {"status": g["status"], "priority": g["priority"]} for k, g in s["gaps"].items()},
            "events": s["events"][-50:],
        },
    )


def list_active_sessions() -> list[dict]:
    return [
        {
            "session_id": s["session_id"],
            "phase": s.get("phase"),
            "status": s["status"],
            "started_at": s["started_at"],
            "live_asks": s["budget"]["live_asks"],
            "open_gaps": sum(1 for g in s["gaps"].values() if g["status"] == "open"),
        }
        for s in shared_session._sessions.values()
        if s.get("session_type") == SESSION_TYPE
    ]
