"""Interview policy — the deterministic half of the planner.

The Flash planner (planner.py) *proposes*: it transcribes the chunk, marks
gaps the narration just filled, notices new gaps, and may propose one to ask
right now. This module *decides*. It is pure Python over plain dicts so it can
be unit-tested exhaustively without Gemini (tests/test_interview_policy.py),
and so every reliability-critical rule — budget, dedup, priority, quiet mode —
lives in code rather than in a prompt the model may or may not honor.

Timing inputs are passed in (monotonic seconds) rather than read from the
clock, for the same reason.

Gap record shape (JSON-serializable; lives in the session dict):

    {
      "topic_key": str, "priority": Priority, "question": str,
      "time_sensitive": bool, "status": GapStatus,
      "first_seen_t": float|None, "asked_count": int,
      "answer": str|None, "question_id": str|None,
    }
"""

from __future__ import annotations

import re
import uuid
from typing import Iterable

from models.interview import Ask, Gap, PlannerDecision, PolicyConfig, Priority

# Mirrored to the phone at session start so the pause gate and the server's
# budget agree. Tune here, not in two places.
DEFAULT_POLICY = PolicyConfig(
    min_seconds_between_live_asks=60.0,
    max_live_asks=6,
    default_ttl_seconds=20.0,
    answer_window_seconds=12.0,
    answer_window_hard_cap_seconds=30.0,
    silence_before_ask_seconds=1.5,
)

# Below this the planner's propose_ask is ignored (same shape as the oracle's
# _ORACLE_CONFIDENCE_FLOOR). Gaps are still recorded for the debrief.
CONFIDENCE_FLOOR = 0.7

PRIORITY_RANK: dict[str, int] = {
    "safety": 0,
    "spec": 1,
    "part": 2,
    "judgment": 3,
    "tip": 4,
}

# Token-set Jaccard above which two questions are "the same question". Flash
# will rephrase a gap it already raised; topic_key dedup catches most of it,
# this catches the rest.
_DUP_JACCARD = 0.6
_STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "which", "how", "do", "does", "you",
    "your", "that", "this", "it", "of", "to", "for", "on", "in", "and", "or",
    "there", "here", "any", "some", "be", "use", "using", "need", "should",
}


# ---- Normalization + dedup -------------------------------------------------------


def normalize_topic_key(key: str) -> str:
    """lowercase snake_case, ascii, no leading/trailing underscores."""
    key = (key or "").strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")[:64]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def is_duplicate_question(question: str, existing: Iterable[str]) -> bool:
    q = _tokens(question)
    if not q:
        return False
    for other in existing:
        o = _tokens(other)
        if not o:
            continue
        jaccard = len(q & o) / len(q | o)
        if jaccard >= _DUP_JACCARD:
            return True
    return False


# ---- Ledger mutation -----------------------------------------------------------------


def merge_decision(
    gaps: dict[str, dict],
    decision: PlannerDecision,
    *,
    video_t: float | None,
) -> list[str]:
    """Fold one planner decision into the gap ledger. Returns new topic_keys.

    Order matters: fills first (the narration may answer a gap the same tick
    the model re-notices it), then new gaps, deduped by key AND by question
    text against everything already in the ledger.
    """
    for raw in decision.filled:
        key = normalize_topic_key(raw)
        gap = gaps.get(key)
        if gap and gap["status"] in ("open", "asked", "deferred", "unanswered"):
            gap["status"] = "answered"
            if gap.get("answer") is None:
                gap["answer"] = "(answered in narration)"

    added: list[str] = []
    existing_questions = [g["question"] for g in gaps.values()]
    for g in decision.new_gaps:
        key = normalize_topic_key(g.topic_key)
        if not key or key in gaps:
            continue
        if is_duplicate_question(g.question, existing_questions):
            continue
        gaps[key] = {
            "topic_key": key,
            "priority": g.priority,
            "question": g.question.strip(),
            # Safety is always time-sensitive by policy, whatever the model said.
            "time_sensitive": bool(g.time_sensitive) or g.priority == "safety",
            "status": "open",
            "first_seen_t": video_t,
            "asked_count": 0,
            "answer": None,
            "question_id": None,
        }
        existing_questions.append(g.question)
        added.append(key)
    return added


def apply_answer(
    gaps: dict[str, dict],
    *,
    topic_key: str,
    outcome: str,
    transcript: str,
) -> None:
    """Apply the phone's answer report to the ledger.

    answered → closed. skipped/deferred/unanswered/interrupted → stays
    eligible for the debrief (never re-asked live; see decide_live_ask).
    """
    gap = gaps.get(topic_key)
    if not gap:
        return
    if outcome == "answered":
        gap["status"] = "answered"
        gap["answer"] = transcript.strip() or "(answered)"
    elif outcome == "skipped":
        gap["status"] = "dropped"
    elif outcome == "deferred":
        gap["status"] = "deferred"
    else:  # unanswered | interrupted
        gap["status"] = "unanswered"


# ---- Decisions --------------------------------------------------------------------------


def decide_live_ask(
    gaps: dict[str, dict],
    decision: PlannerDecision,
    budget: dict,
    *,
    now: float,
    phase: str,
    quiet_mode: bool,
    pending_ask: bool,
    config: PolicyConfig = DEFAULT_POLICY,
) -> Ask | None:
    """Approve (or veto) the planner's proposal for a live question.

    Every rule is a hard gate. Returning None never loses the gap — it stays
    in the ledger for the debrief.
    """
    if phase != "recording" or quiet_mode or pending_ask:
        return None
    if decision.confidence < CONFIDENCE_FLOOR:
        return None
    if budget.get("live_asks", 0) >= config.max_live_asks:
        return None
    last_close = budget.get("last_window_closed_at")
    if last_close is not None and (now - last_close) < config.min_seconds_between_live_asks:
        return None

    key = normalize_topic_key(decision.propose_ask or "")
    gap = gaps.get(key)
    if not gap or gap["status"] != "open" or gap["asked_count"] > 0:
        return None
    if not gap["time_sensitive"]:
        # Not worth interrupting for — it will come up in the debrief.
        return None

    question_id = uuid.uuid4().hex[:12]
    gap["status"] = "asked"
    gap["asked_count"] += 1
    gap["question_id"] = question_id
    budget["live_asks"] = budget.get("live_asks", 0) + 1
    budget["last_ask_at"] = now
    return Ask(
        question_id=question_id,
        topic_key=key,
        priority=gap["priority"],
        question=gap["question"],
        ttl_seconds=config.default_ttl_seconds,
        inject_text=format_ask_inject(gap["question"]),
    )


def pick_debrief_questions(gaps: dict[str, dict], max_questions: int) -> list[Ask]:
    """Order the leftovers for the post-Stop debrief: highest priority first,
    then earliest seen. Open/deferred/unanswered gaps only; one debrief ask
    per topic (asked_count < 2)."""
    eligible = [
        g for g in gaps.values()
        if g["status"] in ("open", "deferred", "unanswered") and g["asked_count"] < 2
    ]
    eligible.sort(key=lambda g: (PRIORITY_RANK.get(g["priority"], 9),
                                 g["first_seen_t"] if g["first_seen_t"] is not None else 1e9))
    asks: list[Ask] = []
    for g in eligible[:max_questions]:
        qid = uuid.uuid4().hex[:12]
        g["status"] = "asked"
        g["asked_count"] += 1
        g["question_id"] = qid
        asks.append(Ask(
            question_id=qid,
            topic_key=g["topic_key"],
            priority=g["priority"],
            question=g["question"],
            ttl_seconds=0.0,
            inject_text=format_ask_inject(g["question"]),
        ))
    return asks


# ---- Inject text ----------------------------------------------------------------------------

# Mirrors the coaching [ADVANCE]/[CORRECTION] convention: a bracketed event
# tag the Live prompt is taught to act on. Keep the instruction inside the
# text — the model sees only this turn, not our code.


def format_ask_inject(question: str) -> str:
    return (
        "[ASK] Ask the expert this now, in one natural spoken sentence, no preamble: "
        f"\"{question.strip()}\""
    )


def format_briefing_inject(task_hint: str) -> str:
    hint = f" They said they're about to: {task_hint.strip()}." if task_hint.strip() else ""
    return (
        "[BRIEFING] Recording hasn't started." + hint +
        " Ask ONE short question: what they're about to show and which tools or parts "
        "they'll need. Then say you'll stay quiet while they record."
    )


def format_debrief_inject(asks: list[Ask]) -> str:
    if not asks:
        return (
            "[DEBRIEF] Recording has stopped. You have no open questions. Say one short "
            "sentence thanking them and that the recording is ready to upload."
        )
    numbered = "\n".join(f"{i + 1}. {a.question}" for i, a in enumerate(asks))
    return (
        "[DEBRIEF] Recording has stopped. Ask these questions one at a time, in order, "
        "waiting for each answer. Keep each to one sentence. If they say 'done' or 'skip', "
        "move on. After the last one say the recording is ready.\n" + numbered
    )


def priority_rank(priority: Priority | str) -> int:
    return PRIORITY_RANK.get(priority, 9)
