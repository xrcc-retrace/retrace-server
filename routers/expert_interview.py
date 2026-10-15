"""Expert Interviewer API surface — parallel to routers/troubleshoot.py.

The phone drives three loops against this router:

  POST /session/start                 once, before recording (mints the Live token)
  POST /session/{id}/observe          every ~5 s: frame + audio chunk → maybe an Ask
  POST /session/{id}/answer           after each answer window closes
  POST /session/{id}/debrief          at Stop: ordered leftover questions
  POST /session/{id}/finalize         before upload: server view of the log (+facts)

The server never speaks and never times the question — it returns an `Ask`
and the phone's pause gate decides when (and whether, given the TTL) to
inject it into Live.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.interview import (
    AnswerReport,
    DebriefRequest,
    DebriefResponse,
    InterviewLog,
    InterviewSessionStartRequest,
    InterviewSessionStartResponse,
    InterviewSessionState,
    ObserveResponse,
    QAEntry,
)
from models.session import (
    EphemeralTokenResponse,
    GeminiConfig,
    GeminiToolDefinition,
    ToolCallRequest,
    ToolCallResponse,
)
from services import session as shared_session
from services.ephemeral_token import create_ephemeral_token
from services.expert_interview import facts as facts_svc
from services.expert_interview import planner, policy
from services.expert_interview import session as iv_session
from services.expert_interview.prompt import (
    GEMINI_CONFIG_INTERVIEW,
    INTERVIEW_TOOL_DEFINITIONS,
    build_interviewer_prompt,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/expert/interview", tags=["expert-interview"])

# After this many consecutive planner failures the phone is told to back off
# (status=planner_unavailable); it resets on the next success.
_PLANNER_BACKOFF_AFTER = 3


def _require(session_id: str) -> dict:
    s = iv_session.get(session_id)
    if not s:
        raise HTTPException(404, "Interview session not found")
    return s


# ---- Lifecycle ---------------------------------------------------------------------------------


@router.post("/session/start", response_model=InterviewSessionStartResponse)
async def start_session(req: InterviewSessionStartRequest):
    s = iv_session.create_interview_session(
        voice=req.voice, transport=req.transport, task_hint=req.task_hint,
    )
    system_prompt = build_interviewer_prompt(transport=req.transport, task_hint=req.task_hint)
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=INTERVIEW_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_INTERVIEW["model"],
        voice=req.voice,
    )
    return InterviewSessionStartResponse(
        session_id=s["session_id"],
        system_prompt=system_prompt,
        tools=[GeminiToolDefinition(**t) for t in INTERVIEW_TOOL_DEFINITIONS],
        gemini_config=GeminiConfig(model=GEMINI_CONFIG_INTERVIEW["model"], voice=req.voice),
        ephemeral_token=EphemeralTokenResponse(**token_data),
        policy=policy.DEFAULT_POLICY,
    )


@router.post("/session/{session_id}/token", response_model=EphemeralTokenResponse)
async def refresh_token(session_id: str, handle: str | None = None):
    s = _require(session_id)
    system_prompt = build_interviewer_prompt(
        transport=s.get("transport", "iphone"), task_hint=s.get("briefing", ""),
    )
    token_data = await create_ephemeral_token(
        system_instruction=system_prompt,
        tools=INTERVIEW_TOOL_DEFINITIONS,
        model=GEMINI_CONFIG_INTERVIEW["model"],
        voice=s.get("voice", GEMINI_CONFIG_INTERVIEW["voice"]),
        resumption_handle=handle,
    )
    return EphemeralTokenResponse(**token_data)


@router.post("/session/{session_id}/tool-call", response_model=ToolCallResponse)
async def tool_call(session_id: str, req: ToolCallRequest):
    """Parity with the other modes. v1 declares no tools, so this only ever
    returns the dispatcher's unknown-tool error — kept so a future
    client-intercepted `record_fact` has somewhere to land."""
    _require(session_id)
    result = await asyncio.to_thread(shared_session.handle_tool_call, session_id, req.tool_name, req.arguments)
    return ToolCallResponse(result=result)


@router.get("/session/{session_id}/briefing-inject")
async def briefing_inject(session_id: str):
    """Text the phone injects before recording to trigger the one pre-roll
    briefing question. Separate endpoint so the phone can skip it."""
    s = _require(session_id)
    return {"inject_text": policy.format_briefing_inject(s.get("briefing", ""))}


# ---- Observe ----------------------------------------------------------------------------------


@router.post("/session/{session_id}/observe", response_model=ObserveResponse)
async def observe(
    session_id: str,
    frame: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    sample_rate: int = Form(16_000),
    video_t: float | None = Form(None),
    phase: str = Form("recording"),
    speaking: bool = Form(False),
):
    """One planner tick. Multipart so the frame (JPEG) and audio chunk (WAV or
    raw PCM16 mono) ride along; `video_t` is seconds into the recording
    (null in pre-roll); `phase` lets the phone drive preroll→recording;
    `speaking` is the phone's own VAD so the planner is never asked to
    propose while the expert is mid-sentence (it still transcribes)."""
    s = _require(session_id)
    if s["phase"] == iv_session.PHASE_FINALIZED:
        return ObserveResponse(status="finalized")
    if phase in (iv_session.PHASE_PREROLL, iv_session.PHASE_RECORDING, iv_session.PHASE_DEBRIEF):
        iv_session.set_phase(s, phase)

    def _counts() -> dict:
        return {
            "open_gaps": sum(1 for g in s["gaps"].values() if g["status"] == "open"),
            "answered": sum(1 for g in s["gaps"].values() if g["status"] == "answered"),
            "live_asks_used": s["budget"]["live_asks"],
            "quiet_mode": s["quiet_mode"],
        }

    # Single in-flight tick, like the oracle's re-entry guard: a slow call
    # must not pile up behind the next 5 s tick.
    if s["_observe_inflight"]:
        return ObserveResponse(status="skipped", **_counts())

    frame_bytes = await frame.read() if frame is not None else None
    audio_bytes = await audio.read() if audio is not None else None
    if not frame_bytes:
        frame_bytes = None
    if not audio_bytes:
        audio_bytes = None

    if planner.should_skip_idle(s, frame_jpg=frame_bytes, audio=audio_bytes):
        return ObserveResponse(status="skipped", **_counts())

    s["_observe_inflight"] = True
    try:
        decision = await planner.observe(
            s, frame_jpg=frame_bytes, audio=audio_bytes, sample_rate=sample_rate,
        )
    finally:
        s["_observe_inflight"] = False

    if decision is None:
        s["_planner_failures"] += 1
        status = "planner_unavailable" if s["_planner_failures"] >= _PLANNER_BACKOFF_AFTER else "skipped"
        return ObserveResponse(status=status, **_counts())
    s["_planner_failures"] = 0

    iv_session.append_transcript(s, decision.transcript_delta, video_t)
    if decision.requested_quiet and not s["quiet_mode"]:
        s["quiet_mode"] = True
        iv_session.log_event(s, "quiet_mode_on", video_t=video_t)
    added = policy.merge_decision(s["gaps"], decision, video_t=video_t)
    if added:
        iv_session.log_event(s, "gaps_added", keys=added, video_t=video_t)

    ask = None
    if not speaking:
        ask = policy.decide_live_ask(
            s["gaps"], decision, s["budget"],
            now=time.monotonic(), phase=s["phase"],
            quiet_mode=s["quiet_mode"], pending_ask=s["pending_ask"] is not None,
        )
    if ask is not None:
        iv_session.record_ask(s, ask, phase=s["phase"], t_ask=video_t)
        iv_session.log_event(s, "ask_approved", topic_key=ask.topic_key, video_t=video_t)
        logger.info("interview %s: ASK %s (%s) %r", session_id, ask.topic_key, ask.priority, ask.question)

    return ObserveResponse(
        status="quiet" if s["quiet_mode"] else "ok",
        ask=ask,
        open_answer_window=decision.addressed_assistant and ask is None,
        **_counts(),
    )


# ---- Answer / debrief / finalize ------------------------------------------------------------------


@router.post("/session/{session_id}/answer")
async def answer(session_id: str, report: AnswerReport):
    s = _require(session_id)
    ok = iv_session.record_answer(
        s,
        question_id=report.question_id,
        transcript=report.transcript,
        outcome=report.outcome,
        t_ask=report.t_ask,
        t_close=report.t_close,
    )
    if not ok:
        raise HTTPException(404, "Unknown question_id")
    # The planner may have missed a spoken "stop asking" inside the window
    # (the mic is on Live then, not on the planner), so honor it here too.
    lowered = report.transcript.lower()
    if any(p in lowered for p in ("stop asking", "no more questions", "hold your questions")):
        s["quiet_mode"] = True
        iv_session.log_event(s, "quiet_mode_on", via="answer")
    return {"status": "ok", "outcome": report.outcome, "quiet_mode": s["quiet_mode"]}


@router.post("/session/{session_id}/debrief", response_model=DebriefResponse)
async def debrief(session_id: str, req: DebriefRequest):
    """Called at Stop. Orders the leftover gaps and returns the inject text
    that flips Live into debrief mode. Idempotent: a second call returns the
    questions already handed out (the phone may retry after a reconnect)."""
    s = _require(session_id)
    if s["phase"] == iv_session.PHASE_DEBRIEF and s.get("_debrief_asks"):
        asks = s["_debrief_asks"]
    else:
        iv_session.set_phase(s, iv_session.PHASE_DEBRIEF)
        s["pending_ask"] = None
        asks = policy.pick_debrief_questions(s["gaps"], req.max_questions)
        s["_debrief_asks"] = asks
        for a in asks:
            iv_session.record_ask(s, a, phase=iv_session.PHASE_DEBRIEF, t_ask=None)
        s["pending_ask"] = None  # debrief answers come as a batch, not one pending
    return DebriefResponse(questions=asks, inject_text=policy.format_debrief_inject(asks))


@router.post("/session/{session_id}/finalize", response_model=InterviewLog)
async def finalize(session_id: str):
    """Server view of the interview with extracted facts. The phone merges
    this into its own log (which also carries the AI speech windows) and
    uploads the result with the video."""
    s = _require(session_id)
    iv_session.set_phase(s, iv_session.PHASE_FINALIZED)
    s["status"] = "completed"
    log = iv_session.to_log(s)
    facts = await facts_svc.extract_facts(log)
    log.facts = facts
    s["_final_log"] = log.model_dump()
    return log


@router.get("/session/{session_id}/state", response_model=InterviewSessionState)
async def state(session_id: str):
    s = _require(session_id)
    return InterviewSessionState(
        session_id=s["session_id"],
        phase=s["phase"],
        status=s["status"],
        started_at=s["started_at"],
        quiet_mode=s["quiet_mode"],
        live_asks_used=s["budget"]["live_asks"],
        gaps=s["gaps"],
        qa=[QAEntry(**q) for q in s["qa"]],
        transcript_chars=sum(len(e.get("text", "")) for e in s["transcript"]),
    )


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    _require(session_id)
    shared_session.delete_session(session_id)
    return {"status": "deleted"}


@router.get("/sessions")
async def list_sessions():
    return {"sessions": iv_session.list_active_sessions()}
