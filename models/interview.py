"""Pydantic models for the Expert Interviewer mode.

The interviewer is a Gemini Live session that runs *alongside* expert
recording: a server-side Flash planner watches a ~1 fps frame + a rolling
audio chunk and decides whether there is a concrete gap worth asking about;
the phone decides when it is safe to speak; Live voices the question and
listens for the answer. Everything learned lands in an `InterviewLog` that
the PHONE owns and uploads next to the video (see routers/expert.py), so a
server restart between Stop and Upload loses nothing.

Three model families live here:

- Wire models for the session endpoints (start / observe / answer / finalize).
- The planner's structured output (`PlannerDecision`) and the deterministic
  policy's result (`Ask`).
- The persisted `InterviewLog` (phone-authoritative; validated on upload and
  re-read by extraction + re-extraction).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from models.session import EphemeralTokenResponse, GeminiConfig, GeminiToolDefinition


# ---- Shared enums ------------------------------------------------------------

# What a learner is most likely to get wrong without the fact. Also the live-ask
# precedence order (see services/expert_interview/policy.py).
Priority = Literal["safety", "spec", "part", "judgment", "tip"]

GapStatus = Literal["open", "asked", "answered", "deferred", "dropped", "unanswered"]

Phase = Literal["preroll", "recording", "debrief", "finalized"]

FactKind = Literal["spec", "tool", "part", "why", "judgment", "warning", "tip", "alternative"]


# ---- Session start -------------------------------------------------------------


class InterviewSessionStartRequest(BaseModel):
    voice: str = "Kore"
    # "iphone" | "glasses" — informational; lets the prompt mention the
    # right device ("the glasses" vs "your phone") and lets the planner budget
    # tighter on glasses where the question card is bandwidth-bound.
    transport: str = "iphone"
    # The expert's pre-record one-liner (optional). Seeds the planner's
    # expected-parts list before any frame arrives.
    task_hint: str = ""


class InterviewSessionStartResponse(BaseModel):
    session_id: str
    system_prompt: str
    tools: list[GeminiToolDefinition]
    gemini_config: GeminiConfig
    ephemeral_token: EphemeralTokenResponse
    # Policy constants mirrored to the phone so its pause gate and the
    # server's budget agree without a second round trip.
    policy: "PolicyConfig"


class PolicyConfig(BaseModel):
    min_seconds_between_live_asks: float
    max_live_asks: int
    default_ttl_seconds: float
    answer_window_seconds: float
    answer_window_hard_cap_seconds: float
    silence_before_ask_seconds: float


# ---- Planner (Flash structured output) --------------------------------------------


class Gap(BaseModel):
    """One thing the learner would need that the recording alone won't give them."""
    # Stable, normalized key the policy dedups on: lowercase, snake_case,
    # names the object + attribute ("lug_nut_socket_size", "torque_feel").
    topic_key: str
    priority: Priority
    # The question the interviewer would ask, phrased for speech (one short
    # sentence, no preamble).
    question: str
    # True only if asking later (in the debrief) would lose information: the
    # object is in frame now, the step is about to become irreversible, or
    # it's a safety issue.
    time_sensitive: bool = False


class PlannerDecision(BaseModel):
    """Structured output of one `observe` tick."""
    # What the expert said in this chunk (verbatim-ish). Empty when silent.
    transcript_delta: str = ""
    # topic_keys the narration in this chunk just answered — the planner
    # listens first and asks second.
    filled: list[str] = []
    # Concrete gaps noticed this tick (new ones only; the prompt shows the
    # ledger so repeats are the model's fault, but the policy dedups anyway).
    new_gaps: list[Gap] = []
    # Which gap, if any, the model proposes asking about RIGHT NOW. Must be a
    # topic_key from new_gaps or the open ledger. The policy has the last word.
    propose_ask: str | None = None
    confidence: float = 0.0
    # Did the expert address the assistant directly in this chunk ("hey
    # Retrace", "note that…")? The phone opens an answer window on this.
    addressed_assistant: bool = False
    # Did the expert ask for quiet ("stop asking", "no more questions")?
    requested_quiet: bool = False


class Ask(BaseModel):
    """What the policy hands the phone when a live question is approved."""
    question_id: str
    topic_key: str
    priority: Priority
    question: str
    # Seconds the phone may hold this before it goes stale (pause gate).
    ttl_seconds: float
    # The text the phone injects as a user turn so Live voices the question.
    inject_text: str


# ---- Observe / answer wire shapes ----------------------------------------------------


class ObserveResponse(BaseModel):
    status: Literal["ok", "quiet", "skipped", "planner_unavailable", "finalized"]
    ask: Ask | None = None
    # Mirrors so the phone's UI can show "N open, M answered" without /state.
    open_gaps: int = 0
    answered: int = 0
    live_asks_used: int = 0
    quiet_mode: bool = False
    # Set when the planner flagged a direct address so the phone opens a window.
    open_answer_window: bool = False


class AnswerReport(BaseModel):
    """Phone → server after an answer window closes."""
    question_id: str
    # Live's input transcript during the window (may be empty).
    transcript: str = ""
    outcome: Literal["answered", "unanswered", "skipped", "interrupted", "deferred"]
    # Video time the question started playing / window closed, if recording.
    t_ask: float | None = None
    t_close: float | None = None


class DebriefRequest(BaseModel):
    """Phone → server at Stop. Returns the ordered questions for the debrief."""
    max_questions: int = 5


class DebriefResponse(BaseModel):
    questions: list[Ask]
    # Text to inject so Live switches from "ask on cue" to "run the debrief".
    inject_text: str


# ---- Persisted log (phone-authoritative) -------------------------------------------


class Fact(BaseModel):
    kind: FactKind
    value: str
    # Glossary-style part name when the fact is about a specific part.
    part: str | None = None
    # Seconds into the video; None for pre-roll / debrief facts.
    t: float | None = None
    source: Literal["live", "debrief", "narration", "preroll"] = "live"
    # Short confidence note from the facts extractor (e.g. "expert stated").
    note: str = ""


class QAEntry(BaseModel):
    question_id: str
    topic_key: str
    priority: Priority
    question: str
    answer: str | None = None
    outcome: str  # answered | unanswered | skipped | interrupted | deferred
    phase: Phase
    t_ask: float | None = None
    t_close: float | None = None


class InterviewLog(BaseModel):
    """What the phone uploads next to the video and the server persists at
    data/interviews/{procedure_id}.json. Read by extraction (first run and
    every re-extraction) and by clip extraction (ducking windows).

    Strict on purpose: every field below has a default, so without
    `extra="forbid"` + a required `version` ANY JSON object (a session-start
    response, an empty `{}`) would validate as an empty interview and the
    procedure would be marked `has_interview` for nothing. The phone always
    sends `version`; an unknown key means a client/server schema mismatch and
    should fail loudly at upload, not silently drop data."""
    model_config = ConfigDict(extra="forbid")

    version: int
    session_id: str | None = None
    transport: str = "iphone"
    briefing: str = ""
    # [[t0, t1], ...] in video seconds during which the AI was speaking.
    # Produced on the phone from the playback clock + writer timeline origin.
    ai_speech_windows: list[list[float]] = []
    qa: list[QAEntry] = []
    facts: list[Fact] = []
    # Planner-transcribed narration, in order: {t, text}.
    transcript: list[dict] = []
    # Free-form client diagnostics (dropped ticks, reconnects…). Never read
    # by extraction; useful when a demo recording looks wrong.
    diagnostics: dict = Field(default_factory=dict)


class InterviewSessionState(BaseModel):
    session_id: str
    session_type: str = "expert_interview"
    phase: Phase
    status: str
    started_at: str
    quiet_mode: bool
    live_asks_used: int
    gaps: dict
    qa: list[QAEntry]
    transcript_chars: int


InterviewSessionStartResponse.model_rebuild()
