"""Pydantic models for the Troubleshoot-mode API surface."""

from pydantic import BaseModel

from models.session import EphemeralTokenResponse, GeminiConfig, GeminiToolDefinition


class TroubleshootSessionStartRequest(BaseModel):
    voice: str = "Charon"


class IdentifiedProduct(BaseModel):
    product_name: str
    category: str | None = None
    confidence: str  # "high" | "medium" | "low"


class CandidateProcedure(BaseModel):
    procedure_id: str
    title: str
    match_reason: str
    confidence: str  # "high" | "medium" | "low"


class DiagnosticLogEntry(BaseModel):
    timestamp: str
    event: str  # "phase_transition" | "tool_call" | "handoff_executed"
    # Free-form fields depending on event type
    tool: str | None = None
    args: dict | None = None
    result: dict | None = None
    # Phase transitions
    from_: str | None = None
    to: str | None = None
    # Handoff
    learner_session_id: str | None = None
    procedure_id: str | None = None


class WebSearchProgress(BaseModel):
    """Progress signal for the `web_search_for_fix` tool call.

    Lives on the troubleshoot session and is exposed via
    `GET /api/troubleshoot/session/{id}/state`. iOS polls the state
    endpoint while a `web_search_for_fix` call is in flight and renders
    UI off the `phase` field.

    Phase progression on the happy path:
        searching_web → synthesizing → complete
    Terminal failure phases:
        no_fix_found  (web returned nothing usable)
        timeout       (overall wall-clock budget exceeded — see
                       services.troubleshoot.handlers.WEB_SEARCH_TIMEOUT_S)
        error         (transport / SDK / save failure)

    `source_count` is set when the phase advances past `searching_web`.
    `procedure_id` is set when the phase reaches `complete`.
    `error_message` is set when the phase reaches `error` or `timeout`.
    """
    phase: str  # "searching_web" | "synthesizing" | "complete" | "no_fix_found" | "timeout" | "error"
    source_count: int | None = None
    procedure_id: str | None = None
    error_message: str | None = None
    started_at: str
    updated_at: str


class TroubleshootSessionState(BaseModel):
    session_id: str
    session_type: str = "troubleshoot"
    phase: str  # "discovering" | "diagnosing" | "resolving" | "resolved"
    status: str  # "in_progress" | "resolved"
    started_at: str
    identified_product: IdentifiedProduct | None = None
    candidate_procedures: list[CandidateProcedure] = []
    generated_procedure_id: str | None = None
    handoff_target: str | None = None
    handoff_learner_session_id: str | None = None
    diagnostic_log: list[dict] = []
    # Set while `web_search_for_fix` is in flight (and stays populated
    # after it completes so a slow poll still sees the final state).
    # `null` when no web search has been kicked off this session.
    web_search_progress: WebSearchProgress | None = None


class TroubleshootSessionStartResponse(BaseModel):
    session_id: str
    system_prompt: str
    tools: list[GeminiToolDefinition]
    gemini_config: GeminiConfig
    ephemeral_token: EphemeralTokenResponse


class HandoffRequest(BaseModel):
    procedure_id: str
    auto_advance: bool = True
    voice: str = "Puck"


class HandoffResponse(BaseModel):
    diagnostic_session_id: str
    learner_session_id: str
    procedure_id: str


class ManualUploadResponse(BaseModel):
    manual_id: str
    status: str  # "uploaded" | "processing" | "ready" | "failed"
    filename: str


class ManualStatusResponse(BaseModel):
    manual_id: str
    status: str
    procedure_id: str | None = None
    error: str | None = None
    filename: str
