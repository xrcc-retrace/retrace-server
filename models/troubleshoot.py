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
