"""Ask-AI mode request/response models. Parallel to models/troubleshoot.py."""

from pydantic import BaseModel

from models.session import (
    EphemeralTokenResponse,
    GeminiConfig,
    GeminiToolDefinition,
)


class AskAISessionStartRequest(BaseModel):
    voice: str = "Aoede"


class AskAISessionStartResponse(BaseModel):
    session_id: str
    system_prompt: str
    tools: list[GeminiToolDefinition]
    gemini_config: GeminiConfig
    ephemeral_token: EphemeralTokenResponse


class AskAIHandoffRequest(BaseModel):
    procedure_id: str
    auto_advance: bool = True
    active_monitoring: bool = True
    voice: str = "Puck"


class AskAICandidate(BaseModel):
    procedure_id: str
    title: str
    match_reason: str
    confidence: str


class AskAISessionState(BaseModel):
    session_id: str
    status: str
    started_at: str
    candidate_procedures: list[AskAICandidate] = []
    handoff_target: str | None = None
    handoff_learner_session_id: str | None = None
