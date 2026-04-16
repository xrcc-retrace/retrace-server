from pydantic import BaseModel

from models.procedure import Procedure


class SessionStartRequest(BaseModel):
    procedure_id: str
    auto_advance: bool = True
    voice: str = "Puck"


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


class ToolCallResponse(BaseModel):
    result: dict


class StepCompletion(BaseModel):
    step_number: int
    completed_at: str
    method: str  # "verbal" | "visual"


class SessionState(BaseModel):
    session_id: str
    procedure_id: str
    current_step: int
    total_steps: int
    status: str  # "in_progress" | "completed"
    started_at: str
    step_history: list[StepCompletion]


class GeminiToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class GeminiConfig(BaseModel):
    model: str
    voice: str


class EphemeralTokenResponse(BaseModel):
    token: str
    expires_at: str
    websocket_url: str


class SessionStartResponse(BaseModel):
    session_id: str
    procedure: Procedure
    system_prompt: str
    tools: list[GeminiToolDefinition]
    gemini_config: GeminiConfig
    ephemeral_token: EphemeralTokenResponse
