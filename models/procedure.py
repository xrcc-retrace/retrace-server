from pydantic import BaseModel


class ProcedureStep(BaseModel):
    step_number: int
    title: str
    description: str
    timestamp_start: float
    timestamp_end: float
    tips: list[str] = []
    warnings: list[str] = []
    keyframe_url: str | None = None


class Procedure(BaseModel):
    id: str
    title: str
    description: str
    steps: list[ProcedureStep]
    total_duration: float
    source_video: str
    created_at: str


# Schema for Gemini structured output (no keyframe_url, no id/metadata)
class GeminiProcedureStep(BaseModel):
    step_number: int
    title: str
    description: str
    timestamp_start: float
    timestamp_end: float
    tips: list[str] = []
    warnings: list[str] = []


class GeminiProcedureOutput(BaseModel):
    title: str
    description: str
    steps: list[GeminiProcedureStep]
    total_duration: float
