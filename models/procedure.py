from pydantic import BaseModel


class ProcedureStep(BaseModel):
    step_number: int
    title: str
    description: str
    timestamp_start: float
    timestamp_end: float
    tips: list[str] = []
    warnings: list[str] = []
    completion_criteria: str = ""
    # Observable visual error states the AI coach watches for during the
    # heartbeat ([check]) path. Populated by Gemini 2.5 Pro during extraction
    # and editable in the expert review flow.
    error_criteria: list[str] = []
    clip_url: str | None = None
    clip_status: str = "ready"  # "ready" | "failed"


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
    completion_criteria: str
    # Observable errors the learner might make during this step. 2-5 short
    # camera-detectable phrases. Empty list allowed when no meaningful
    # visual errors apply (e.g., a purely verbal step).
    error_criteria: list[str] = []


class GeminiProcedureOutput(BaseModel):
    title: str
    description: str
    steps: list[GeminiProcedureStep]
    total_duration: float
