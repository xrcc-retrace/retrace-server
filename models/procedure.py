from typing import Literal

from pydantic import BaseModel

# Closed taxonomy of procedure categories. Must stay in sync with the iOS
# chip list in DiscoverViewModel. "All" is a UI affordance only — Gemini
# never picks it. "Other" is the catch-all when nothing else fits.
ProcedureCategory = Literal[
    "Assembly",
    "Electrical",
    "Calibration",
    "Maintenance",
    "Repair",
    "Inspection",
    "Other",
]
CATEGORY_NAMES: tuple[str, ...] = (
    "Assembly",
    "Electrical",
    "Calibration",
    "Maintenance",
    "Repair",
    "Inspection",
    "Other",
)


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


class ProcedureSource(BaseModel):
    """One online source cited in a web-search-derived procedure.

    Populated only for procedures with `source_type='web'`. The list
    lives at procedure level (not per-step); iOS renders it as a tappable
    "Sources" footer that opens each `url` in SFSafariViewController.
    """
    title: str
    url: str
    snippet: str = ""


class Procedure(BaseModel):
    id: str
    title: str
    description: str
    steps: list[ProcedureStep]
    total_duration: float
    source_video: str
    created_at: str
    # Procedure-level iconography. `icon_symbol` is an SF Symbol name picked
    # by Gemini at extraction time (e.g. "humidifier", "refrigerator").
    # `icon_emoji` is a graceful fallback when the symbol doesn't resolve on
    # the client's iOS version. Either may be empty for legacy rows that
    # predate the backfill.
    icon_symbol: str = ""
    icon_emoji: str = ""
    # One of CATEGORY_NAMES. Picked by Gemini at extraction time. Defaults
    # to "Other" for legacy rows that predate the backfill.
    category: str = "Other"
    # Online sources cited by the troubleshoot web-search flow. Empty for
    # video- and manual-derived procedures.
    sources: list[ProcedureSource] = []


class StepBoundingBox(BaseModel):
    """Bounding box on the source PDF page for a manual-derived step.

    Coordinates are percentages of page dimensions (0..100). `(x, y)` is
    the top-left corner; `(w, h)` is width/height. All-zero box (the
    default) means "no per-step figure exists" — the ingest pipeline
    skips the image for this step entirely; the iOS row renders the
    description with no inline image. There is no full-page fallback:
    a wrong bbox or a step without a figure both result in no image.
    """
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0


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
    # 1-indexed PDF page that contains this step's figure. Populated by the
    # manual-ingest prompt; defaults to 0 ("no page mapping") so video
    # extraction still validates against this schema.
    source_page: int = 0
    # Region within source_page that holds just this step's illustration.
    # Page-percent coordinates 0..100. Empty/zero box → no image rendered
    # for this step (the iOS row shows description only, no AsyncImage).
    source_bbox: StepBoundingBox = StepBoundingBox()


class GeminiProcedureOutput(BaseModel):
    title: str
    description: str
    steps: list[GeminiProcedureStep]
    total_duration: float
    # SF Symbol name representing the main physical object of the procedure
    # (e.g. "humidifier", "refrigerator", "iphone.gen3"). Picked by Gemini.
    icon_symbol: str = ""
    # Single emoji fallback used when the SF Symbol doesn't resolve on the
    # client's iOS version (e.g. "💧", "🧊", "📱").
    icon_emoji: str = ""
    # Closed-enum category Gemini assigns at extraction time. Used by the
    # iOS Discover chip filter. "Other" is the catch-all.
    category: ProcedureCategory = "Other"
    # Online sources for web-search-derived procedures. The structured-output
    # call echoes back the URL list provided in the prompt with a
    # one-sentence `snippet` per source explaining its contribution.
    # Empty for video- and manual-derived procedures.
    sources: list[ProcedureSource] = []
