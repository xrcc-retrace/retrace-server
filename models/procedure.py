from typing import Literal

from pydantic import BaseModel, Field

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


# --- Gemini structured-output schemas, one per flow -------------------------
# Each flow asks Pro for exactly what it needs and nothing else. There used
# to be a single shared GeminiProcedureOutput; we split it because each
# flow had different fields it actually consumed (timestamps drive ffmpeg
# only for video; source_page/bbox crop figures only for PDF; icon/emoji/
# category come from the server's pick_metadata Flash call now).
# Per-field shape rules live in Field(description=...) so the structured-
# output decoder reads them directly — no schema duplication in the prompt.
#
# The three step schemas inherit from _BaseProcedureStep, which carries
# the field shape that's actually shared. Each subclass overrides the
# descriptions on inherited fields where flow-specific framing matters
# (e.g. video tips say "expert demonstrates"; PDF tips say "manual calls
# out") and appends fields the flow needs (timestamps for video;
# source_page / source_bbox for PDF; nothing for web).


# --- Shared base for all step schemas ---------------------------------------

class _BaseProcedureStep(BaseModel):
    """Common step shape across video, PDF, and web flows. Subclasses
    add flow-specific fields and override descriptions where flow-
    specific framing is needed. Field descriptions here are the most
    generic phrasing; if a subclass doesn't override a field, the base
    description is what Pro reads via Pydantic's structured-output
    schema serialization."""

    step_number: int = Field(
        description="1-indexed contiguous step number, starting at 1.",
    )
    title: str = Field(
        description="Short and action-oriented.",
    )
    description: str = Field(
        description=(
            "1-3 sentences a first-time learner can follow. Concrete "
            "actions, not paraphrased platitudes."
        ),
    )
    tips: list[str] = Field(
        default=[],
        description=(
            "Short technique notes (when to be gentle, which "
            "direction). Empty list if none."
        ),
    )
    warnings: list[str] = Field(
        default=[],
        description="Safety callouts. Empty list if none.",
    )
    completion_criteria: str = Field(
        description=(
            "ONE observable end-state sentence written for VLM "
            "verification from a head-mounted camera. AVOID criteria "
            "that require counting events, measuring elapsed time, "
            "text recognition, torque, or sub-millimeter alignment — "
            "those aren't verifiable in a single camera frame."
        ),
    )


# --- Video flow (services.gemini_processor) ---------------------------------

class GeminiVideoProcedureStep(_BaseProcedureStep):
    title: str = Field(
        description=(
            'Short and action-oriented. Examples: "Remove the drip tray", '
            '"Insert cleaning tablet".'
        ),
    )
    description: str = Field(
        description=(
            "Clear instructions that teach a first-time learner to "
            "replicate this step. 1-3 sentences."
        ),
    )
    tips: list[str] = Field(
        default=[],
        description=(
            "Technique, positioning, or tool-handling notes the expert "
            "demonstrates or narrates. Empty list if none."
        ),
    )
    warnings: list[str] = Field(
        default=[],
        description=(
            "Safety warnings or common mistakes visible in the video. "
            "Empty list if none."
        ),
    )
    completion_criteria: str = Field(
        description=(
            "ONE observable end-state sentence describing what changed "
            "visually from start to finish — written for VLM "
            "verification from a head-mounted camera."
        ),
    )
    timestamp_start: float = Field(
        description="Start of step in source video, in seconds.",
    )
    timestamp_end: float = Field(
        description=(
            "End of step in source video, in seconds. Steps must not "
            "overlap and together must cover the entire video duration."
        ),
    )
    error_criteria: list[str] = Field(
        default=[],
        description=(
            "Exactly THREE observable visual errors. Each is a single "
            "declarative sentence describing a visible state (named "
            "object plus named wrong configuration) that, if a camera "
            "saw it during live coaching, should cause the AI to "
            "interject. See PROCEDURE_PROMPT for the (a)/(b)/(c) "
            "template and full hard rules."
        ),
    )


class GeminiVideoProcedureOutput(BaseModel):
    title: str = Field(
        description="Short, descriptive procedure title.",
    )
    description: str = Field(
        description="One or two sentences summarizing what the procedure does.",
    )
    total_duration: float = Field(
        description="Total procedure duration in seconds (matches source video).",
    )
    steps: list[GeminiVideoProcedureStep] = Field(
        description=(
            "Ordered list of discrete, actionable steps. Each step is "
            "a non-overlapping time range; together they cover the "
            "entire video duration."
        ),
    )


# --- Manual / PDF flow (services.manual_ingest) -----------------------------

class GeminiManualProcedureStep(_BaseProcedureStep):
    title: str = Field(
        description=(
            'Short and action-oriented. Examples: "Fill the water tank", '
            '"Press and hold the button until the LED blinks orange".'
        ),
    )
    tips: list[str] = Field(
        default=[],
        description=(
            "Short technique notes the manual calls out (when to be "
            "gentle, which direction to turn, helpful timing). Empty "
            "list if none."
        ),
    )
    warnings: list[str] = Field(
        default=[],
        description=(
            "Safety callouts the manual highlights. Empty list if none."
        ),
    )
    completion_criteria: str = Field(
        description=(
            "ONE observable end-state sentence written for VLM "
            "verification from a head-mounted camera. Prefer LED color/"
            "blink, lid open vs closed, container present vs absent, "
            "water level vs marker. If the manual describes an LED "
            "state machine, use those LED states — they are the "
            "cleanest visual signal. AVOID criteria that require "
            "counting events, measuring elapsed time, text recognition, "
            "torque, or sub-millimeter alignment — those aren't "
            "verifiable in a single camera frame."
        ),
    )
    error_criteria: list[str] = Field(
        default=[],
        description=(
            "Exactly three observable visual errors a learner might make "
            "on this step. Each is a single declarative sentence — named "
            "object plus named wrong configuration, visible in a single "
            "frame, no judgment words, no cognitive claims, no "
            "counting. Mutually distinct."
        ),
    )
    source_page: int = Field(
        default=0,
        description=(
            "1-indexed PDF page that CONTAINS this step's figure. If "
            "the step spans multiple pages, pick the one with the most "
            "relevant figure, photo, or LED diagram. Set to 0 if this "
            "step has no figure in the manual."
        ),
    )
    source_bbox: StepBoundingBox = Field(
        default=StepBoundingBox(),
        description=(
            "Bounding box of THIS step's figure on source_page, in "
            "page-percent coordinates 0-100. (x, y) is the top-left "
            "corner; (w, h) is width/height. Add ~5% padding. Crop "
            "ONLY this step's illustration — NOT the whole page, NOT a "
            "neighboring step's figure. If the step has no figure, "
            "return all-zero — the pipeline will skip the image. "
            "Do NOT guess; a wrong bbox is worse than no image."
        ),
    )


class GeminiManualProcedureOutput(BaseModel):
    title: str = Field(
        description=(
            'Procedure title in the form '
            '"<product name> — <2-5 word task summary>". Pick the most '
            'natural English summary (e.g. "Descaling", "Filter '
            'Replacement", "First-time Setup").'
        ),
    )
    description: str = Field(
        description=(
            "One or two sentences naming the product and the task, "
            "with no marketing language."
        ),
    )
    steps: list[GeminiManualProcedureStep] = Field(
        description=(
            "Step ordering follows the manual's own sequence — "
            "do not reorder."
        ),
    )


# --- Web-search synthesis schemas (services.troubleshoot.web_research) ------
# Smaller than the video / manual schemas because web procedures carry no
# real timestamps (no video), no PDF page mapping, and no error_criteria
# (those need a real source frame to anchor against). icon / emoji /
# category come from a separate pick_metadata Flash call. Pro emits
# exactly the fields the synthesis actually needs to think about.

class GeminiWebProcedureStep(_BaseProcedureStep):
    title: str = Field(
        description=(
            'Short and action-oriented. Examples: "Unplug the unit", '
            '"Remove the water tank".'
        ),
    )
    # description, tips, warnings, completion_criteria all inherited
    # verbatim from _BaseProcedureStep — web procedures use the most
    # generic framing because there's no source modality to be specific
    # about. No flow-specific fields to add.


class GeminiWebProcedureOutput(BaseModel):
    title: str = Field(
        description=(
            'Procedure title in the form '
            '"<product name> — <2-5 word task summary>".'
        ),
    )
    description: str = Field(
        description="One sentence naming the product and the fix.",
    )
    steps: list[GeminiWebProcedureStep] = Field(
        description="Cap at 15 steps. Prefer fewer when fewer suffice.",
    )
    # No sources — server attaches them from step 1's grounding metadata
    # (see services.troubleshoot.web_research.research_and_extract).
