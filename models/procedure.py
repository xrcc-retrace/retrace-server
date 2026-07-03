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

# Per-step gate for the out-of-band visual-verification oracle. Set by Gemini
# Pro at extraction time from the spatial delta between the end of the prior
# step and the end of this step:
#   reliable — a clear static spatial-configuration change detectable in one
#              frame (object moved onto/into another, lid open↔closed, part
#              present↔absent). Oracle polls; a single confident match advances.
#   marginal — a smaller/orientation-only change that a VLM can sometimes miss.
#              Oracle polls but requires sustained (multi-poll) confirmation.
#   none     — no single-frame visual end-state exists (counting, timing,
#              torque, sub-mm alignment). Oracle is NOT polled; the step is
#              verbal-only (learner says "done").
VisualVerifiability = Literal["reliable", "marginal", "none"]
VISUAL_VERIFIABILITY_LEVELS: tuple[str, ...] = ("reliable", "marginal", "none")


class ProcedureStep(BaseModel):
    step_number: int
    title: str
    description: str
    timestamp_start: float
    timestamp_end: float
    tips: list[str] = []
    warnings: list[str] = []
    # Condensed 1-2 sentence version of `description` for the glasses lens
    # (the lens card truncates long text; Live still speaks the full
    # description). Empty for legacy rows — clients fall back to description.
    display_summary: str = ""
    completion_criteria: str = ""
    # Observable visual error states the verification oracle watches for.
    # Populated by Gemini 2.5 Pro during extraction and editable in the expert
    # review flow.
    error_criteria: list[str] = []
    # Gate for the out-of-band verification oracle (see VisualVerifiability).
    # Defaults to "reliable" so legacy rows keep auto-advance behavior — the
    # oracle is itself a second gate, so a poorly-specified legacy step simply
    # never gets a positive verdict and falls back to verbal advancement.
    visual_verifiability: VisualVerifiability = "reliable"
    clip_url: str | None = None
    clip_status: str = "ready"  # "ready" | "failed"


class ProcedurePart(BaseModel):
    """One distinct physical part or tool handled in the procedure.

    Extracted by Gemini Pro from the expert video and embedded into the
    Live coaching prompt as a parts glossary, so the coach can answer
    "is this the right piece?" from a real description instead of
    agreeing with whatever the learner asserts.
    """
    name: str
    appearance: str


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
    # Visual parts glossary extracted from the expert video. Empty for
    # manual/web procedures and legacy rows predating the parts extraction.
    parts: list[ProcedurePart] = []
    # Optional reference image (e.g. a diagram) shown on the learner's phone
    # for this procedure. Served from /api/reference_images/. None when unset.
    reference_image_url: str | None = None


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
    display_summary: str = Field(
        default="",
        description=(
            "Condensed version of the description for a small heads-up "
            "display: 1-2 short sentences, max 160 characters, self-contained "
            "enough to act on. Keep the key action, part names, and "
            "quantities; cut rationale, caveats, and anything already covered "
            "by tips/warnings. Always materially shorter than the "
            "description — never a copy of it."
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
            "ONE standalone sentence, in the PRESENT TENSE, describing what "
            "IS VISIBLE in the frame once this step is done — verifiable from "
            "a single head-mounted-camera frame WITHOUT remembering an earlier "
            "frame. Assert POSITIVE presence/configuration of named objects "
            "(the port is empty; the lid rests on top of the bottle; the tray "
            "sits out of the slot). NEVER describe absence-of, removal-of, "
            "motion, or 'no longer' — those need the before-frame. Rewrite "
            "'the cable is removed from the port' as 'the port is empty with no "
            "cable inserted'. FORBIDDEN: present-perfect/past ('has been', 'has "
            "pressed', 'has tapped'), motion verbs ('lifts off', 'is lowered'), "
            "and any 'no longer'. AVOID criteria needing counting, elapsed "
            "time, text recognition, torque, or sub-millimeter alignment."
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
            "ONE standalone sentence, PRESENT TENSE, describing what IS "
            "VISIBLE once this step is done — verifiable from a single camera "
            "frame without the 'before' frame. Assert POSITIVE presence/"
            "configuration of named objects ('the glasses rest on top of the "
            "keyboard'; 'the drip tray sits out of the base on the counter'; "
            "'the port is empty with no cable inserted'). NEVER use absence/"
            "removal/motion/'no longer' or present-perfect ('has been', 'has "
            "pressed'): the oracle cannot see a change, only a current state."
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
            "UP TO three observable visual errors — only include ones that "
            "genuinely exist for this step; an empty list is acceptable. "
            "Each is a single declarative sentence naming an object in a "
            "COMMITTED, PERSISTENT wrong configuration (wrong orientation, "
            "wrong item in hand, wrong sub-part engaged). NOT a transient "
            "absence or not-yet-done state: never 'X is not visible', 'the "
            "hand is not touching …', or 'not yet …' — those are true during "
            "normal pauses and cause false interjections. Must be mutually "
            "distinct — never pad to three. See PROCEDURE_PROMPT for hard rules."
        ),
    )
    visual_verifiability: VisualVerifiability = Field(
        default="reliable",
        description=(
            "How reliably this step's completion can be confirmed from a "
            "single camera frame, judged from the spatial delta between the "
            "end of the prior step and the end of this step. 'reliable' = a "
            "clear static spatial-configuration change (object moved onto/"
            "into another, lid open↔closed, part present↔absent). 'marginal' "
            "= a smaller or orientation-only change a VLM can miss. 'none' = "
            "no single-frame visual end-state (counting presses, waiting, "
            "torque, sub-mm alignment). A key press, button tap, or brief "
            "finger contact leaves NO lasting visual change → 'none' (the "
            "finger being off the key looks identical to before the press); "
            "keystroke and reset sequences are canonically 'none'. Before "
            "choosing 'none', hunt for a spatial PROXY for the RESULT (e.g. "
            "'the LED is solid green', 'the panel sits flush') and, if one "
            "exists, use it with 'reliable'/'marginal' instead."
        ),
    )


class GeminiVideoProcedurePart(BaseModel):
    """Visual identification entry for one physical part/tool in the video.

    These strings are embedded verbatim into the Live coaching prompt so
    the coach can visually disambiguate parts when the learner holds one
    up and asks "is this the right piece?" — precision here directly
    determines whether the coach confirms the right part or the wrong one.
    """

    name: str = Field(
        description=(
            "The name the coach should call this part — prefer the term "
            "the expert uses in narration; otherwise a short natural name "
            'a first-time learner would understand (e.g. "nose piece", '
            '"M5 mandrel", "locking collar").'
        ),
    )
    appearance: str = Field(
        description=(
            "1-2 sentences of VISUAL identification written for telling "
            "parts apart on a live camera: color, material/finish, shape, "
            "approximate size relative to a hand or the main tool, and the "
            "single most distinctive feature. If the kit contains a "
            "similar-looking part it could be confused with, end with how "
            'to tell them apart (e.g. "unlike the smaller silver mandrel, '
            'this one is matte black and nearly palm-length").'
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
    parts: list[GeminiVideoProcedurePart] = Field(
        default=[],
        description=(
            "Every DISTINCT physical part, component, or tool the expert "
            "handles anywhere in the video — one entry each, no "
            "duplicates. Include parts that are attached/consumed along "
            "the way. Empty list only if the video genuinely involves no "
            "distinct handled parts."
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
            "UP TO three observable visual errors a learner might make on "
            "this step — only ones that genuinely exist; an empty list is "
            "fine. Each is a single declarative sentence — named object "
            "plus named wrong configuration, visible in a single frame, no "
            "judgment words, no cognitive claims, no counting. Mutually "
            "distinct — never pad with paraphrases."
        ),
    )
    visual_verifiability: VisualVerifiability = Field(
        default="reliable",
        description=(
            "How reliably this step's completion can be confirmed from a "
            "single camera frame. 'reliable' = a clear static spatial or "
            "LED-state change (lid open↔closed, tank present↔absent, LED "
            "now solid green). 'marginal' = a smaller/orientation change a "
            "VLM can miss. 'none' = no single-frame visual end-state "
            "(counting presses, waiting a duration, torque). Prefer a "
            "spatial/LED proxy over 'none' whenever the manual describes one."
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
