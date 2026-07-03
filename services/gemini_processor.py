import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from config import GEMINI_MODEL
from models.procedure import GeminiVideoProcedureOutput
from services.metadata_picker import pick_metadata

FALLBACK_MODEL = "gemini-2.5-flash"
PRO_TIMEOUT_MS = 180_000
FLASH_TIMEOUT_MS = 120_000

logger = logging.getLogger(__name__)

# System prompt focuses on what's distinctive about the video flow:
# real timestamps that drive ffmpeg cuts, audio + visual cue fusion,
# and the rigorous error_criteria block (those strings are injected
# verbatim into the Live coaching prompt's heartbeat path, so quality
# matters). Per-field shape rules (title/description/tips/warnings/
# completion_criteria) live in GeminiVideoProcedureStep field
# descriptions in models/procedure.py.
PROCEDURE_PROMPT = """You are an expert procedure analyst. Watch this video of an expert performing a hands-on task and extract a detailed step-by-step procedure (Standard Operating Procedure).

The expert may narrate what they're doing — use both visual and audio cues to build the procedure.

Be precise with timestamps. Each step should be a discrete, actionable unit. Steps should not overlap in time and together should cover the entire video duration.

=== SPATIAL END-STATE + VERIFIABILITY (critical) ===
A separate vision model will later confirm each step by comparing the learner's live camera frame against this step's `completion_criteria`. That model is reliable at STATIC SPATIAL RELATIONSHIPS (object A now on top of / inside / removed from object B; lid open vs closed; part present vs absent) and UNRELIABLE at counting events, timing, torque, and sub-millimeter alignment.

For each step, mentally compare the scene at the END of the previous step to the scene at the END of this step and name the spatial change:
- Write `completion_criteria` in the PRESENT TENSE, describing what IS VISIBLE in a single frame when done — verifiable without remembering the "before" frame. Assert the POSITIVE presence/configuration of named objects, never an absence, removal, motion, or "no longer" (the vision model sees a current state, not a change). Rewrite the anti-patterns:
  - "the cable is removed from the port"            → "the port is empty, with no cable inserted"
  - "the user's right hand lifts off the keyboard"  → (usually `none`; if a result is visible, name it, e.g. "the login screen is shown")
  - "the lid has been screwed onto the bottle"      → "the lid sits flush on top of the bottle"
  - "a portion of the avocado has been eaten"       → (a delta; `none` unless a stable end-state is visible)
  - "the finger is no longer touching the escape key" → `none` (a released key looks identical to before the press)
- Make the criterion SPATIALLY EXACT — the vision model can only check what the sentence names:
  - Call every part by its EXACT name from the parts glossary you produce below ("the M6 mandrel", never "the pin" or "the piece"). The verifier uses the glossary to identify parts on camera, so criterion and glossary must share vocabulary.
  - When a step joins, inserts, or mounts parts, state WHICH part sits in/on/through WHICH ("the anvil nut is threaded onto the mandrel tip", not "the parts are assembled").
  - When a part has a visibly asymmetric feature (flange, taper, thread, label, cap), state its ORIENTATION relative to a stable landmark whenever a wrong orientation would look different in-frame ("the rivet nut sits on the mandrel with its flanged head against the nosepiece"). Skip orientation words for symmetric parts where no wrong orientation is visible.
  - Rewrite the vague anti-patterns:
    - "the parts are assembled together"   → "the rivet nut is threaded onto the mandrel, its flanged head flush against the nosepiece"
    - "the piece is inserted"              → "the mandrel passes through the rivet nut, its threaded tip protruding past the flange"
  FORBIDDEN in completion_criteria: present-perfect/past ("has been", "has pressed", "has tapped"), motion verbs ("lifts off", "is lowered", "removes"), and any "no longer".
- Set `visual_verifiability`:
  - `reliable` — a clear static spatial-configuration change detectable in one frame.
  - `marginal` — a smaller or orientation-only change a vision model can sometimes miss.
  - `none` — NO single-frame visual end-state. A key press, button tap, or brief finger contact leaves no lasting visual change, so grade it `none` — a released key/button looks identical to before the press. Counting presses, waiting, tightening, and fine alignment are also `none`. Keystroke and device-reset sequences are canonically `none`. Before choosing `none`, hunt for a spatial PROXY for the RESULT (e.g. "the LED is solid green" after N presses, "the panel sits flush", "the screen shows the reset menu") and, if one exists, use it with `reliable`/`marginal`. Only use `none` when no visual consequence exists — that step is advanced verbally instead.

For the per-step `error_criteria` field, produce UP TO THREE observable visual errors — only ones that genuinely exist for this step. An empty list is acceptable; never pad to three with paraphrases. Each must be a single full declarative sentence describing a visible STATE — never an action, intent, or judgment.

HARD RULES — every error must pass all of these:
- Describes a COMMITTED, PERSISTENT wrong state visible in a single frame — a configuration the learner has actually settled into, not a transient absence or a not-yet-done state. Must NOT require counting events over time, measuring duration, or inferring what the learner meant.
- Is NOT true during a normal pause. Never "X is not visible", "the hand is not touching …", or "not yet …" — those fire constantly while the learner simply looks away or rests between actions, causing false interjections.
- Names the specific object and the specific wrong configuration (orientation, position, pairing, which sub-part is being touched). No vague nouns like "thing" or "item" — use the part's EXACT name from the parts glossary, and for orientation errors name the visible feature that gives the mistake away ("flanged head facing away from the tool").
- Contains NO judgment words ("careless", "sloppy", "wrong-ish"), NO cognitive claims ("seems confused", "distracted", "not paying attention"), NO abstract faults ("technique is off", "something is off", "not quite right"). These are not observable states.
- The errors are mutually distinct — no paraphrases of the same mistake with different wording.

USEFUL TEMPLATE for the errors (use the ones that fit; skip any that don't):
(a) Wrong object state — the object is visible but in a state that contradicts what this step requires (wrong orientation, wrong item in hand, closed-when-should-be-open, pre-condition not satisfied).
(b) Wrong interaction — the learner is acting on the correct object in the wrong way (wrong button, wrong grip, wrong side).
(c) Target missing — ONLY when the object's absence is itself the mistake for this step (e.g. a required part was never brought into the workspace), NOT mere off-camera framing during normal work. Use sparingly.

GOOD examples (note specificity and sentence form):
- "The mouse is right-side up, but this step needs its base facing the camera."
- "The container lid is still closed, though this step requires the container to be open before the learner continues."
- "The learner is pressing a button other than the circular button at the center of the mouse's base."
- "The rivet nut is threaded onto the mandrel backwards, with its flanged head facing away from the nosepiece."

BAD examples — do NOT produce output in these shapes:
- "The mouse is not visible in the camera view." — transient absence; true whenever the learner glances away. (Only allowed if the object's absence is itself the step's mistake.)
- "The learner's hand is not touching any key." — true during every normal pause; fires constantly.
- "The learner is being careless." — judgment, not an observable state.
- "The technique is off." — abstract, no named object or state.
- "The button has not been pressed the correct number of times yet." — requires counting events over time, not a visible state.
- "The user seems confused." — cognitive claim, not visible.

Output at most three entries per step — fewer is fine, zero is fine when nothing distinct applies. These strings are handed to the verification model, so clarity and specificity directly determine whether the coach interjects correctly or hallucinates.

=== PARTS GLOSSARY ===
Separately from the steps, build the procedure-level `parts` list: every DISTINCT physical part, component, or tool the expert handles anywhere in the video. During the live coaching session the learner will hold parts up to the camera and ask "is this the right piece?" — the coach answers from these descriptions, so write each `appearance` for VISUAL DISAMBIGUATION on a live camera feed:
- Name the color, material/finish (matte, glossy, knurled, threaded), overall shape, and approximate size relative to a hand or the main tool.
- Lead with the single most distinctive feature — the thing that separates this part from everything else on the workbench.
- End EVERY entry's `appearance` with an explicit contrast against the part in THIS kit it is most likely to be confused with on camera — name what visibly separates them (length, solid vs hollow, threaded vs smooth, flanged vs plain, matte vs shiny, color). E.g. "unlike the rivet nut, the mandrel is longer, solid, and threaded on only one end." The learner sees these two parts as similar; your job is to give the coach the one visible cue that tells them apart.
- If a part resembles nothing else in the kit, instead close with the single feature that makes it unmistakable.
- One entry per part — no duplicates across steps; describe the part itself, not what is done with it.
- These names are the CANONICAL VOCABULARY for the whole procedure: every `completion_criteria` and `error_criteria` sentence that mentions a part must reuse its glossary name verbatim.

Output JSON matching the GeminiVideoProcedureOutput schema. Field-level shape requirements (title format, description length, display_summary brevity, tips/warnings, completion_criteria observability) live in the response schema's field descriptions — follow those."""


_client: genai.Client | None = None


def _get_client(api_key: str) -> genai.Client:
    """Lazy module-level Gemini client, mirroring ephemeral_token._get_client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


async def process_expert_video(video_path: str, api_key: str) -> dict[str, Any]:
    """Upload video to Gemini and extract a structured procedure.

    Returns a dict carrying the procedure's top-level fields plus the
    full step list. icon_symbol / icon_emoji / category are populated
    server-side via pick_metadata after Pro returns — they aren't part
    of the GeminiVideoProcedureOutput schema, so the caller (router)
    reads them off this dict, not off Pydantic attributes.

    Shape:
        {
            "title": str,
            "description": str,
            "total_duration": float,    # from Pro; router will overwrite
                                         # with ffprobe duration
            "icon_symbol": str,         # from pick_metadata
            "icon_emoji": str,          # from pick_metadata
            "category": str,            # from pick_metadata
            "steps": list[dict],        # each step as a dict
            "parts": list[dict],        # visual parts glossary ({name, appearance})
        }
    """
    client = _get_client(api_key)

    # Upload video via Files API (blocking call, run in thread)
    logger.info("Uploading video to Gemini Files API: %s", video_path)
    video_file = await asyncio.to_thread(
        client.files.upload, file=video_path
    )
    logger.info("Upload complete, file name: %s, state: %s", video_file.name, video_file.state)

    try:
        # Poll until processing is complete
        while video_file.state == "PROCESSING":
            logger.info("Video still processing, waiting...")
            await asyncio.sleep(3)
            video_file = await asyncio.to_thread(
                client.files.get, name=video_file.name
            )

        if video_file.state == "FAILED":
            raise RuntimeError(f"Gemini video processing failed for {video_file.name}")

        logger.info("Video processing complete, generating procedure...")

        contents = [
            types.Content(parts=[
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type,
                ),
                types.Part.from_text(text=PROCEDURE_PROMPT),
            ])
        ]

        def _generate(model: str, timeout_ms: int):
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiVideoProcedureOutput,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                ),
            )

        try:
            response = await asyncio.to_thread(_generate, GEMINI_MODEL, PRO_TIMEOUT_MS)
        except ServerError as e:
            if e.code != 503:
                raise
            logger.warning("%s returned 503, falling back to %s", GEMINI_MODEL, FALLBACK_MODEL)
            response = await asyncio.to_thread(_generate, FALLBACK_MODEL, FLASH_TIMEOUT_MS)

        logger.info("Gemini response received, parsing...")
        parsed = GeminiVideoProcedureOutput.model_validate_json(response.text)

        # Pick icon + category via Flash now that we have the title and
        # description Pro just produced. Sequential — for video uploads
        # we have no product_name input upfront, so we can only run this
        # after the synthesis completes.
        metadata = await pick_metadata(
            context=f"Title: {parsed.title}\nDescription: {parsed.description}",
        )

        return {
            "title": parsed.title,
            "description": parsed.description,
            "total_duration": parsed.total_duration,
            "icon_symbol": metadata["icon_symbol"],
            "icon_emoji": "",
            "category": metadata["category"],
            "steps": [s.model_dump() for s in parsed.steps],
            "parts": [p.model_dump() for p in parsed.parts],
        }
    finally:
        try:
            await asyncio.to_thread(client.files.delete, name=video_file.name)
            logger.info("Cleaned up uploaded file: %s", video_file.name)
        except Exception:
            logger.warning("Failed to clean up uploaded file: %s", video_file.name)
