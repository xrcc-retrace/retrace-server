import asyncio
import logging

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from config import GEMINI_MODEL
from models.procedure import GeminiProcedureOutput

FALLBACK_MODEL = "gemini-2.5-flash"
PRO_TIMEOUT_MS = 180_000
FLASH_TIMEOUT_MS = 120_000

logger = logging.getLogger(__name__)

PROCEDURE_PROMPT = """You are an expert procedure analyst. Watch this video of an expert performing a hands-on task and extract a detailed step-by-step procedure (Standard Operating Procedure).

For each step:
1. Identify the distinct action being performed
2. Note the exact timestamps (start and end) in seconds in the video
3. Write clear instructions as if teaching a first-time learner to replicate this
4. Include any tips the expert demonstrates or narrates (technique, positioning, tool handling)
5. Note any safety warnings or common mistakes visible
6. Describe the visual completion criteria — what does "done" look like for this step? Focus on observable object states that a camera could detect (e.g., "the collar is completely detached from the bar", "the weight plate is resting on the ground, no longer on the bar", "the lid is fully closed"). Be specific about what changed visually from start to finish of the step.
7. Produce EXACTLY THREE observable visual errors for this step. These are the three things that, if a camera persistently saw them during live coaching, should cause the AI coach to interject and correct the learner. Each error must be written as a single full declarative sentence in the same style as `completion_criteria` — a described visible STATE, never an action, intent, or judgment.

   HARD RULES — every error must pass all of these:
   - Describes a physical state visible in a single frame (or a couple of consecutive frames). Must NOT require counting events over time, measuring duration, or inferring what the learner meant.
   - Names the specific object and the specific wrong configuration (orientation, position, pairing, which sub-part is being touched). No vague nouns like "thing" or "item".
   - Contains NO judgment words ("careless", "sloppy", "wrong-ish"), NO cognitive claims ("seems confused", "distracted", "not paying attention"), NO abstract faults ("technique is off", "something is off", "not quite right"). These are not observable states.
   - The three errors are mutually distinct — no paraphrases of the same mistake with different wording.

   USEFUL TEMPLATE for the three errors (adapt if a category genuinely doesn't fit):
   (a) Target missing — e.g., "[the key object] is not visible in the camera view at all." Almost always applicable to hands-on tasks; include it unless the step is purely verbal or uses no anchoring object.
   (b) Wrong object state — the object IS visible but in a state that contradicts what this step requires (wrong orientation, wrong item in hand, closed-when-should-be-open, pre-condition not satisfied).
   (c) Wrong interaction — the learner is acting on the correct object in the wrong way (wrong button, wrong grip, wrong side, wrong direction of motion detectable across frames).

   GOOD examples (note specificity and sentence form):
   - "No mouse is visible in the camera view at all."
   - "The container lid is still closed, though this step requires the container to be open before the learner continues."
   - "The learner is pressing a button other than the circular button at the center of the mouse's base."

   BAD examples — do NOT produce output in these shapes:
   - "The learner is being careless." — judgment, not an observable state.
   - "The technique is off." — abstract, no named object or state.
   - "The button has not been pressed the correct number of times yet." — requires counting events over time, not a visible state.
   - "The user seems confused." — cognitive claim, not visible.
   - "Something is wrong with how the mouse is being held." — vague, no specific wrong configuration.

   Output exactly three entries per step — no more, no fewer. These strings will be injected verbatim into a live coaching system prompt, so clarity and specificity directly determine whether the AI interjects correctly or hallucinates.

The expert may narrate what they're doing — use both visual and audio cues to build the procedure.
Be precise with timestamps. Each step should be a discrete, actionable unit.
Steps should not overlap in time. Cover the entire video duration.
Keep step titles short and action-oriented (e.g., "Remove the drip tray", "Insert cleaning tablet").

In addition to the steps, also produce two top-level iconography fields representing the MAIN PHYSICAL OBJECT this procedure works on (the appliance / device, not the action).

- `icon_symbol`: a single Iconify icon ID in the form "<prefix>:<name>" — e.g. "lucide:refrigerator", "material-symbols:humidifier-large", "phosphor:speaker-high", "tabler:washing-machine". Prefer these packs in order: lucide (clean, broad coverage), material-symbols (best for appliances), phosphor, tabler. If you don't know an exact icon name, prefer common nouns: refrigerator, microwave, oven, washing-machine, humidifier, fan, lightbulb, tv, speaker, monitor, laptop, mouse, headphones, printer, gamepad, camera, wrench, screwdriver, coffee, bicycle, car. The icons render white on a dark surface — pick line-style monochrome icons, not multi-color packs (avoid emojione, twemoji, fxemoji).

- `icon_emoji`: a single emoji that represents the same object (e.g. 💧 humidifier, 🧊 fridge, 📱 iPhone). Used as a fallback when the icon CDN is unreachable.

Both fields are required. Pick a single icon_symbol and a single icon_emoji that represent the same physical object.

Also pick a `category` for this procedure — exactly one value from this closed list:
- "Assembly" — joining parts or components into a working unit for the first time (e.g. mounting a PCB into an enclosure, building a hydraulic pump subassembly, IKEA-style furniture build).
- "Electrical" — wiring, terminating, or configuring electrical/electronic systems (e.g. crimping a wire harness, programming a PLC I/O module, replacing a fuse block).
- "Calibration" — setting, adjusting, or verifying that a machine or instrument meets a specified output (e.g. zeroing a torque wrench, tuning a CNC backlash, adjusting a flow sensor setpoint).
- "Maintenance" — scheduled upkeep that keeps a working system working: cleaning, lubricating, replacing consumables, descaling (e.g. descaling a coffee machine, replacing a filter on a PM schedule, lubricating a chain).
- "Repair" — diagnosing and fixing a system that has failed or degraded below spec (e.g. replacing a seized bearing after a machine trip, reflowing a cold solder joint).
- "Inspection" — verifying conformance to a standard without making any change (e.g. visual weld inspection, dimensional check, post-assembly continuity test).
- "Other" — only when none of the above fits.

Precedence rule for ambiguous cases: if the procedure changes the system state, classify by why — scheduled/preventive → Maintenance; unplanned/corrective → Repair; adjustment to spec → Calibration. If it only checks or records without making any change, it is Inspection. Pick "Other" only as a last resort."""


_client: genai.Client | None = None


def _get_client(api_key: str) -> genai.Client:
    """Lazy module-level Gemini client, mirroring ephemeral_token._get_client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


async def process_expert_video(video_path: str, api_key: str) -> GeminiProcedureOutput:
    """Upload video to Gemini and extract a structured procedure."""
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
                    response_schema=GeminiProcedureOutput,
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
        return GeminiProcedureOutput.model_validate_json(response.text)
    finally:
        try:
            await asyncio.to_thread(client.files.delete, name=video_file.name)
            logger.info("Cleaned up uploaded file: %s", video_file.name)
        except Exception:
            logger.warning("Failed to clean up uploaded file: %s", video_file.name)
