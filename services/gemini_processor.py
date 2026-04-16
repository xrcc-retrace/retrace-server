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

The expert may narrate what they're doing — use both visual and audio cues to build the procedure.
Be precise with timestamps. Each step should be a discrete, actionable unit.
Steps should not overlap in time. Cover the entire video duration.
Keep step titles short and action-oriented (e.g., "Remove the drip tray", "Insert cleaning tablet")."""


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
