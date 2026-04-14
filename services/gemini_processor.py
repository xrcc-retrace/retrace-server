import asyncio
import logging

from google import genai
from google.genai import types

from models.procedure import GeminiProcedureOutput

logger = logging.getLogger(__name__)

PROCEDURE_PROMPT = """You are an expert procedure analyst. Watch this video of an expert performing a hands-on task and extract a detailed step-by-step procedure (Standard Operating Procedure).

For each step:
1. Identify the distinct action being performed
2. Note the exact timestamps (start and end) in seconds in the video
3. Write clear instructions as if teaching a first-time learner to replicate this
4. Include any tips the expert demonstrates or narrates (technique, positioning, tool handling)
5. Note any safety warnings or common mistakes visible

The expert may narrate what they're doing — use both visual and audio cues to build the procedure.
Be precise with timestamps. Each step should be a discrete, actionable unit.
Steps should not overlap in time. Cover the entire video duration.
Keep step titles short and action-oriented (e.g., "Remove the drip tray", "Insert cleaning tablet")."""


async def process_expert_video(video_path: str, api_key: str) -> GeminiProcedureOutput:
    """Upload video to Gemini and extract a structured procedure."""
    client = genai.Client(api_key=api_key)

    # Upload video via Files API (blocking call, run in thread)
    logger.info("Uploading video to Gemini Files API: %s", video_path)
    video_file = await asyncio.to_thread(
        client.files.upload, file=video_path
    )
    logger.info("Upload complete, file name: %s, state: %s", video_file.name, video_file.state)

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

    # Generate structured procedure
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-pro",
        contents=[
            types.Content(parts=[
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type,
                ),
                types.Part.from_text(text=PROCEDURE_PROMPT),
            ])
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiProcedureOutput,
        ),
    )

    logger.info("Gemini response received, parsing...")
    result = GeminiProcedureOutput.model_validate_json(response.text)

    # Clean up uploaded file
    try:
        await asyncio.to_thread(client.files.delete, name=video_file.name)
        logger.info("Cleaned up uploaded file: %s", video_file.name)
    except Exception:
        logger.warning("Failed to clean up uploaded file: %s", video_file.name)

    return result
