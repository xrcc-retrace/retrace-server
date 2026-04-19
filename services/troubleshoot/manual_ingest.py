"""PDF manual → structured SOP via Gemini Files API.

Mirrors services.gemini_processor.process_expert_video but:
  - Accepts a PDF path (not video).
  - No ffprobe, no clip extraction.
  - Prompt is tailored to service/troubleshooting manuals and takes a
    diagnostic_summary + focus_problem so the generated SOP zeros in on
    the actual fault path instead of dumping the whole manual.
  - Saves into the existing procedures/steps tables with source_type='manual'
    and every step's clip_status='none'.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from config import DATA_DIR, GEMINI_API_KEY, GEMINI_MODEL
from models.procedure import GeminiProcedureOutput
from services.database import (
    create_procedure_placeholder,
    save_procedure,
    update_procedure_status,
)

FALLBACK_MODEL = "gemini-2.5-flash"
PRO_TIMEOUT_MS = 180_000
FLASH_TIMEOUT_MS = 120_000

logger = logging.getLogger(__name__)

MANUALS_DIR = DATA_DIR / "manuals"
MANUALS_DIR.mkdir(parents=True, exist_ok=True)

# Tracks async ingest jobs so the phone can poll status.
# Shape: {manual_id: {"status": "processing|ready|failed", "procedure_id": ...,
#                      "error": ..., "filename": ..., "created_at": ...}}
_manuals: dict[str, dict] = {}


_MANUAL_PROMPT_TEMPLATE = """You are a service-manual analyst. Given this product's manufacturer manual (a PDF), extract a focused repair procedure for a specific reported problem.

Focus problem: {focus_problem}

Diagnostic summary (what was already ruled in/out during conversation with the user):
{diagnostic_summary}

Produce a Standard Operating Procedure with these rules:
1. Scope tightly to the focus problem — skip sections unrelated to this specific fault path.
2. Cap the procedure at 15 steps. Prefer fewer steps if that's enough to fix the problem.
3. Each step must have:
   - A short, action-oriented title (e.g., "Unplug the unit", "Remove the water tank").
   - A 1-3 sentence description that a first-time learner can follow.
   - A `completion_criteria` string describing the observable end-state (what "done" looks like), so the downstream coaching system can verify visually.
   - 2-5 `error_criteria` strings — observable error states that, if visible, should cause an AI coach to interject. Follow the same rules as video-extracted errors: single declarative sentence, specific object + specific wrong configuration, visible in a single frame, no judgment words, no abstract faults.
4. Include tips for technique (when to be gentle, which direction to turn, etc.) and safety warnings where the manual calls them out.
5. Use timestamps starting from 0.0 and ticking up in reasonable ~30-second increments per step. The manual is a PDF, not a video — timestamps here are just for API compatibility, not real clip positions.

Title the procedure: "{product_name} — {focus_title}" where focus_title summarizes the problem in 3-5 words.

Description: one or two sentences naming the product and the problem being addressed.

Output JSON matching the GeminiProcedureOutput schema."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def save_uploaded_manual(pdf_bytes: bytes, original_filename: str) -> str:
    """Save an uploaded PDF to disk and return the manual_id."""
    manual_id = str(uuid.uuid4())
    path = MANUALS_DIR / f"{manual_id}.pdf"
    path.write_bytes(pdf_bytes)
    _manuals[manual_id] = {
        "manual_id": manual_id,
        "status": "uploaded",  # ingest not yet kicked off
        "procedure_id": None,
        "error": None,
        "filename": original_filename,
        "path": str(path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    logger.info("Saved uploaded manual %s (%s, %d bytes)", manual_id, original_filename, len(pdf_bytes))
    return manual_id


def register_local_manual(manual_id: str, path: Path, filename: str) -> None:
    """Register a bundled/fixture PDF in the in-memory registry without
    writing a new copy. Used by the startup fixture loader.
    """
    _manuals[manual_id] = {
        "manual_id": manual_id,
        "status": "uploaded",
        "procedure_id": None,
        "error": None,
        "filename": filename,
        "path": str(path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }


def get_manual_status(manual_id: str) -> dict | None:
    return _manuals.get(manual_id)


def list_manuals() -> list[dict]:
    return list(_manuals.values())


async def ingest_manual(
    manual_id: str,
    product_name: str,
    focus_problem: str,
    diagnostic_summary: str,
) -> str | None:
    """Run the PDF → SOP pipeline. Returns the generated procedure_id on
    success, None on failure. Status is written to the _manuals registry
    so the phone can poll via GET /api/troubleshoot/manuals/{id}.
    """
    entry = _manuals.get(manual_id)
    if not entry:
        logger.warning("ingest_manual: unknown manual_id %s", manual_id)
        return None

    pdf_path = Path(entry["path"])
    if not pdf_path.exists():
        entry["status"] = "failed"
        entry["error"] = "pdf_missing_on_disk"
        return None

    entry["status"] = "processing"
    procedure_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    # Pre-register the procedure row so dashboard listings pick it up as "processing"
    create_procedure_placeholder(procedure_id, source_video=str(pdf_path), created_at=created_at)

    client = _get_client()

    logger.info("Uploading manual PDF to Gemini Files API: %s", pdf_path)
    manual_file = None
    try:
        manual_file = await asyncio.to_thread(client.files.upload, file=str(pdf_path))
        # PDFs come back ACTIVE quickly; tolerate PROCESSING just in case.
        while manual_file.state == "PROCESSING":
            logger.info("Manual still processing, waiting...")
            await asyncio.sleep(2)
            manual_file = await asyncio.to_thread(client.files.get, name=manual_file.name)

        if manual_file.state == "FAILED":
            raise RuntimeError(f"Gemini rejected manual: {manual_file.name}")

        focus_title = (focus_problem.split(".")[0] or focus_problem)[:60]
        prompt = _MANUAL_PROMPT_TEMPLATE.format(
            product_name=product_name,
            focus_problem=focus_problem,
            focus_title=focus_title,
            diagnostic_summary=diagnostic_summary or "(no diagnostic summary provided)",
        )
        contents = [
            types.Content(parts=[
                types.Part.from_uri(
                    file_uri=manual_file.uri,
                    mime_type=manual_file.mime_type or "application/pdf",
                ),
                types.Part.from_text(text=prompt),
            ]),
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
            logger.warning("Manual ingest: %s returned 503, falling back to %s", GEMINI_MODEL, FALLBACK_MODEL)
            response = await asyncio.to_thread(_generate, FALLBACK_MODEL, FLASH_TIMEOUT_MS)

        parsed = GeminiProcedureOutput.model_validate_json(response.text)

        procedure_row = {
            "id": procedure_id,
            "title": parsed.title,
            "description": parsed.description,
            "total_duration": parsed.total_duration,
            "status": "completed",
            "source_type": "manual",
        }
        step_rows = [
            {
                "step_number": s.step_number,
                "title": s.title,
                "description": s.description,
                "timestamp_start": s.timestamp_start,
                "timestamp_end": s.timestamp_end,
                "tips": s.tips,
                "warnings": s.warnings,
                "completion_criteria": s.completion_criteria,
                "error_criteria": s.error_criteria,
                # Manual-derived procedures have no video, hence no clips.
                "clip_url": None,
                "clip_status": "none",
            }
            for s in parsed.steps[:15]
        ]
        save_procedure(procedure_row, step_rows)

        entry["status"] = "ready"
        entry["procedure_id"] = procedure_id
        logger.info(
            "Manual ingest complete: manual=%s → procedure=%s (%d steps)",
            manual_id, procedure_id, len(step_rows),
        )
        return procedure_id

    except Exception as exc:
        logger.exception("Manual ingest failed: %s", exc)
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        update_procedure_status(procedure_id, "failed", error_message=entry["error"])
        return None
    finally:
        if manual_file is not None:
            try:
                await asyncio.to_thread(client.files.delete, name=manual_file.name)
                logger.info("Cleaned up uploaded manual file: %s", manual_file.name)
            except Exception:
                logger.warning("Failed to clean up uploaded manual: %s", manual_file.name)


def find_preloaded_for_product(product_name: str) -> dict | None:
    """If a manual-ingested procedure already covers this product, return
    a reference to it. Matches by lowercase substring against the
    procedure title.
    """
    from services.database import list_procedures
    needle = product_name.lower().strip()
    if not needle:
        return None
    for p in list_procedures():
        if p.get("source_type") != "manual":
            continue
        if needle in (p.get("title") or "").lower():
            return {
                "procedure_id": p["id"],
                "title": p["title"],
                "description": p.get("description", ""),
            }
    return None
