"""PDF manual → structured SOP via Gemini Files API.

Shared by both the Expert mode (PDF upload from iOS) and Troubleshoot
mode (Live model decides it needs a manual). Caller picks the prompt
flavor via the `mode` parameter:

- `mode="expert"` — extract the end-to-end procedure described in the
  manual. `focus_problem` / `diagnostic_summary` are ignored.
- `mode="troubleshoot"` — scope to a specific fault path; both
  `focus_problem` and `diagnostic_summary` are required.

After Gemini returns a structured procedure, the pipeline renders each
step's `source_page` as a PNG and stores it under
`config.MANUAL_IMAGES_DIR/{procedure_id}/step_{n}.png`. The PNG path is
written to the existing `step.clip_url` field with `clip_status='ready'`,
so learner endpoints serve manual procedures without any schema change.
Steps that don't get a usable image (out-of-range page, render failure)
fall back to `clip_url=None`, `clip_status='none'`.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from config import DATA_DIR, GEMINI_API_KEY, GEMINI_MODEL, MANUAL_IMAGES_DIR
from models.procedure import GeminiManualProcedureOutput
from services.database import (
    create_procedure_placeholder,
    save_procedure,
    update_procedure_status,
)
from services.manual_prompts import EXPERT_MANUAL_PROMPT, TROUBLESHOOT_MANUAL_PROMPT
from services.metadata_picker import pick_metadata
from services.pdf_images import (
    BBoxRejected,
    list_image_regions,
    page_rect as _get_page_rect,
    percent_to_rect,
    rect_to_page_percent,
    render_page_png,
)
import fitz

FALLBACK_MODEL = "gemini-2.5-flash"
PRO_TIMEOUT_MS = 180_000
FLASH_TIMEOUT_MS = 120_000

logger = logging.getLogger(__name__)

MANUALS_DIR = DATA_DIR / "manuals"
MANUALS_DIR.mkdir(parents=True, exist_ok=True)

# Tracks async ingest jobs so the phone can poll status.
# Shape: {manual_id: {"status": "uploaded|processing|ready|failed",
#                      "procedure_id": ..., "error": ..., "filename": ...,
#                      "path": ..., "created_at": ...}}
_manuals: dict[str, dict] = {}


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
        "status": "uploaded",
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
    *,
    mode: Literal["expert", "troubleshoot"],
    product_name: str,
    focus_problem: str = "",
    diagnostic_summary: str = "",
) -> str | None:
    """Run the PDF → SOP pipeline. Returns the generated procedure_id on
    success, None on failure. Status is written to the _manuals registry
    so the phone can poll via GET /api/{expert,troubleshoot}/manuals/{id}.

    `mode` selects the prompt template. Troubleshoot mode requires
    `focus_problem` (and ideally `diagnostic_summary` for context).
    """
    if mode not in ("expert", "troubleshoot"):
        raise ValueError(f"Unknown mode: {mode!r}")
    if mode == "troubleshoot" and not focus_problem:
        raise ValueError("focus_problem is required for troubleshoot mode")

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
    create_procedure_placeholder(
        procedure_id,
        source_video=str(pdf_path),
        created_at=created_at,
        source_type="manual",
    )

    client = _get_client()

    logger.info("Uploading manual PDF to Gemini Files API: %s (mode=%s)", pdf_path, mode)
    manual_file = None
    try:
        manual_file = await asyncio.to_thread(client.files.upload, file=str(pdf_path))
        while manual_file.state == "PROCESSING":
            logger.info("Manual still processing, waiting...")
            await asyncio.sleep(2)
            manual_file = await asyncio.to_thread(client.files.get, name=manual_file.name)

        if manual_file.state == "FAILED":
            raise RuntimeError(f"Gemini rejected manual: {manual_file.name}")

        prompt = _build_prompt(
            mode=mode,
            product_name=product_name,
            focus_problem=focus_problem,
            diagnostic_summary=diagnostic_summary,
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
                    response_schema=GeminiManualProcedureOutput,
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

        parsed = GeminiManualProcedureOutput.model_validate_json(response.text)

        # Render per-step reference images. Failure on a single step is
        # tolerated: that step lands with clip_url=None / clip_status='none'.
        # Run image rendering and metadata picking concurrently — rendering
        # is CPU/disk-bound (PyMuPDF in a worker thread), pick_metadata is
        # network-bound on Flash, so they overlap cleanly.
        image_urls, metadata = await asyncio.gather(
            asyncio.to_thread(
                _render_step_images,
                pdf_path=pdf_path,
                procedure_id=procedure_id,
                steps=parsed.steps[:15],
            ),
            pick_metadata(
                context=f"Product: {product_name}\nTitle: {parsed.title}\nDescription: {parsed.description}",
            ),
        )

        procedure_row = {
            "id": procedure_id,
            "title": parsed.title,
            "description": parsed.description,
            # PDF procedures aren't backed by a video — total_duration is
            # meaningless. Hardcode 0.0; the new schema doesn't carry
            # this field at all (that's the slim-down).
            "total_duration": 0.0,
            "status": "completed",
            "source_type": "manual",
            "icon_symbol": metadata["icon_symbol"],
            "category": metadata["category"],
        }
        step_rows = []
        for idx, s in enumerate(parsed.steps[:15], start=1):
            url = image_urls.get(s.step_number)
            step_rows.append({
                "step_number": s.step_number,
                "title": s.title,
                "description": s.description,
                # Server-fill placeholder timestamps — the manual schema
                # doesn't ask Pro for these, so we synthesize 0/30/60…
                # for API compatibility with the database layer.
                "timestamp_start": (idx - 1) * 30.0,
                "timestamp_end": idx * 30.0,
                "tips": s.tips,
                "warnings": s.warnings,
                "completion_criteria": s.completion_criteria,
                "error_criteria": s.error_criteria,
                "clip_url": url,
                "clip_status": "ready" if url else "none",
            })
        save_procedure(procedure_row, step_rows)

        entry["status"] = "ready"
        entry["procedure_id"] = procedure_id
        logger.info(
            "Manual ingest complete: manual=%s → procedure=%s (%d steps, %d images)",
            manual_id, procedure_id, len(step_rows),
            sum(1 for r in step_rows if r["clip_url"]),
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


def _build_prompt(
    *,
    mode: str,
    product_name: str,
    focus_problem: str,
    diagnostic_summary: str,
) -> str:
    if mode == "expert":
        return EXPERT_MANUAL_PROMPT.format(product_name=product_name)
    focus_title = (focus_problem.split(".")[0] or focus_problem)[:60]
    return TROUBLESHOOT_MANUAL_PROMPT.format(
        product_name=product_name,
        focus_problem=focus_problem,
        focus_title=focus_title,
        diagnostic_summary=diagnostic_summary or "(no diagnostic summary provided)",
    )


def _render_step_images(
    *,
    pdf_path: Path,
    procedure_id: str,
    steps,
) -> dict[int, str]:
    """Crop each step's per-step figure to a PNG. Returns a map of
    step_number → public URL.

    Two-stage pipeline:
      1. Use PyMuPDF to detect raster images embedded on `source_page`.
         Filters out header logos / page numbers / full-page backgrounds.
      2. Snap Gemini's `source_bbox` to the detected image whose bbox is
         best contained inside it. The snapped region is the actual crop
         — this gives a tight illustration-only crop instead of the
         "illustration + caption text" Gemini bboxes tend to produce.

    Fallbacks:
      - No detected images on the page → use Gemini's bbox raw.
      - Detected images but none meaningfully overlapping Gemini's bbox →
        use Gemini's bbox raw.
      - Empty/zero source_bbox → skip image entirely. The iOS step row
        renders description only. NO full-page fallback.
    """
    out_dir = MANUAL_IMAGES_DIR / procedure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    urls: dict[int, str] = {}

    # Cache page-level data so we open the PDF once per page, not per step.
    page_cache: dict[int, tuple["fitz.Rect", list["fitz.Rect"]]] = {}

    for s in steps:
        page_1indexed = s.source_page or 0
        if page_1indexed <= 0:
            continue

        sb = getattr(s, "source_bbox", None)
        if sb is None or sb.w <= 0 or sb.h <= 0:
            # Gemini said "no figure for this step" (e.g. a purely textual
            # step like "wait 30 seconds"). Skip the image entirely.
            logger.info(
                "Step %d has no source_bbox; rendering no image.",
                s.step_number,
            )
            continue

        # Look up detected image regions on this page (cached).
        page_idx = page_1indexed - 1
        if page_1indexed not in page_cache:
            try:
                pr = _get_page_rect(pdf_path, page_idx)
                regions = list_image_regions(pdf_path, page_idx)
            except Exception as exc:
                logger.warning(
                    "Failed to inspect page %d for figure detection: %s",
                    page_1indexed, exc,
                )
                pr, regions = None, []
            page_cache[page_1indexed] = (pr, regions)
        page_rect_obj, detected = page_cache[page_1indexed]

        gemini_pct = (sb.x, sb.y, sb.w, sb.h)
        snapped_pct = None
        if page_rect_obj is not None and detected:
            snapped_pct = _snap_to_detected_image(gemini_pct, page_rect_obj, detected)
        if snapped_pct is None:
            # Snap couldn't find a detected illustration that matches
            # Gemini's bbox. Per product direction, we skip the image
            # rather than rendering Gemini's raw bbox (which is often a
            # cell-sized region containing caption text). The iOS step
            # row will render description-only.
            logger.info(
                "Step %d (page %d): NO SNAP — gemini=(%.1f,%.1f,%.1f,%.1f), "
                "%d detected regions on page; rendering no image.",
                s.step_number, page_1indexed, *gemini_pct, len(detected),
            )
            continue

        logger.info(
            "Step %d (page %d): SNAPPED gemini=(%.1f,%.1f,%.1f,%.1f) → "
            "detected=(%.1f,%.1f,%.1f,%.1f).",
            s.step_number, page_1indexed, *gemini_pct, *snapped_pct,
        )

        bbox = snapped_pct
        out_path = out_dir / f"step_{s.step_number}.png"
        try:
            render_page_png(pdf_path, page_idx, out_path, bbox=bbox)
            urls[s.step_number] = f"/api/manual_images/{procedure_id}/step_{s.step_number}.png"
        except BBoxRejected as exc:
            logger.info(
                "Step %d bbox rejected (page %d, bbox=%s); rendering no image. (%s)",
                s.step_number, page_1indexed, bbox, exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to render step %d image (page %d, bbox %s, %s): %s",
                s.step_number, page_1indexed, bbox, pdf_path.name, exc,
            )
    return urls


def _snap_to_detected_image(
    gemini_pct: tuple[float, float, float, float],
    page_rect_obj: "fitz.Rect",
    detected: list["fitz.Rect"],
) -> tuple[float, float, float, float] | None:
    """Pick the detected raster image bbox that's mostly contained inside
    Gemini's suggested bbox. Returns the snapped percent tuple, or None
    if no detection is a clean match.

    Containment threshold: the detected image must have at least 50% of
    its area inside Gemini's bbox. This catches the common case where
    Gemini's bbox covers (illustration + caption text) and the
    illustration is fully inside it, while rejecting cases where Gemini's
    bbox is so wrong that no illustration overlaps meaningfully.
    """
    gemini_rect = percent_to_rect(gemini_pct, page_rect_obj)
    best: fitz.Rect | None = None
    best_ratio = 0.0
    for det in detected:
        inter = gemini_rect & det
        if inter.is_empty:
            continue
        det_area = det.width * det.height
        if det_area <= 0:
            continue
        inter_area = inter.width * inter.height
        contained = inter_area / det_area
        # Tiebreaker: prefer the most-contained candidate; on ties, prefer
        # the larger detected image so we don't snap to a stray bullet.
        if contained > best_ratio + 1e-6 or (
            abs(contained - best_ratio) < 1e-6
            and best is not None
            and det_area > best.width * best.height
        ):
            best_ratio = contained
            best = det

    if best is None or best_ratio < 0.5:
        return None
    return rect_to_page_percent(best, page_rect_obj)


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
