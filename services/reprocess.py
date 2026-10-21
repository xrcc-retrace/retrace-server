"""In-place re-extraction of video procedures.

Two responsibilities:

1. `process_video_into_procedure` — the shared Gemini-analysis → clip-extraction
   → DB-save core. Both the expert upload router (`routers/expert.py`) and the
   batch re-extractor below call it, so a fresh upload and a re-extraction always
   run the *exact same* pipeline and can't drift.

2. `reextract_stale_procedures` — the startup/on-demand driver. It finds every
   video-sourced procedure whose stored `extraction_version` is stale relative to
   `config.EXTRACTION_PIPELINE_VERSION` and re-runs the pipeline over the original
   source video, overwriting the row in place.

Why a version gate instead of "re-extract on every deploy": re-extraction hits
Gemini 2.5 Pro (minutes + real spend per procedure). Bumping the version string
in config.py is the deliberate signal that the pipeline changed and old rows are
now out of date; unrelated deploys leave every stamp current, so this is a no-op.

The re-extraction OVERWRITES procedures wholesale — any manual dashboard edits to
a re-extracted procedure are lost. That's the chosen trade-off (repopulation
correctness over edit preservation).
"""

import asyncio
import logging
import os
from pathlib import Path

from config import CLIPS_DIR, EXTRACTION_PIPELINE_VERSION, GEMINI_API_KEY, UPLOAD_DIR
from models.procedure import ProcedureStep
from services.clip_extractor import (
    extract_clips,
    transcode_lens_variant,
    upload_lens_variant_path,
)
from services.database import list_stale_video_procedures, save_procedure
from services.expert_interview import facts as interview_facts
from services.expert_interview import store as interview_store
from services.gemini_processor import process_expert_video
from services.video_probe import probe_duration

logger = logging.getLogger(__name__)


async def process_video_into_procedure(
    procedure_id: str, video_path: Path, actual_duration: float
) -> str:
    """Run the full extraction pipeline for one video and save the result.

    Gemini structured extraction → timestamp clamping → per-step clip extraction
    → DB save (which stamps the current EXTRACTION_PIPELINE_VERSION and replaces
    any existing steps). Returns the final procedure status
    (`completed` | `completed_partial`). Raises on unrecoverable failure — the
    caller owns the error policy (the upload path marks the row failed and
    deletes the source; the re-extract path leaves the old row intact).
    """
    # Interview log (if the phone uploaded one) is read from disk every run —
    # first extraction and re-extraction alike — so the facts survive a
    # pipeline-version bump. If the phone's finalize raced the upload and the
    # log has Q/A but no facts yet, derive them here (non-fatal).
    interview = interview_store.load_log(procedure_id)
    if interview is not None and interview.qa and not interview.facts:
        interview.facts = await interview_facts.extract_facts(interview)
        if interview.facts:
            interview_store.save_log(procedure_id, interview)

    gemini_output = await process_expert_video(
        str(video_path), GEMINI_API_KEY, interview=interview,
    )
    duck_windows = interview_store.duck_windows(interview)

    # Clamp Gemini timestamps to the true video bounds.
    clamped_steps: list[dict] = []
    for s in gemini_output["steps"]:
        raw_start, raw_end = s["timestamp_start"], s["timestamp_end"]
        start = max(0.0, min(raw_start, actual_duration))
        end = max(start, min(raw_end, actual_duration))
        if (start, end) != (raw_start, raw_end):
            logger.warning(
                "Step %d timestamps clamped: (%.2f, %.2f) -> (%.2f, %.2f)",
                s["step_number"], raw_start, raw_end, start, end,
            )
        step_dict = dict(s)
        step_dict["timestamp_start"] = start
        step_dict["timestamp_end"] = end
        clamped_steps.append(step_dict)

    # Extract clips
    try:
        clip_paths, failed_steps = await extract_clips(
            str(video_path), procedure_id, clamped_steps, CLIPS_DIR,
            duck_windows=duck_windows,
        )
    except Exception as e:
        logger.error("Clip extraction failed entirely: %s", e)
        clip_paths = {}
        failed_steps = [s["step_number"] for s in clamped_steps]

    # Pre-generate the bitrate-capped lens variant of the FULL recording so the
    # "Watch full recording" lens button streams smoothly on first play. Non-fatal.
    asyncio.create_task(
        asyncio.to_thread(
            transcode_lens_variant, video_path, upload_lens_variant_path(video_path),
            duck_windows,
        )
    )

    # Build step data with clip URLs + per-step status
    failed_set = set(failed_steps)
    steps: list[ProcedureStep] = []
    for s in clamped_steps:
        step_num = s["step_number"]
        if step_num in failed_set:
            clip_url = None
            clip_status = "failed"
        else:
            clip_url = f"/api/clips/{procedure_id}/step_{step_num}.mp4"
            clip_status = "ready"
        steps.append(ProcedureStep(
            step_number=step_num,
            title=s["title"],
            description=s["description"],
            display_summary=s.get("display_summary", ""),
            timestamp_start=s["timestamp_start"],
            timestamp_end=s["timestamp_end"],
            tips=s.get("tips", []),
            warnings=s.get("warnings", []),
            completion_criteria=s.get("completion_criteria", ""),
            error_criteria=s.get("error_criteria", []),
            visual_verifiability=s.get("visual_verifiability", "reliable"),
            clip_url=clip_url,
            clip_status=clip_status,
        ))

    status = "completed_partial" if failed_steps else "completed"
    procedure_dict = {
        "id": procedure_id,
        "title": gemini_output["title"],
        "description": gemini_output["description"],
        "total_duration": actual_duration,
        "status": status,
        "icon_symbol": gemini_output["icon_symbol"],
        "icon_emoji": gemini_output["icon_emoji"],
        "category": gemini_output["category"],
        "parts": gemini_output.get("parts", []),
    }
    steps_dicts = [s.model_dump() for s in steps]
    save_procedure(procedure_dict, steps_dicts)

    logger.info(
        "Procedure saved: %s (%s, %d steps, status=%s)",
        procedure_id, gemini_output["title"], len(steps), status,
    )
    return status


def _resolve_source_video(procedure_id: str, stored_path: str | None) -> Path | None:
    """Locate the on-disk source video for a procedure.

    Prefer the exact path stored at upload time; fall back to globbing uploads/
    by procedure id (robust if the DB was seeded on another machine and carries a
    foreign absolute path). The `{id}.*` glob can't match the `{id}_lens.mp4`
    variant, but we guard against it explicitly anyway.
    """
    if stored_path:
        p = Path(stored_path)
        if p.exists():
            return p
    for cand in sorted(UPLOAD_DIR.glob(f"{procedure_id}.*")):
        if "_lens" not in cand.name:
            return cand
    return None


async def reextract_stale_procedures(force: bool = False) -> None:
    """Re-extract every video procedure whose extraction_version is stale.

    Sequential on purpose: re-extraction uploads a whole video to the Gemini
    Files API and the deploy box is small — one at a time avoids memory spikes
    and API bursts. A failure on one procedure is logged and skipped (its old row
    is left intact), so a single bad video can't stall the rest.

    `force=True` bypasses the RETRACE_AUTO_REEXTRACT env gate (used by the manual
    CLI in reextract.py). On startup we pass force=False so operators can disable
    the automatic pass with RETRACE_AUTO_REEXTRACT=0.
    """
    if not force and os.getenv("RETRACE_AUTO_REEXTRACT", "1") != "1":
        logger.info("Re-extraction: disabled (RETRACE_AUTO_REEXTRACT=0)")
        return
    if not GEMINI_API_KEY:
        logger.warning("Re-extraction: GEMINI_API_KEY not set, skipping")
        return

    stale = list_stale_video_procedures()
    if not stale:
        logger.info(
            "Re-extraction: no stale video procedures (pipeline v=%s)",
            EXTRACTION_PIPELINE_VERSION,
        )
        return

    logger.info(
        "Re-extraction: %d stale video procedure(s) → target v=%s",
        len(stale), EXTRACTION_PIPELINE_VERSION,
    )
    ok = skipped = failed = 0
    for i, p in enumerate(stale, 1):
        pid = p["id"]
        title = p.get("title") or "(untitled)"
        source = _resolve_source_video(pid, p.get("source_video"))
        if source is None:
            logger.warning(
                "Re-extraction [%d/%d] %s (%s): source video missing on disk, skipping",
                i, len(stale), pid, title,
            )
            skipped += 1
            continue
        try:
            duration = await probe_duration(str(source))
            status = await process_video_into_procedure(pid, source, duration)
            ok += 1
            logger.info(
                "Re-extraction [%d/%d] %s (%s): done (status=%s)",
                i, len(stale), pid, title, status,
            )
        except Exception as exc:
            failed += 1
            logger.exception(
                "Re-extraction [%d/%d] %s (%s): FAILED — old data left intact: %s",
                i, len(stale), pid, title, exc,
            )
    logger.info(
        "Re-extraction complete: %d ok, %d skipped (no source), %d failed",
        ok, skipped, failed,
    )


def schedule_reextract() -> asyncio.Task:
    """Fire-and-forget wrapper for the lifespan (mirrors schedule_fixtures_load).

    Returns the Task so the caller can cancel it on shutdown.
    """
    return asyncio.create_task(reextract_stale_procedures())
