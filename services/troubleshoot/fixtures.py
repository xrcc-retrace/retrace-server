"""Startup fixture loader — ensures a baseline set of pre-ingested manuals
exists so the dashboard and demo flow aren't empty.

v1: Levoit LV600HH humidifier manual. Downloaded once from the VeSync CDN
to fixtures/levoit_lv600hh_manual.pdf, then ingested into the procedure
library with source_type='manual' so fetch_manual can short-circuit to it.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from config import BASE_DIR
from services.database import list_procedures
from services.troubleshoot import manual_ingest

logger = logging.getLogger(__name__)

FIXTURES_DIR = BASE_DIR / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

LEVOIT_FIXTURE = {
    "filename": "levoit_lv600hh_manual.pdf",
    "url": "https://files.vesync.com/desc/04.00_ML_LV600HH-Shared_2021-11-09_US.pdf",
    "product_name": "Levoit LV600HH",
    "title_needle": "levoit lv600hh",  # matches on procedure title via find_preloaded_for_product
    "focus_problem": "humidifier is not producing mist despite a full tank and running fan",
    "diagnostic_summary": (
        "Common Levoit LV600HH no-mist scenarios: mineral buildup on the "
        "ultrasonic transducer after weeks without descaling, a stuck or "
        "clogged float valve preventing water flow into the atomizing "
        "chamber, or a water-level sensor that isn't triggering. Procedure "
        "should cover descaling the transducer with vinegar, clearing the "
        "float valve, and verifying water reaches the chamber."
    ),
}


async def _download_if_missing(fixture: dict) -> Path | None:
    """Ensure the fixture PDF is on disk. Returns path or None on failure."""
    path = FIXTURES_DIR / fixture["filename"]
    if path.exists() and path.stat().st_size > 0:
        return path
    logger.info("Downloading fixture PDF: %s", fixture["url"])
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
            r = await c.get(fixture["url"])
            if r.status_code != 200:
                logger.warning(
                    "Fixture download failed (%s): status=%d",
                    fixture["filename"], r.status_code,
                )
                return None
            path.write_bytes(r.content)
            logger.info(
                "Fixture downloaded: %s (%d bytes)",
                path.name, len(r.content),
            )
            return path
    except Exception as exc:
        logger.exception("Fixture download errored: %s", exc)
        return None


def _fixture_already_ingested(fixture: dict) -> bool:
    needle = fixture["title_needle"]
    for p in list_procedures():
        if p.get("source_type") != "manual":
            continue
        if needle in (p.get("title") or "").lower():
            return True
    return False


async def ensure_fixtures_loaded() -> None:
    """Download + ingest any fixture PDFs that aren't already in the library.

    Runs async in the server's lifespan. On cold start (empty DB) this
    takes 30-60s. The server comes up immediately; fixtures ingest in the
    background as an asyncio task so /health and API endpoints are
    available while we wait.
    """
    fixtures = [LEVOIT_FIXTURE]
    for f in fixtures:
        if _fixture_already_ingested(f):
            logger.info("Fixture already ingested: %s", f["filename"])
            continue
        path = await _download_if_missing(f)
        if not path:
            logger.warning("Skipping fixture ingest (download failed): %s", f["filename"])
            continue

        manual_id = f"fixture-{uuid.uuid4()}"
        manual_ingest.register_local_manual(
            manual_id=manual_id,
            path=path,
            filename=f["filename"],
        )
        logger.info("Ingesting fixture: %s (manual_id=%s)", f["filename"], manual_id)
        try:
            procedure_id = await manual_ingest.ingest_manual(
                manual_id=manual_id,
                product_name=f["product_name"],
                focus_problem=f["focus_problem"],
                diagnostic_summary=f["diagnostic_summary"],
            )
            if procedure_id:
                logger.info(
                    "Fixture ingested: %s → procedure %s",
                    f["filename"], procedure_id,
                )
            else:
                logger.warning("Fixture ingest returned None: %s", f["filename"])
        except Exception as exc:
            logger.exception("Fixture ingest errored: %s", exc)


def schedule_fixtures_load() -> asyncio.Task:
    """Fire-and-forget wrapper for the lifespan. Returns the Task so the
    caller can cancel if the server shuts down mid-ingest.
    """
    return asyncio.create_task(ensure_fixtures_loaded())
