"""Startup fixture loader — ensures a baseline manual exists so the
dashboard and demo flow aren't empty on a fresh DB.

Single fixture: Nespresso Vertuo Next descaling guide. The PDF is
shipped checked-in at fixtures/nespresso_vertuo_next_descaling.pdf and
ingested via the shared `services.manual_ingest` pipeline in
`mode="expert"` (end-to-end procedure, not a fault-scoped subset).

If the checked-in PDF is missing for any reason, the loader falls back
to downloading from the official Nespresso URL.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from config import BASE_DIR
from services import manual_ingest
from services.database import list_procedures

logger = logging.getLogger(__name__)

FIXTURES_DIR = BASE_DIR / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

NESPRESSO_FIXTURE = {
    "filename": "nespresso_vertuo_next_descaling.pdf",
    "url": "https://www.nespresso.com/static/us/documents/Vertuo%20Next%20Descaling%20Guide%20FINAL.pdf",
    "product_name": "Nespresso Vertuo Next",
    "title_needle": "nespresso vertuo next",  # matches procedure title
}


async def _ensure_pdf_on_disk(fixture: dict) -> Path | None:
    """Return the PDF path. Prefer the checked-in file; fall back to URL."""
    path = FIXTURES_DIR / fixture["filename"]
    if path.exists() and path.stat().st_size > 0:
        return path
    logger.info("Fixture PDF missing on disk, downloading: %s", fixture["url"])
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
                "Fixture downloaded: %s (%d bytes)", path.name, len(r.content),
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
    """Ingest the Nespresso fixture if it isn't already in the library.

    Runs async in the server's lifespan. Cold start takes 30-90s. The
    server comes up immediately; this task runs in the background so
    /health and API endpoints stay available during ingest.
    """
    fixture = NESPRESSO_FIXTURE
    if _fixture_already_ingested(fixture):
        logger.info("Fixture already ingested: %s", fixture["filename"])
        return
    path = await _ensure_pdf_on_disk(fixture)
    if not path:
        logger.warning("Skipping fixture ingest (PDF unavailable): %s", fixture["filename"])
        return

    manual_id = f"fixture-{uuid.uuid4()}"
    manual_ingest.register_local_manual(
        manual_id=manual_id,
        path=path,
        filename=fixture["filename"],
    )
    logger.info("Ingesting fixture: %s (manual_id=%s)", fixture["filename"], manual_id)
    try:
        procedure_id = await manual_ingest.ingest_manual(
            manual_id=manual_id,
            mode="expert",
            product_name=fixture["product_name"],
        )
        if procedure_id:
            logger.info(
                "Fixture ingested: %s → procedure %s",
                fixture["filename"], procedure_id,
            )
        else:
            logger.warning("Fixture ingest returned None: %s", fixture["filename"])
    except Exception as exc:
        logger.exception("Fixture ingest errored: %s", exc)


def schedule_fixtures_load() -> asyncio.Task:
    """Fire-and-forget wrapper for the lifespan. Returns the Task so the
    caller can cancel if the server shuts down mid-ingest.
    """
    return asyncio.create_task(ensure_fixtures_loaded())
