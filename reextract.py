"""On-demand: re-extract every video procedure whose extraction_version is stale.

Same work the server runs automatically on startup after you bump
config.EXTRACTION_PIPELINE_VERSION (see services/reprocess.py) — this is the
manual escape hatch when you want to repopulate without a redeploy/restart.

Re-runs Gemini 2.5 Pro extraction + clip extraction over each procedure's
original source video in uploads/ and OVERWRITES the row (any manual dashboard
edits to those procedures are lost). Only source_type='video' procedures are
touched; manual/web procedures are left alone.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python reextract.py

Note: if the production server is running against the same data/retrace.db, run
this on the box (or accept that both processes write the same SQLite file — WAL
tolerates it, but prefer restarting the server, which does the same pass).
"""

import asyncio
import logging
import sys

from services.reprocess import reextract_stale_procedures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


if __name__ == "__main__":
    # force=True bypasses the RETRACE_AUTO_REEXTRACT gate — running this script is
    # itself the explicit opt-in.
    asyncio.run(reextract_stale_procedures(force=True))
    sys.exit(0)
