"""Tool handlers for Ask-AI-mode Gemini Live function calls.

Just one tool: search_procedures(query). It ranks the existing library via the
shared procedure_search service and stores the candidates on the session so the
iOS client can render them on the lens. Selection + coaching handoff is driven
by the lens tap (POST /api/ask_ai/session/{id}/handoff), not by a tool call.

Registration with the shared dispatcher happens at import time, so importing
the ask_ai package wires the handler up.
"""

import asyncio
import logging

from services import procedure_search
from services import session as shared_session
from services.ask_ai import session as ask_session

logger = logging.getLogger(__name__)


def _handle_search_procedures(session_id: str, args: dict) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"status": "error", "error": "query is required"}

    # The library ranker takes a product + symptom; Ask AI has a single
    # free-text description, so pass it as both — the ranker matches on
    # semantic overlap with each procedure's title/description either way.
    candidates = _run_coro(procedure_search.search(query, query))
    ask_session.set_candidates(session_id, candidates)

    if candidates:
        return {
            "status": "ok",
            "candidates": candidates,
            "message": (
                f"Found {len(candidates)} matching procedure"
                f"{'s' if len(candidates) != 1 else ''}. The learner sees them "
                "on their display. Name the top one in one sentence and tell "
                "them to pick the one they want."
            ),
        }
    return {
        "status": "no_match",
        "candidates": [],
        "message": (
            "No procedures in the library match that. Tell the learner you "
            "don't have a guide for it yet and offer to try a different "
            "description."
        ),
    }


# ---- Helpers ----------------------------------------------------------------


def _run_coro(coro):
    """Run an async coroutine from a sync context (the dispatcher runs inside a
    FastAPI async handler, so the loop is already running). Mirrors
    services.troubleshoot.handlers._run_coro.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---- Registration -----------------------------------------------------------

shared_session.register_tool_handler("ask_ai", "search_procedures", _handle_search_procedures)
