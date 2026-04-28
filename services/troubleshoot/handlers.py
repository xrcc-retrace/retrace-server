"""Tool handlers for troubleshoot-mode Gemini Live function calls.

Each handler:
  - Runs synchronously (matches services.session.handle_tool_call shape).
    Async work (Gemini ranking, PDF ingest) is dispatched via asyncio
    helpers or background tasks; the tool-call response returns a quick
    acknowledgement plus any data already available.
  - Mutates session state via services.troubleshoot.session helpers.
  - Appends a structured entry to the diagnostic log.
  - Returns a dict the phone will forward back to Gemini as a toolResponse.

Registration with the shared dispatcher happens at import time — so
importing the troubleshoot package (or any submodule that imports it)
wires the handlers up.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from services import session as shared_session
from services import procedure_search
from services.database import create_procedure_placeholder, save_procedure
from services.troubleshoot import session as ts_session
from services.troubleshoot import web_research

logger = logging.getLogger(__name__)


def _handle_identify_product(session_id: str, args: dict) -> dict:
    query = (args.get("query") or "").strip()
    category = (args.get("category") or "").strip() or None

    # v1: we don't have a real product KB. Echo back the query with a
    # medium-confidence tag so the model has something to confirm with
    # the user. When a real KB lands, replace this body without changing
    # the tool's public shape.
    product = {
        "product_name": query,
        "category": category,
        "confidence": "medium",
    }
    ts_session.set_identified_product(session_id, product)
    # NOTE: phase stays in DISCOVERING. The session only advances to
    # DIAGNOSING when the user explicitly confirms via the
    # `confirm_identification` tool — see _handle_confirm_identification.

    result = {
        "status": "ok",
        "product": product,
        "message": (
            f"Tentatively identified as '{query}'. "
            "Describe it back to the user and ASK them to confirm. "
            "On affirmative, call confirm_identification. On negative, "
            "call identify_product again with a new guess."
        ),
    }
    ts_session.log_tool_call(session_id, "identify_product", args, result)
    return result


def _handle_confirm_identification(session_id: str, args: dict) -> dict:
    """User explicitly confirmed the identified product. Advance to diagnose."""
    ts_session.transition_phase(session_id, ts_session.PHASE_DIAGNOSING)
    result = {
        "status": "ok",
        "message": (
            "Confirmed. Now ask one observable-symptom question to begin "
            "diagnosing."
        ),
    }
    ts_session.log_tool_call(session_id, "confirm_identification", args, result)
    return result


def _handle_search_procedures(session_id: str, args: dict) -> dict:
    product_name = (args.get("product_name") or "").strip()
    symptom = (args.get("symptom") or "").strip()
    if not product_name or not symptom:
        err = {"status": "error", "error": "product_name and symptom are required"}
        ts_session.log_tool_call(session_id, "search_procedures", args, err)
        return err

    # Run the async Gemini ranker synchronously — we're already inside a
    # FastAPI request, so asyncio.run would explode. Use run_until_complete
    # on a fresh loop to stay sync-compatible with the dispatcher shape.
    candidates = _run_coro(procedure_search.search(product_name, symptom))
    ts_session.set_candidate_procedures(session_id, candidates)
    ts_session.transition_phase(session_id, ts_session.PHASE_RESOLVING)

    if candidates:
        result = {
            "status": "ok",
            "candidates": candidates,
            "message": (
                f"Found {len(candidates)} candidate procedures. "
                "Describe the top one to the user and ask for confirmation."
            ),
        }
    else:
        result = {
            "status": "no_match",
            "candidates": [],
            "message": (
                "No procedures in the library match this product + symptom. "
                "Call web_search_for_fix next to look online."
            ),
        }
    ts_session.log_tool_call(session_id, "search_procedures", args, result)
    return result


def _handle_web_search_for_fix(session_id: str, args: dict) -> dict:
    """Run a grounded Gemini web search, synthesize a procedure, persist it.

    Called by Gemini Live after `search_procedures` returns no_match. The
    tool call blocks for ~10-20s — Gemini Live should give the user a
    short status line ("Looking that up — give me a sec") before the call
    so the silence isn't awkward.
    """
    product_name = (args.get("product_name") or "").strip()
    symptom = (args.get("symptom") or "").strip()
    diagnostic_summary = (args.get("diagnostic_summary") or "").strip()
    if not product_name or not symptom:
        err = {
            "status": "error",
            "error": "product_name and symptom are required",
        }
        ts_session.log_tool_call(session_id, "web_search_for_fix", args, err)
        return err

    try:
        result_obj = _run_coro(
            web_research.research_and_extract(
                product_name=product_name,
                symptom=symptom,
                diagnostic_summary=diagnostic_summary,
            )
        )
    except Exception as exc:
        logger.exception("web_search_for_fix: research failed: %s", exc)
        err = {
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "message": (
                "Couldn't reach the search service. Tell the user we hit a "
                "snag and ask them to try again in a moment."
            ),
        }
        ts_session.log_tool_call(session_id, "web_search_for_fix", args, err)
        return err

    if result_obj is None:
        no_fix = {
            "status": "no_fix_found",
            "message": (
                "I searched the web and couldn't find a matching fix for this. "
                "Tell the user we don't have a procedure for this and suggest "
                "they contact a service tech."
            ),
        }
        ts_session.log_tool_call(session_id, "web_search_for_fix", args, no_fix)
        return no_fix

    # Persist the new procedure.
    procedure_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    # `source_video` is NOT NULL in the schema. Use a self-documenting URI
    # so anyone inspecting the row knows this came from the web flow.
    source_video = f"web-search://{symptom}"
    create_procedure_placeholder(procedure_id, source_video=source_video, created_at=created_at)

    procedure_row = {**result_obj.procedure, "id": procedure_id}
    save_procedure(procedure_row, result_obj.steps)

    sess = shared_session.get_session(session_id)
    if sess is not None:
        sess["generated_procedure_id"] = procedure_id

    sources_count = len(result_obj.procedure.get("sources") or [])
    result = {
        "status": "ok",
        "procedure_id": procedure_id,
        "title": result_obj.procedure.get("title", ""),
        "sources_count": sources_count,
        "message": (
            f"Generated a fix from {sources_count} online source"
            f"{'s' if sources_count != 1 else ''} "
            f"(procedure_id={procedure_id}). Describe it to the user in one "
            "sentence and ask if they want to run through it. On confirmation, "
            "call handoff_to_learner with this procedure_id."
        ),
    }
    ts_session.log_tool_call(session_id, "web_search_for_fix", args, result)
    return result


def _handle_handoff_to_learner(session_id: str, args: dict) -> dict:
    procedure_id = (args.get("procedure_id") or "").strip()
    summary = (args.get("summary") or "").strip()
    if not procedure_id:
        err = {"status": "error", "error": "procedure_id required"}
        ts_session.log_tool_call(session_id, "handoff_to_learner", args, err)
        return err

    # Record the intent. Actual socket teardown + learner session mint is
    # triggered by the phone calling POST /api/troubleshoot/session/{id}/handoff,
    # so this tool response stays fast.
    ts_session.set_handoff_target(session_id, procedure_id, summary)

    result = {
        "status": "ok",
        "procedure_id": procedure_id,
        "message": (
            "Handoff acknowledged. The phone will transition to Learner coaching "
            "on this procedure. Give a brief closing line and stop talking."
        ),
    }
    ts_session.log_tool_call(session_id, "handoff_to_learner", args, result)
    return result


# ---- Helpers ----------------------------------------------------------------


def _run_coro(coro):
    """Run an async coroutine from a sync context.

    The tool-call dispatcher is called from a FastAPI async handler, so the
    event loop is already running. In that case, wrap the coroutine in a
    thread. Otherwise, spin up a fresh loop.
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

shared_session.register_tool_handler("troubleshoot", "identify_product", _handle_identify_product)
shared_session.register_tool_handler("troubleshoot", "confirm_identification", _handle_confirm_identification)
shared_session.register_tool_handler("troubleshoot", "search_procedures", _handle_search_procedures)
shared_session.register_tool_handler("troubleshoot", "web_search_for_fix", _handle_web_search_for_fix)
shared_session.register_tool_handler("troubleshoot", "handoff_to_learner", _handle_handoff_to_learner)
