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

from services import session as shared_session
from services import procedure_search
from services.troubleshoot import session as ts_session
from services.troubleshoot import manual_ingest

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
    ts_session.transition_phase(session_id, ts_session.PHASE_DIAGNOSING)

    result = {
        "status": "ok",
        "product": product,
        "message": (
            f"Tentatively identified as '{query}'. "
            "Confirm with the user, then diagnose the symptom."
        ),
    }
    ts_session.log_tool_call(session_id, "identify_product", args, result)
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
                "Call fetch_manual next."
            ),
        }
    ts_session.log_tool_call(session_id, "search_procedures", args, result)
    return result


def _handle_fetch_manual(session_id: str, args: dict) -> dict:
    product_name = (args.get("product_name") or "").strip()
    symptom_keywords = args.get("symptom_keywords") or []
    if not product_name:
        err = {"status": "error", "error": "product_name required"}
        ts_session.log_tool_call(session_id, "fetch_manual", args, err)
        return err

    # First, check for a pre-loaded manual-ingested procedure that already
    # covers this product. If found, short-circuit: the model can hand off
    # directly without uploading or regenerating.
    preloaded = manual_ingest.find_preloaded_for_product(product_name)
    if preloaded:
        sess = shared_session.get_session(session_id)
        if sess is not None:
            sess["generated_procedure_id"] = preloaded["procedure_id"]
        result = {
            "status": "preloaded",
            "procedure_id": preloaded["procedure_id"],
            "title": preloaded["title"],
            "description": preloaded["description"],
            "message": (
                "Pre-loaded manual exists for this product and has already been "
                "converted to a procedure. Confirm with the user and call "
                "handoff_to_learner with this procedure_id — skip "
                "generate_sop_from_manual."
            ),
        }
    else:
        # No pre-loaded manual. Signal the phone to prompt the user for an
        # upload. Once uploaded, the phone polls and sends a synthetic
        # follow-up to Gemini so it can proceed to generate_sop_from_manual.
        result = {
            "status": "user_upload_required",
            "upload_url": "/api/troubleshoot/manuals/upload",
            "symptom_keywords": symptom_keywords,
            "message": (
                "No manual on file for this product. Tell the user: "
                "'I need the manual — there should be a prompt on your "
                "phone to upload the PDF.' Wait for the phone to send a "
                "'manual is ready' signal before calling generate_sop_from_manual."
            ),
        }

    ts_session.log_tool_call(session_id, "fetch_manual", args, result)
    return result


def _handle_generate_sop_from_manual(session_id: str, args: dict) -> dict:
    manual_id = (args.get("manual_id") or "").strip()
    focus_problem = (args.get("focus_problem") or "").strip()
    diagnostic_summary = (args.get("diagnostic_summary") or "").strip()
    if not manual_id or not focus_problem:
        err = {"status": "error", "error": "manual_id and focus_problem required"}
        ts_session.log_tool_call(session_id, "generate_sop_from_manual", args, err)
        return err

    # Resolve product_name from the session's identified product.
    sess = shared_session.get_session(session_id)
    product_name = "Unknown product"
    if sess:
        prod = sess.get("identified_product") or {}
        product_name = prod.get("product_name") or product_name

    # Ingestion takes 20-60s. Return early with a "processing" response so
    # Gemini can keep the user company while we work; the phone polls
    # /api/troubleshoot/manuals/{id} for completion.
    import asyncio as _asyncio
    loop = None
    try:
        loop = _asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    async def _run_ingest():
        return await manual_ingest.ingest_manual(
            manual_id=manual_id,
            product_name=product_name,
            focus_problem=focus_problem,
            diagnostic_summary=diagnostic_summary,
        )

    # Run synchronously — Gemini Live tool calls block until we respond;
    # we need the resulting procedure_id to hand back.
    procedure_id = _run_coro(_run_ingest())
    if not procedure_id:
        err = {
            "status": "failed",
            "error": "SOP generation failed. See server logs.",
        }
        ts_session.log_tool_call(session_id, "generate_sop_from_manual", args, err)
        return err

    if sess is not None:
        sess["generated_procedure_id"] = procedure_id

    result = {
        "status": "ok",
        "procedure_id": procedure_id,
        "message": (
            f"Generated a focused repair procedure (procedure_id={procedure_id}). "
            "Describe it to the user in one sentence and ask if they want to run through it. "
            "On confirmation, call handoff_to_learner with this procedure_id."
        ),
    }
    ts_session.log_tool_call(session_id, "generate_sop_from_manual", args, result)
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
shared_session.register_tool_handler("troubleshoot", "search_procedures", _handle_search_procedures)
shared_session.register_tool_handler("troubleshoot", "fetch_manual", _handle_fetch_manual)
shared_session.register_tool_handler("troubleshoot", "generate_sop_from_manual", _handle_generate_sop_from_manual)
shared_session.register_tool_handler("troubleshoot", "handoff_to_learner", _handle_handoff_to_learner)
