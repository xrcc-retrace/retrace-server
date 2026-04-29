"""Gemini-ranked search over the existing procedure library.

Used by the troubleshoot mode (and potentially any future mode) to find
procedures matching a product + symptom without maintaining a dedicated
full-text index. Relies on Gemini 2.5 Flash to rank by semantic match —
good enough for the hackathon library size (~tens of procedures).

Configured per Gemini official guidance: the per-field shape rules live
in the response_schema as Field descriptions (so the model reads them
through structured-output decoding rather than the prompt), and Flash
runs with thinking_budget=0 since this is a string-matching/ranking
task that doesn't benefit from extended reasoning.
"""

import asyncio
import json
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY
from services.database import get_procedure, list_procedures

logger = logging.getLogger(__name__)

RANKING_MODEL = "gemini-2.5-flash"
RANKING_TIMEOUT_MS = 30_000


class _RankedCandidate(BaseModel):
    procedure_id: str = Field(
        description=(
            "Verbatim procedure_id from the input candidate list. "
            "Never invent ids that weren't in the input."
        ),
    )
    title: str = Field(
        description="Verbatim title from the input candidate.",
    )
    match_reason: str = Field(
        description=(
            "One-sentence plain-language reason this procedure addresses "
            "the symptom."
        ),
    )
    confidence: str = Field(
        description=(
            '"high", "medium", or "low" — how confident you are this is '
            "a real match."
        ),
    )


class _RankingResult(BaseModel):
    matches: list[_RankedCandidate] = Field(
        description=(
            "Up to 3 candidates ordered best match → worst. Empty list "
            "if none are a good match. Exclude any candidate whose "
            "relevance is weak."
        ),
    )


_RANKING_PROMPT = """You rank procedures by how well they match a user's broken-equipment symptom.

You will receive:
- A product name (e.g. "Levoit LV600HH humidifier")
- A symptom summary (e.g. "no mist output, fan running, tank full, no cleaning in 6 months")
- A list of candidate procedures with id, title, description, and first-step preview

Rank candidates by how directly they address the symptom for that product. Output shape and constraints (max count, ordering, confidence enum) live in the response schema (_RankingResult) — follow the field descriptions there."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def search(product_name: str, symptom: str) -> list[dict]:
    """Return up to 3 candidate procedures ranked by relevance.

    Each candidate: {procedure_id, title, match_reason, confidence}.
    Empty list means no good match — caller should fall back to
    web_search_for_fix.
    """
    all_procs = list_procedures()
    # Filter to completed procedures only — half-processed ones won't have
    # meaningful titles.
    ready = [p for p in all_procs if p.get("status") in ("completed", "completed_partial")]
    if not ready:
        logger.info("search_procedures: library is empty")
        return []

    # Build the candidate list for Gemini to rank. Include first-step title so
    # the model has more signal than just the top-level description.
    candidates = []
    for p in ready:
        proc = get_procedure(p["id"])
        first_step_title = proc["steps"][0]["title"] if proc and proc.get("steps") else ""
        candidates.append({
            "procedure_id": p["id"],
            "title": p["title"],
            "description": p.get("description", "")[:300],
            "first_step": first_step_title,
            "source_type": p.get("source_type", "video"),
        })

    user_payload = json.dumps({
        "product": product_name,
        "symptom": symptom,
        "candidates": candidates,
    }, ensure_ascii=False)

    try:
        def _call():
            return _get_client().models.generate_content(
                model=RANKING_MODEL,
                contents=[
                    types.Content(parts=[types.Part.from_text(text=user_payload)]),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_RANKING_PROMPT,
                    response_mime_type="application/json",
                    response_schema=_RankingResult,
                    # Ranking against a fixed candidate list is a string-
                    # matching task; thinking adds latency, not quality.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    http_options=types.HttpOptions(timeout=RANKING_TIMEOUT_MS),
                ),
            )
        response = await asyncio.to_thread(_call)
        parsed = _RankingResult.model_validate_json(response.text)

        # Sanity check: filter out any fabricated ids before returning.
        valid_ids = {p["id"] for p in ready}
        matches = [
            m.model_dump() for m in parsed.matches
            if m.procedure_id in valid_ids
        ]
        logger.info(
            "search_procedures('%s', '%s') → %d matches",
            product_name, symptom[:40], len(matches),
        )
        return matches[:3]
    except Exception as exc:
        logger.exception("search_procedures failed: %s", exc)
        return []
