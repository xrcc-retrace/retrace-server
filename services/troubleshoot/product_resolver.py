"""Canonicalize identify_product queries into short product names.

Gemini Live sometimes passes a whole visual description ("hand operated
tool with two long lever handles meeting at a head with a vertical
threaded mandrel...") or a lookalike guess ("power drill") as the
identify_product query. This module resolves that raw query against the
procedure library and always returns a short canonical product name, so
the session never adopts a descriptor sentence as the product's name.

Modeled on services.procedure_search: Flash + structured output +
thinking_budget=0, with a hard timeout and a raw-query fallback so
identification never hard-fails on a Gemini hiccup.
"""

import asyncio
import json
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY
from services.database import list_procedures

logger = logging.getLogger(__name__)

RESOLVER_MODEL = "gemini-2.5-flash"
RESOLVER_TIMEOUT_MS = 15_000


class ResolvedProduct(BaseModel):
    product_name: str = Field(
        description=(
            "Short canonical product name, 2-6 words (e.g. 'Nut Rivet "
            "Tool', 'Levoit LV600HH humidifier'). NEVER a sentence, NEVER "
            "a visual description, NEVER phrases like 'user holding'. If "
            "the query matches a library product, use that product's name "
            "with any procedure verbs stripped ('Nut Rivet Tool Assembly' "
            "-> 'Nut Rivet Tool'). Otherwise distill the query itself to "
            "a short name."
        ),
    )
    matched_library_title: str | None = Field(
        default=None,
        description=(
            "Verbatim library title the query resolved to, or null if no "
            "library product plausibly matches. Never invent titles that "
            "weren't in the input list."
        ),
    )
    confidence: str = Field(
        description=(
            '"high", "medium", or "low" — how confident you are in the '
            "resolution."
        ),
    )


_RESOLVER_PROMPT = """You canonicalize a raw product query from a voice assistant into a short product name.

You will receive:
- A raw query. It may be a clean product name, a model number, a long VISUAL DESCRIPTION of an object ("hand operated tool with two long lever handles..."), or a mistaken lookalike guess (e.g. "power drill" when the real object is a rivet nut tool — visually similar categories get confused).
- An optional category hint.
- The library: a list of known products (title + category).
- recognition_hints: descriptions of equipment that is commonly misidentified, each naming the product it describes.

Resolve the query to a library product when the described features or the guessed category are plausibly compatible with one — prefer a library match over a generic interpretation. When the query is a visual description, match it against the recognition_hints too: a hint that fits the description gives you the product name even if the library has no matching procedure. Output shape and constraints live in the response schema (ResolvedProduct); follow the field descriptions there."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _library_entries() -> list[dict]:
    """Deduped completed-procedure titles + categories, mirroring the
    KNOWN PRODUCTS block in services.troubleshoot.prompt."""
    entries = []
    seen = set()
    for p in list_procedures():
        if p.get("status") not in ("completed", "completed_partial"):
            continue
        title = " ".join((p.get("title") or "").split())
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        entries.append({"title": title, "category": (p.get("category") or "").strip()})
    return entries


def _equipment_hints() -> list[str]:
    """The prompt module's unconditional visual-ID hints. Each names the
    product it describes, so they let the resolver map a descriptor query
    to its product even when the library has no matching procedure."""
    from services.troubleshoot.prompt import _EQUIPMENT_ID_HINTS
    return list(_EQUIPMENT_ID_HINTS)


async def resolve(query: str, category: str | None = None) -> ResolvedProduct:
    """Resolve a raw identify_product query to a short canonical name.

    Never raises: on any failure, falls back to the raw query so the
    identify flow degrades to the old echo behavior instead of erroring.
    """
    fallback = ResolvedProduct(
        product_name=query, matched_library_title=None, confidence="low"
    )
    user_payload = json.dumps({
        "query": query,
        "category_hint": category,
        "library": _library_entries(),
        "recognition_hints": _equipment_hints(),
    }, ensure_ascii=False)

    try:
        def _call():
            return _get_client().models.generate_content(
                model=RESOLVER_MODEL,
                contents=[
                    types.Content(parts=[types.Part.from_text(text=user_payload)]),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_RESOLVER_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ResolvedProduct,
                    # Name extraction against a fixed list — thinking adds
                    # latency inside a live tool call, not quality.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    http_options=types.HttpOptions(timeout=RESOLVER_TIMEOUT_MS),
                ),
            )
        response = await asyncio.to_thread(_call)
        resolved = ResolvedProduct.model_validate_json(response.text)
        if not resolved.product_name.strip():
            logger.warning("product_resolver: empty product_name, falling back to raw query")
            return fallback
        logger.info(
            "product_resolver: %r -> %r (library=%r, confidence=%s)",
            query[:80], resolved.product_name,
            resolved.matched_library_title, resolved.confidence,
        )
        return resolved
    except Exception as exc:
        logger.warning("product_resolver: resolution failed (%s), falling back to raw query", exc)
        return fallback
