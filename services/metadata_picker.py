"""Pick top-level procedure metadata (icon + category) via a fast Flash call.

Used by every procedure-generation flow (expert video, expert PDF,
troubleshoot PDF, troubleshoot web search). The heavy synthesis prompts
on Pro stay focused on per-step content; this helper handles the
metadata that doesn't need Pro's reasoning depth — picking an Iconify
icon ID and a category from a closed enum.

Runs on `gemini-2.5-flash` with `thinking_budget=0` and a tiny
structured-output schema. Typical latency ~1-2 s. Safe to call in
`asyncio.gather` alongside the synthesis call when the inputs are known
up-front (web flow has product_name + symptom from the user). Otherwise
call sequentially after the synthesis returns (video / PDF flows need
the parsed title + description to give Flash enough context).

Scope note: this picks icon_symbol + category only. icon_emoji is left
to the schema default ("") — the server's Iconify-PNG path
(`routers/icons.py`) is the canonical icon source and the emoji
fallback never fires in production.

Failure policy: on any exception, off-list category, or empty response,
log + return the supplied defaults. Never raises. The procedure save
path must not fail because of an icon decision.
"""

import asyncio
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import GEMINI_API_KEY
from models.procedure import CATEGORY_NAMES, ProcedureCategory

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
TIMEOUT_MS = 15_000


class _MetadataOutput(BaseModel):
    icon_symbol: str
    category: ProcedureCategory


_SYSTEM_PROMPT = """You pick top-level metadata for a hands-on procedure.

Given the procedure context below, produce two fields:

- icon_symbol: a single Iconify icon ID for the MAIN PHYSICAL OBJECT the procedure works on (the appliance / device / tool — NOT the action). Use the lucide pack: "lucide:<name>". Examples: "lucide:refrigerator", "lucide:humidifier", "lucide:speaker", "lucide:laptop", "lucide:microwave", "lucide:washing-machine", "lucide:wrench". Lucide line-style monochrome — the icon renders white on a dark surface.

- category: exactly one of "Assembly", "Electrical", "Calibration", "Maintenance", "Repair", "Inspection", "Other".
  - Assembly: joining parts into a working unit for the first time.
  - Electrical: wiring, terminating, configuring electrical/electronic systems.
  - Calibration: setting/adjusting/verifying a machine meets a specified output.
  - Maintenance: scheduled upkeep — cleaning, lubricating, descaling, replacing consumables.
  - Repair: diagnosing and fixing a system that has failed or degraded.
  - Inspection: verifying conformance without making any change.
  - Other: only when nothing else fits.

Precedence for ambiguous cases: scheduled/preventive change → Maintenance; unplanned/corrective change → Repair; adjustment to spec → Calibration; check-only → Inspection.

Output JSON matching the schema."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def pick_metadata(
    *,
    context: str,
    category_hint: str = "",
    default_icon_symbol: str = "lucide:wrench",
    default_category: str = "Other",
) -> dict[str, str]:
    """Pick (icon_symbol, category) for a procedure.

    `context` is a free-form blob describing the procedure — title,
    description, product name, symptom, whatever the caller has on hand.
    `category_hint` is appended verbatim to the system prompt to bias
    selection (e.g. web flow uses it to nudge toward Repair/Maintenance).

    Returns a dict with two string keys. On any failure, returns the
    supplied defaults — never raises.
    """
    if default_category not in CATEGORY_NAMES:
        default_category = "Other"

    user_text = f"Procedure context:\n{context}".strip()
    system_prompt = _SYSTEM_PROMPT
    if category_hint:
        system_prompt = f"{_SYSTEM_PROMPT}\n\n{category_hint}"

    def _call():
        return _get_client().models.generate_content(
            model=MODEL,
            contents=[types.Content(parts=[types.Part.from_text(text=user_text)])],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=_MetadataOutput,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )

    defaults = {"icon_symbol": default_icon_symbol, "category": default_category}

    try:
        response = await asyncio.to_thread(_call)
    except Exception as exc:
        logger.warning("pick_metadata: call failed (%s); using defaults", exc)
        return defaults

    text = getattr(response, "text", None)
    if not text:
        logger.warning("pick_metadata: empty response; using defaults")
        return defaults

    try:
        parsed = _MetadataOutput.model_validate_json(text)
    except Exception as exc:
        logger.warning("pick_metadata: parse failed (%s); using defaults. raw=%r", exc, text[:200])
        return defaults

    icon_symbol = parsed.icon_symbol.strip() or default_icon_symbol
    category = parsed.category if parsed.category in CATEGORY_NAMES else default_category

    logger.info("pick_metadata: icon=%s category=%s", icon_symbol, category)
    return {"icon_symbol": icon_symbol, "category": category}
