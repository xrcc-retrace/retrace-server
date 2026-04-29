"""Two-step Gemini pipeline that turns a product+symptom into a structured
repair procedure with cited online sources.

Used by the troubleshoot-mode `web_search_for_fix` tool when
`search_procedures` returns no_match. Replaces the old
`fetch_manual` + `generate_sop_from_manual` flow.

Design notes
------------
- `google_search` and `response_schema` are mutually exclusive on the
  v1beta `generateContent` endpoint, so we run two calls:
  * Step 1: Gemini Flash + GoogleSearch tool → free-form research brief +
    grounding metadata (URIs of cited pages). One-shot search bias —
    issue one strong query, stop as soon as a high-signal community
    thread (Reddit/iFixit) names the cause and fix.
  * Step 2: Gemini Pro + structured output → final procedure JSON, with
    the source URLs from step 1 baked verbatim into the prompt so the
    model echoes them back into the `sources` array.
- Step 1 uses Flash + `thinking_budget=0` for latency. Step 2 stays on
  Pro because step 2 has no web access — it must reconstruct the actual
  repair steps from the brief alone, so synthesis quality matters.
- Each leg has the same Pro→Flash 503 fallback pattern as
  `services.gemini_processor` (step 1's "fallback" is a no-op since it
  already runs on Flash, but the structure stays uniform).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from google import genai
from google.genai import types
from google.genai.errors import ServerError

from config import GEMINI_API_KEY
from models.procedure import GeminiWebProcedureOutput
from services.metadata_picker import pick_metadata


# Optional progress-signal callback. The caller (handler) wires this
# into the troubleshoot session so iOS can poll /state and watch the
# phase advance from `searching_web` → `synthesizing` → `complete`.
# Terminal phases (`complete` / `no_fix_found` / `error`) are emitted
# by the handler, not this module — `research_and_extract` doesn't
# know about persistence or session lifetime.
PhaseCallback = Callable[[str, dict], None]

logger = logging.getLogger(__name__)

# Step 1: Flash + zero thinking budget for fast grounded search. The
# brief is mostly summarization of pages Google retrieved; Flash handles
# that fine and the prompt enforces a one-shot search loop with strong
# Reddit/iFixit bias.
STEP1_MODEL = "gemini-2.5-flash"
# Step 2 stays on Pro: it has NO web access (google_search and
# response_schema are mutually exclusive) so it must reconstruct the
# actual repair steps from step 1's brief alone. Flash drops mechanical
# detail here (wrong screw count, missing safety step). Pro is the
# fallback only on transient server errors (503/504/429).
STEP2_MODEL = "gemini-2.5-pro"
FALLBACK_MODEL = "gemini-2.5-flash"
# 60s per leg. Flash one-shot grounded search lands ~6-12s; Pro
# structured synthesis ~10-15s; the budget keeps both within Google's
# own deadline so we don't tail into 504 during normal load.
STEP_TIMEOUT_MS = 60_000
# HTTP codes we treat as "transient — retry on Flash". 503 is the
# documented overload signal; 504 fires on backend deadline; 429 is rate
# limit. All three are recoverable with a different model/route.
_TRANSIENT_CODES = {429, 503, 504}
MAX_SOURCES = 8

# Gemini 2.5 Pro can't disable thinking (floor is 128 tokens; cannot be
# set to 0 like Flash can). Default is dynamic (-1), which lets Pro
# spend up to ~8192 thinking tokens — that's most of the 30+ second
# wall-clock on step 2. Pinning to the documented floor (128) is the
# biggest available latency lever for this model. Bump up if synthesis
# quality regresses (mechanical detail dropped, steps too generic).
STEP2_THINKING_BUDGET = 128
# Synthesis is reconstruction from a fixed prose brief — there's no
# value in sampling diverse continuations. Determinism stops Pro from
# exploring low-probability branches and tightens generation.
STEP2_TEMPERATURE = 0.0

# Bias the metadata picker toward the two categories web fixes almost
# always land in. Falls back to "Other" if Flash chooses something
# clearly off-domain (very rare in practice).
_WEB_CATEGORY_HINT = (
    "Web-search-derived fix procedures almost always land in 'Repair' "
    "(unplanned/corrective fix) or 'Maintenance' (scheduled upkeep — "
    "cleaning, descaling, replacing consumables). Pick one of those "
    "unless the procedure is clearly something else."
)


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# --- Prompts -----------------------------------------------------------------


_STEP1_SYSTEM = """You are a repair-research assistant. Your ONLY job is to search the live web and report what real people have written about fixing the user's problem. You have NO domain knowledge of your own to draw on — every factual claim you make MUST come from a page you actually retrieved via the google_search tool in this turn.

REQUIRED — Tool use, not memory:
- You MUST call the google_search tool. Do NOT answer from prior knowledge. If you have not searched, say so and stop.
- ONE-SHOT SEARCH: issue ONE strong query first. As soon as that query returns a high-signal Reddit / iFixit / community-forum thread that names the likely cause AND the fix, STOP SEARCHING and write the brief. Do not chain follow-up queries for completeness — speed matters more than coverage.
- Only issue a second query if the first returns no usable hits at all. Cap at two queries total.
- Strong first-query shape: `"<product model> <symptom> reddit OR ifixit"`. If that draws a blank, try `"<product model> <symptom> fix forum"`.

SOURCE BIAS — apply STRICTLY in this order:
1. STRONGLY PREFER community sources — Reddit threads (r/fixit, product-specific subs), iFixit guides, repair forums. Real owners, real fixes, real failure modes. These are your default authority for the cause and the hands-on repair steps.
2. Manufacturer / OEM documentation is a LAST RESORT for the actual fix. Manufacturer manuals tell you how to use the product, not how to repair it. Cite OEM ONLY when the step involves AC mains wiring, chemical handling, refrigerant, or pressure-vessel release — and even then, only for those specific safety-critical steps. Everything else: prefer the community.
3. Treat blogspam, SEO listicles, and AI-content-farm sites as last resorts.

OUTPUT FORMAT — plain prose, no markdown, no JSON, ~250-400 words:
- Open with one short paragraph naming the most likely cause and the diagnostic check that confirms it.
- Then a clear sequence of the repair steps as the source threads describe them — concrete actions, in order, with the mechanical detail your sources actually provide (which part to remove, which direction, what to look for, how long to soak/wait, etc.). This is what the next stage will turn into a step-by-step procedure for a first-time learner, so include the substantive how-to.
- Cite each source inline by URL the first time you draw on it.
- If the live web search returns nothing useful for this exact problem, say so explicitly in one sentence — do not pad with general advice."""


_STEP1_USER = """Product: {product_name}
Symptom: {symptom}

Diagnostic context (already established with the user):
{diagnostic_summary}

Search the web and produce the brief."""


_STEP2_SYSTEM = """You are a procedure analyst. Given a research brief about how to fix a problem, synthesize a step-by-step repair procedure that an AI coach can teach a first-time learner.

Guidelines:
- Each step must be a discrete, actionable unit a first-time learner can perform safely.
- Field-level shape requirements live in the response schema (GeminiWebProcedureOutput) — follow the field descriptions there for title, description, tips, warnings, completion_criteria, and the step cap.

Output JSON matching the GeminiWebProcedureOutput schema."""


_STEP2_USER = """Product: {product_name}
Symptom: {symptom}

Diagnostic context already collected with the user:
{diagnostic_summary}

Research brief from the web:
{research_text}

Output the procedure JSON."""


# --- Public surface ----------------------------------------------------------


@dataclass
class ResearchResult:
    """Output of `research_and_extract`. Caller builds a Procedure row and
    saves it via `services.database.save_procedure`."""
    procedure: dict
    """All Procedure-level fields ready for save_procedure(): title,
    description, total_duration, status, source_type, icon_symbol,
    icon_emoji, category, sources."""
    steps: list[dict]
    """Step rows ready for save_procedure()."""


async def research_and_extract(
    product_name: str,
    symptom: str,
    diagnostic_summary: str,
    on_phase: PhaseCallback | None = None,
) -> ResearchResult | None:
    """Two-step Gemini pipeline. Returns None when search yielded nothing
    or the synthesis came back empty — caller maps to `no_fix_found`.
    Raises on hard transport/SDK failures so the handler can surface
    `error` rather than `no_fix_found`.

    `on_phase`, when supplied, is invoked at progress-signal boundaries
    so the caller can mutate session state for iOS to poll. Phases
    emitted by this function:
      - `searching_web` — fired immediately, before step 1 starts.
      - `synthesizing` — fired after step 1 returns sources, before
        step 2 + pick_metadata start. Carries `source_count`.
    Terminal phases (`complete`, `no_fix_found`, `error`) are the
    caller's job — only the handler knows about persistence.
    """
    def _emit(phase: str, **extra) -> None:
        if on_phase is None:
            return
        try:
            on_phase(phase, extra)
        except Exception as exc:
            # Progress signaling is best-effort — don't fail the
            # pipeline because the callback raised.
            logger.warning("web_research: on_phase callback failed: %s", exc)

    client = _get_client()

    # Phase 1 — searching the web. Emit BEFORE the network call so iOS
    # sees the state flip the moment the handler kicks off.
    _emit("searching_web")

    # Step 1: grounded research
    step1 = await _step1_research(client, product_name, symptom, diagnostic_summary)
    research_text = (step1.text or "").strip() if step1 else ""
    sources = _extract_grounding_sources(step1)
    logger.info(
        "web_research step1: brief=%d chars, sources=%d for product=%r symptom=%r",
        len(research_text), len(sources), product_name, symptom,
    )
    if not sources:
        # Either grounding metadata was empty (model answered from memory
        # without searching) or search returned nothing actionable. Treat
        # as "no fix found" rather than a hard error.
        return None
    if not research_text:
        logger.info("web_research: empty research brief; treating as no_fix_found")
        return None

    # Phase 2 — synthesizing. Now we know the source count, hand it off
    # to iOS so the UI can render "Found N sources, writing the
    # procedure…".
    _emit("synthesizing", source_count=len(sources))

    # Step 2 + pick_metadata in parallel. Wall-clock = max(step2, pick).
    # Step 2 dominates at ~14 s on Pro; pick_metadata lands in 1-2 s and
    # hides behind it.
    parsed, metadata = await asyncio.gather(
        _step2_synthesize(
            client,
            product_name=product_name,
            symptom=symptom,
            diagnostic_summary=diagnostic_summary,
            research_text=research_text,
        ),
        pick_metadata(
            context=f"Product: {product_name}\nSymptom: {symptom}",
            category_hint=_WEB_CATEGORY_HINT,
            default_category="Repair",
        ),
    )
    if parsed is None or not parsed.steps:
        logger.info(
            "web_research: synthesis returned %s steps; treating as no_fix_found",
            len(parsed.steps) if parsed else "<None>",
        )
        return None

    # Build the dicts save_procedure expects. The handler attaches id +
    # source_video + created_at and persists. Metadata fields come from
    # pick_metadata (Flash); the synthesizer is told to emit defaults.
    procedure_dict = {
        "title": parsed.title,
        "description": parsed.description,
        # Web procedures aren't backed by a video — total_duration is
        # meaningless. Hardcode 0.0; the new schema doesn't carry this
        # field at all (that's part of the slim-down).
        "total_duration": 0.0,
        "status": "completed",
        "source_type": "web",
        "icon_symbol": metadata["icon_symbol"],
        "category": metadata["category"],
        # Step 1's grounding metadata already gives us
        # {title=hostname, url, snippet=""} — that's all iOS needs to
        # render a clickable source link. No descriptive snippet
        # generation: the URL is the contract.
        "sources": sources,
    }
    step_rows = []
    for idx, s in enumerate(parsed.steps[:15], start=1):
        step_rows.append({
            "step_number": s.step_number,
            "title": s.title,
            "description": s.description,
            # Web procedures aren't backed by a video — synthesize
            # placeholder timestamps server-side so Pro doesn't have to
            # spend any output tokens on numbers nobody reads.
            "timestamp_start": (idx - 1) * 30.0,
            "timestamp_end": idx * 30.0,
            "tips": s.tips,
            "warnings": s.warnings,
            "completion_criteria": s.completion_criteria,
            # error_criteria are heartbeat-path inputs that need a real
            # source frame to anchor against. Web procedures have none,
            # so we emit an empty list and the heartbeat path no-ops.
            "error_criteria": [],
            # No clip and no PDF page; learner endpoints render text-only.
            "clip_url": None,
            "clip_status": "none",
        })

    return ResearchResult(procedure=procedure_dict, steps=step_rows)


# --- Step 1: grounded research ----------------------------------------------


async def _step1_research(
    client: genai.Client,
    product_name: str,
    symptom: str,
    diagnostic_summary: str,
):
    """Fire Gemini Pro with the GoogleSearch tool and return the raw
    GenerateContentResponse so the caller can read both `.text` and the
    grounding metadata. Pro 503 falls back to Flash."""
    user_text = _STEP1_USER.format(
        product_name=product_name,
        symptom=symptom,
        diagnostic_summary=diagnostic_summary or "(no diagnostic summary provided)",
    )
    contents = [types.Content(parts=[types.Part.from_text(text=user_text)])]

    def _generate(model: str):
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_STEP1_SYSTEM,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                # Disable thinking on step 1 — Flash + grounded search is
                # already doing retrieval; the extra reasoning budget mostly
                # buys latency, not quality. ~5-8s saved per call.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=STEP_TIMEOUT_MS),
            ),
        )

    try:
        return await asyncio.to_thread(_generate, STEP1_MODEL)
    except ServerError as e:
        if e.code not in _TRANSIENT_CODES:
            raise
        logger.warning(
            "web_research step1: %s returned %s, falling back to %s",
            STEP1_MODEL, e.code, FALLBACK_MODEL,
        )
        return await asyncio.to_thread(_generate, FALLBACK_MODEL)


def _extract_grounding_sources(response) -> list[dict]:
    """Pull (title, url) from `groundingMetadata.groundingChunks[].web` and
    return a deduped list capped at MAX_SOURCES. Empty list when the
    response carried no grounding info."""
    if response is None:
        return []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    gm = getattr(candidates[0], "grounding_metadata", None)
    chunks = getattr(gm, "grounding_chunks", None) or []

    seen: set[str] = set()
    out: list[dict] = []
    for ch in chunks:
        web = getattr(ch, "web", None)
        if not web:
            continue
        url = (getattr(web, "uri", None) or "").strip()
        title = (getattr(web, "title", None) or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": title or url, "url": url, "snippet": ""})
        if len(out) >= MAX_SOURCES:
            break
    return out


# --- Step 2: structured synthesis -------------------------------------------


async def _step2_synthesize(
    client: genai.Client,
    *,
    product_name: str,
    symptom: str,
    diagnostic_summary: str,
    research_text: str,
) -> GeminiWebProcedureOutput | None:
    user_text = _STEP2_USER.format(
        product_name=product_name,
        symptom=symptom,
        diagnostic_summary=diagnostic_summary or "(no diagnostic summary provided)",
        research_text=research_text,
    )
    contents = [types.Content(parts=[types.Part.from_text(text=user_text)])]

    def _generate(model: str):
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_STEP2_SYSTEM,
                response_mime_type="application/json",
                response_schema=GeminiWebProcedureOutput,
                # See STEP2_THINKING_BUDGET / STEP2_TEMPERATURE comments
                # at the top of the module — the headline latency knobs
                # for 2.5 Pro structured synthesis.
                thinking_config=types.ThinkingConfig(
                    thinking_budget=STEP2_THINKING_BUDGET,
                ),
                temperature=STEP2_TEMPERATURE,
                http_options=types.HttpOptions(timeout=STEP_TIMEOUT_MS),
            ),
        )

    try:
        response = await asyncio.to_thread(_generate, STEP2_MODEL)
    except ServerError as e:
        if e.code not in _TRANSIENT_CODES:
            raise
        logger.warning(
            "web_research step2: %s returned %s, falling back to %s",
            STEP2_MODEL, e.code, FALLBACK_MODEL,
        )
        response = await asyncio.to_thread(_generate, FALLBACK_MODEL)

    text = getattr(response, "text", None)
    if not text:
        logger.warning("web_research step2: empty response")
        return None
    try:
        return GeminiWebProcedureOutput.model_validate_json(text)
    except Exception as exc:
        logger.exception("web_research step2: failed to parse procedure JSON: %s", exc)
        return None
