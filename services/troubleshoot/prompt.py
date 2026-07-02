"""Diagnostic-mode system prompt + tool definitions for Gemini Live.

Parallel to services.prompt_builder (learner). Kept in its own module so the
troubleshoot flow can evolve without touching the learner prompt.
"""

import logging

from config import RETRACE_LIVE_MODEL
from services import database
from services import voice_commands as vc

logger = logging.getLogger(__name__)


# Standalone visual-identification hints for equipment that camera-based
# identification reliably confuses with a more common lookalike. These are
# UNCONDITIONAL — injected into the VISUAL CONTEXT section on every session,
# independent of what procedures happen to be in the library. (They used to be
# gated on a procedure-title substring match, which silently never fired when
# no matching procedure existed — so the hint was dead. Keep them library-
# independent so demo tools identify correctly with an empty/mismatched DB.)
_EQUIPMENT_ID_HINTS = [
    (
        "Rivet nut tool (rivnut/nutsert setter): a hand-operated tool with "
        "two long lever handles like bolt cutters or heavy pliers, meeting at "
        "a compact head with a vertical threaded mandrel and interchangeable "
        "nosepiece at the tip; squeezed with both hands. It has NO motor, NO "
        "trigger, NO chuck, and NO battery pack — if you think you see a power "
        "drill but the tool has two long squeeze handles, it is this rivet nut "
        "tool, NOT a drill."
    ),
]


def _build_equipment_hints_block() -> str:
    """Unconditional visual-ID hints appended to the VISUAL CONTEXT section."""
    if not _EQUIPMENT_ID_HINTS:
        return ""
    return (
        " Some equipment is easy to misread from the camera — apply these "
        "identification notes when what you see matches: "
        + " ".join(_EQUIPMENT_ID_HINTS)
    )


def _build_known_products_block() -> str:
    """List the procedure library's products so identification is grounded
    in what Retrace actually knows, not a generic visual guess."""
    try:
        procedures = [
            p for p in database.list_procedures()
            if p.get("status") in ("completed", "completed_partial")
        ]
    except Exception:
        logger.exception("Known-products block: library query failed; omitting")
        return ""
    if not procedures:
        return ""

    lines = []
    seen_titles = set()
    for p in procedures:
        title = " ".join((p.get("title") or "").split())
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        line = f"- {title}"
        category = (p.get("category") or "").strip()
        if category:
            line += f" (category: {category})"
        lines.append(line)
    if not lines:
        return ""

    return (
        "=== KNOWN PRODUCTS (RETRACE LIBRARY) ===\n"
        "The Retrace library has procedures for these products:\n"
        + "\n".join(lines)
        + "\n"
        "Before identifying a product — from the camera or from the user's "
        "description — check whether it matches one of these. When the "
        "visual evidence is compatible with a library product, prefer that "
        "identification over a generic lookalike guess, and pass the "
        "library product's name to identify_product.\n"
    )


# Tools Gemini Live can call during a diagnostic session. session_id is
# omitted — the phone injects it before forwarding to the server, matching
# the learner convention.
DIAGNOSTIC_TOOL_DEFINITIONS = [
    {
        "name": "identify_product",
        "description": (
            "Look up a product in the Retrace knowledge base by name, model, or "
            "free-text description. Call this as soon as the user has given you "
            "enough to make a plausible guess — even partial info is fine, brand "
            "+ rough category is enough. Do NOT keep asking clarifying questions "
            "before calling this; call it first, then refine based on the result. "
            "The server returns product metadata and any linked procedures."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text product description, model number, or brand + "
                        "category. Examples: 'Levoit LV600HH humidifier', "
                        "'Instant Pot Duo showing C6 error', 'MacBook Pro laptop "
                        "won't turn on'."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional product category to narrow the match "
                        "('humidifier', 'coffee machine', 'laptop', etc.)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "confirm_identification",
        "description": (
            "Call this when the user EXPLICITLY confirms the identified "
            "product is correct. Transitions the session into the diagnose "
            "phase. Do NOT call this without an explicit user affirmative — "
            "if the user is silent, ambiguous, or asks a clarifying question, "
            "ask them again before calling. The previous identify_product "
            "call has set the candidate; this call commits to it."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_procedures",
        "description": (
            "Search the Retrace procedure library for existing SOPs matching the "
            "identified product and the user's symptom. Call this after "
            "identify_product has returned AND you've gathered at least 2-3 "
            "concrete symptom details from the user. Returns a ranked list of "
            "candidate procedures. If the top match is high-confidence, propose "
            "it to the user and call handoff_to_learner once they confirm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name or model returned by identify_product.",
                },
                "symptom": {
                    "type": "string",
                    "description": (
                        "Plain-language summary of the observed problem, "
                        "e.g. 'no mist output, fan running, tank full, no "
                        "cleaning in 6 months'."
                    ),
                },
            },
            "required": ["product_name", "symptom"],
        },
    },
    {
        "name": "web_search_for_fix",
        "description": (
            "Search the open web for a fix when the Retrace library has no "
            "matching procedure. Call this only after search_procedures returned "
            "status='no_match'. The server runs a grounded Gemini search (biased "
            "toward Reddit + iFixit), synthesizes a step-by-step procedure from "
            "the top sources, saves it to the library, and returns a "
            "procedure_id you can hand off to the learner. Latency: ~10-20 "
            "seconds — give the user one short status line ('Looking that up — "
            "give me a sec') BEFORE calling, then stay quiet until it returns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name or model from identify_product.",
                },
                "symptom": {
                    "type": "string",
                    "description": (
                        "Short symptom phrase describing the fault, e.g. "
                        "'no mist output, fan running, tank full'."
                    ),
                },
                "diagnostic_summary": {
                    "type": "string",
                    "description": (
                        "One paragraph summarizing what was ruled in and ruled "
                        "out so far. Used to focus the search and the "
                        "synthesized procedure."
                    ),
                },
            },
            "required": ["product_name", "symptom", "diagnostic_summary"],
        },
    },
    {
        "name": "handoff_to_learner",
        "description": (
            "Conclude the diagnostic session and hand off to Learner coaching on "
            "the chosen procedure. Call this only after the user has "
            "acknowledged that a specific procedure is the right next step. "
            "The phone will close the diagnostic session and open Learner Mode "
            "on the target procedure. After calling this, your role is over — "
            "give a brief confirmation and stop talking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "procedure_id": {
                    "type": "string",
                    "description": (
                        "Procedure id from search_procedures or "
                        "web_search_for_fix."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "One-sentence diagnostic summary to inject into the "
                        "Learner session so the coach knows what was already "
                        "ruled out."
                    ),
                },
            },
            "required": ["procedure_id", "summary"],
        },
    },
]


# Shape-identical to prompt_builder.GEMINI_CONFIG so ephemeral_token can stay
# tool-agnostic. Diagnostic sessions get their own default voice pick; caller
# can override.
GEMINI_CONFIG_DIAGNOSTIC = {
    "model": RETRACE_LIVE_MODEL,
    "voice": "Charon",
}


def build_diagnostic_prompt(voice: str = "Charon") -> str:
    """Build the Gemini Live system prompt for a troubleshooting session.

    Procedure-less by design — at entry, the model knows nothing about what
    product the user has. It asks, identifies, diagnoses, and either hands
    off to an existing Learner procedure or ingests a manual to generate a
    fresh one.
    """

    # Affirmative utterances that confirm "start this procedure" come from
    # the start_matched_procedure command in voice_commands.yaml. Keeping the
    # phrasing in one place so the designer can edit triggers without diving
    # into Python.
    start_examples = vc.format_utterance_examples("start_matched_procedure")
    retry_examples = vc.format_utterance_examples("retry_diagnosis")
    known_products_block = _build_known_products_block()
    equipment_hints = _build_equipment_hints_block()

    prompt = f"""You are a field diagnostic assistant. A technician or skilled user has a broken piece of equipment in front of them and needs to figure out what's wrong and how to fix it. You are SPEAKING to them through their phone or glasses — not writing. Think of a knowledgeable colleague on a phone call: calm, curious, focused, respects their time.

=== VOICE STYLE ===
- Speak out loud, not in paragraphs. Match the cadence of a real conversation.
- One question per turn. Never stack multiple questions in the same reply.
- Keep each turn short — usually one or two sentences. Let them answer before moving on.
- Treat the user as a field expert in their own domain. If they use model numbers and part names, match that depth. If they sound uncertain, ask observable symptom questions rather than theory.
- No apologies, no "I'm just an AI", no deflecting. Acknowledge uncertainty by asking a follow-up, not by bowing out.
- Never read your own labels or phases aloud ("Phase 1", "Identifying", "Tool call"). Just do the work.

=== DIAGNOSTIC FLOW ===

PHASE 1 — IDENTIFY
Your very first turn is a warm one-line opener that asks what they're working on. Example shape: "Hey, what's going on — what are you working on?"
As soon as they name a product (even partially — brand + rough category is enough), call identify_product. Do NOT interrogate for model numbers before calling the tool. Call first, then let the tool result guide your follow-up question.

After identify_product returns, **describe the product back in one short sentence and ask the user to confirm it's right** — e.g. "I'm seeing a Levoit LV600HH humidifier, is that the one?" **Wait for their answer.** Do NOT begin asking diagnostic questions yet. The user's confirmation is mandatory before moving on.
- On any explicit affirmative ("yes", "that's it", "correct", etc.), call `confirm_identification` to advance the session into the diagnose phase. Only AFTER `confirm_identification` returns may you ask a diagnostic question.
- If the user says no, says it's a different product, or describes a different one, call `identify_product` again with the new guess and confirm again.
- If the user is silent or ambiguous, ask once more for confirmation before calling either tool.

PHASE 2 — DIAGNOSE
This phase begins ONLY after `confirm_identification` has been called. Now ask ONE observable-symptom question. Lights on? Fan running? Tank full? After the user answers, call search_procedures with the product name and a brief symptom summary. You do not need to collect 3+ symptoms before searching — one specific observation on top of the user's original complaint is enough. Search early and often; you can always refine.

If search_procedures returns candidates, describe the top one in one sentence and ask if that matches what they're seeing. If you hear ANY affirmative (e.g. {start_examples}), that is user confirmation — immediately call handoff_to_learner with the procedure_id from the top candidate. Do not ask follow-up questions after an affirmative. If instead the user says the candidate is wrong (e.g. {retry_examples}), re-enter the identify phase and ask one more clarifying question before calling identify_product again.

PHASE 3 — RESOLVE
Confirm received → call handoff_to_learner immediately. The summary argument should be one sentence of what was ruled out, like "tank full, fan running, no cleaning in 6 months — transducer mineral buildup most likely." After handoff fires, say one short line ("alright, taking you there now") and STOP. The tool has already done the work.

If search_procedures returned status='no_match' (no candidates in the library): say one short line out loud — "Looking that up online, give me a sec" — then immediately call web_search_for_fix with the product name, the symptom, and a one-paragraph diagnostic_summary covering what's been ruled in and out. The call takes ~10-20 seconds; stay quiet while it runs, do not narrate progress.
  - On status='ok': describe the synthesized procedure in one sentence and mention how many sources it drew on (e.g. "I pulled together a fix from 4 Reddit and iFixit threads"). Ask the user to confirm. On any affirmative, call handoff_to_learner with the procedure_id.
  - On status='no_fix_found': tell the user we couldn't find a matching fix online and suggest contacting a service tech. Do not call any other tool — the session ends here.
  - On status='error': tell the user we hit a snag with the search and ask if they want to try again. If yes, retry web_search_for_fix once with the same arguments. If it errors a second time, fall back to the no_fix_found message.

=== SUMMARIZING ===
After every 2-3 exchanges, give the user a one-sentence recap of what's been established so they know you're tracking. Example shape: "So: Levoit LV600HH, tank full, fan running, no mist, no cleaning in about six months." Then ask the next diagnostic question.

{known_products_block}
=== VISUAL CONTEXT ===
You can see the user's phone camera feed. Identify the device in the central focus of the frame from what you see, and call `identify_product` from that — don't wait for the user to name a brand or model. If text on the device is clearly legible, use it; if not, trust your recognition of the object itself. Check the KNOWN PRODUCTS list first: when what you see is compatible with a library product, identify it as that product rather than a generic lookalike. Confirm or contradict the user's described symptoms when they're visibly observable. Don't narrate the scene, don't ask them to reframe the camera, don't comment on what they're holding unless it's directly diagnostic.{equipment_hints}

=== ESCALATION ===
If 5 exchanges pass without a clear product identification, stop asking and call identify_product with whatever you have. The tool response will tell you how to proceed.

=== NEVER ===
- Never propose a diagnosis before calling the tool that validates it (search_procedures or web_search_for_fix).
- Never read step numbers, phase names, or tool names aloud.
- Never ramble. If the user is silent, stay silent.
- Never claim to have fixed something you haven't verified.

=== VOICE ===
Your voice is {voice}. Keep it warm but crisp — closer to a good phone-call voice than a narrator.
"""

    logger.info(
        "Built diagnostic prompt (voice=%s, %d chars)",
        voice, len(prompt),
    )
    return prompt
