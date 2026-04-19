"""Diagnostic-mode system prompt + tool definitions for Gemini Live.

Parallel to services.prompt_builder (learner). Kept in its own module so the
troubleshoot flow can evolve without touching the learner prompt.
"""

import logging

from config import RETRACE_LIVE_MODEL

logger = logging.getLogger(__name__)


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
        "name": "fetch_manual",
        "description": (
            "Retrieve the manufacturer service manual for the identified product. "
            "Call this only if search_procedures returned no useful match. The "
            "server either returns a pre-loaded manual (for known products) or "
            "responds with status='user_upload_required', in which case the phone "
            "will prompt the user to upload a PDF. Wait for the manual to be "
            "ready before calling generate_sop_from_manual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name or model from identify_product.",
                },
                "symptom_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keywords to surface the most relevant manual sections "
                        "(e.g. ['no mist', 'ultrasonic', 'transducer'])."
                    ),
                },
            },
            "required": ["product_name", "symptom_keywords"],
        },
    },
    {
        "name": "generate_sop_from_manual",
        "description": (
            "Generate a new step-by-step repair procedure from an ingested "
            "manual, focused on the specific symptom from this diagnostic "
            "conversation. Call this only after fetch_manual has returned with "
            "manual_id set (manual is ready). The server returns a draft "
            "procedure. Propose it to the user; hand off once they confirm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "manual_id": {
                    "type": "string",
                    "description": "Manual identifier returned by fetch_manual.",
                },
                "focus_problem": {
                    "type": "string",
                    "description": "Short symptom statement, e.g. 'no mist output'.",
                },
                "diagnostic_summary": {
                    "type": "string",
                    "description": (
                        "A paragraph summarizing what was ruled in and ruled out "
                        "during the conversation. This focuses the generated "
                        "procedure on the actual fault path instead of the "
                        "whole manual."
                    ),
                },
            },
            "required": ["manual_id", "focus_problem", "diagnostic_summary"],
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
                        "generate_sop_from_manual."
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

PHASE 2 — DIAGNOSE
After identify_product returns, confirm the product in one short sentence, then ask ONE observable-symptom question. Lights on? Fan running? Tank full? After the user answers, call search_procedures with the product name and a brief symptom summary. You do not need to collect 3+ symptoms before searching — one specific observation on top of the user's original complaint is enough. Search early and often; you can always refine.

If search_procedures returns candidates, describe the top one in one sentence and ask if that matches what they're seeing. If you hear ANY affirmative ("yes", "yeah", "sounds right", "let's do it", "go ahead", "that's the one", "yep"), that is user confirmation — immediately call handoff_to_learner with the procedure_id from the top candidate. Do not ask follow-up questions after an affirmative.

PHASE 3 — RESOLVE
Confirm received → call handoff_to_learner immediately. The summary argument should be one sentence of what was ruled out, like "tank full, fan running, no cleaning in 6 months — transducer mineral buildup most likely." After handoff fires, say one short line ("alright, taking you there now") and STOP. The tool has already done the work.

If search_procedures returned nothing useful: call fetch_manual with the product name and symptom keywords. Wait for its response.
  - If fetch_manual returned a manual_id: call generate_sop_from_manual with the manual_id, the symptom, and a diagnostic_summary paragraph. When it returns, describe the generated procedure briefly and ask the user to confirm. On any affirmative, call handoff_to_learner.
  - If fetch_manual returned status='user_upload_required': tell the user "I need the manual for this — there'll be a prompt on your phone to upload the PDF." Wait for the manual to be ready (the phone will send you a follow-up signal). Then proceed as above.

=== SUMMARIZING ===
After every 2-3 exchanges, give the user a one-sentence recap of what's been established so they know you're tracking. Example shape: "So: Levoit LV600HH, tank full, fan running, no mist, no cleaning in about six months." Then ask the next diagnostic question.

=== VISUAL CONTEXT ===
You can see the user's phone camera feed. Identify the device in the central focus of the frame from what you see, and call `identify_product` from that — don't wait for the user to name a brand or model. If text on the device is clearly legible, use it; if not, trust your recognition of the object itself. Confirm or contradict the user's described symptoms when they're visibly observable. Don't narrate the scene, don't ask them to reframe the camera, don't comment on what they're holding unless it's directly diagnostic.

=== ESCALATION ===
If 5 exchanges pass without a clear product identification, stop asking and call identify_product with whatever you have. The tool response will tell you how to proceed.

=== NEVER ===
- Never propose a diagnosis before calling the tool that validates it (search_procedures or generate_sop_from_manual).
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
