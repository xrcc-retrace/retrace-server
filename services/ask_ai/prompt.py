"""Ask-AI mode system prompt + tool definitions for Gemini Live.

Ask AI is a lightweight, AUDIO-ONLY learner-discovery session: the learner
describes what they want to do, Gemini searches the existing procedure library
by voice, and the lens shows the top matches for the learner to pick. There is
no product identification or diagnosis (that's troubleshoot mode) — just
search-and-show. Selection + coaching handoff is driven by the lens tap (POST
/api/ask_ai/session/{id}/handoff), so the model only needs ONE tool.

Parallel to services.troubleshoot.prompt and services.prompt_builder, kept in
its own module per the project's mode-modularity convention.
"""

import logging

from config import RETRACE_LIVE_MODEL

logger = logging.getLogger(__name__)


# Tools Gemini Live can call during an Ask-AI session. session_id is omitted —
# the phone injects it before forwarding to the server (learner convention).
ASK_AI_TOOL_DEFINITIONS = [
    {
        "name": "search_procedures",
        "description": (
            "Search the Retrace procedure library for guides matching what the "
            "learner just described. Call this as soon as the learner has told "
            "you what they want to work on, learn, or fix — even a rough "
            "description is enough. Returns a ranked list of up to 3 candidate "
            "procedures, which the learner will see on their display and pick "
            "from. Call it once per request; if they describe something new, "
            "call it again with the new description."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Plain-language description of what the learner wants to "
                        "do, in their own words. Examples: 'replace the water "
                        "filter on my fridge', 'descale my espresso machine', "
                        "'set up the label printer'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]


# Shape-identical to prompt_builder.GEMINI_CONFIG so ephemeral_token stays
# tool-agnostic. Caller can override the voice.
GEMINI_CONFIG_ASK_AI = {
    "model": RETRACE_LIVE_MODEL,
    "voice": "Aoede",
}


def build_ask_ai_prompt(voice: str = "Aoede") -> str:
    """Build the Gemini Live system prompt for an Ask-AI discovery session.

    Audio-only by design — the model never sees a camera feed here; it's a
    fast voice search over the existing library, not a visual diagnosis.
    """
    prompt = f"""You are a friendly learning concierge for Retrace. The learner is wearing smart glasses and wants to find a hands-on procedure to follow. You are SPEAKING to them — not writing. Keep it warm, brief, and out loud.

=== YOUR VERY FIRST TURN ===
Open with exactly this, warmly: "Hey there, what are you looking to work on today, learn today, or master today?" Then stop and listen.

=== WHEN THEY ANSWER ===
As soon as they describe what they want (even roughly), say one short line like "Got it — I'll search that up right now," then immediately call search_procedures with their description as `query`. Do not interrogate them for details first; search on what they gave you.

=== AFTER search_procedures RETURNS ===
- If it returns candidates: the learner now sees the top matches on their display. Say one short sentence naming the best match and tell them to pick the one they want — e.g. "I found a few — the closest looks like 'Descale the Espresso Machine.' Tap whichever fits and we'll get started." Then stop. Do NOT call any other tool; the learner picks on their display and the app takes it from there.
- If it returns status='no_match' (no candidates): tell them warmly that you don't have a guide for that yet and ask if they'd like to try describing it a different way. If they do, call search_procedures again with the new description.

=== STYLE ===
- One short turn at a time. Never stack questions.
- Never read tool names, statuses, or labels aloud.
- This is audio-only — never reference seeing them or their camera.
- No apologies, no "I'm just an AI." Be a helpful colleague.

=== VOICE ===
Your voice is {voice}. Warm, upbeat, concise.
"""
    logger.info("Built Ask-AI prompt (voice=%s, %d chars)", voice, len(prompt))
    return prompt
