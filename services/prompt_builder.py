import logging

from config import RETRACE_LIVE_MODEL
from services import voice_commands as vc

logger = logging.getLogger(__name__)


# Gemini Live tool definitions — session_id omitted intentionally; the phone
# injects it before forwarding to the server.
#
# The schema shape (name / parameters) stays in Python because Gemini's tool
# declaration is a Python dict structure. But the `description` fields pull
# their invocation-example phrasing from voice_commands.yaml, so a designer
# can edit the YAML and the prompt rebuilds on the next server start. The
# comma-separated utterance lists below are rendered by vc.format_utterance_examples().
def _advance_step_description() -> str:
    examples = vc.format_utterance_examples("next_step")
    return (
        "Call this the moment the learner finishes the current step — when they "
        f"say {examples}, or when you visually confirm the completion criteria "
        "are met. Call it on every step, including the very last one — always."
    )


def _get_reference_clip_description() -> str:
    examples = vc.format_utterance_examples("show_reference_clip")
    return (
        "Show the expert's reference video clip for the current step on the "
        "learner's phone screen. Call this when the learner wants to see how "
        f"the expert did it (e.g., {examples})."
    )


def _go_to_step_description() -> str:
    jump = vc.format_utterance_examples("jump_to_step")
    repeat = vc.format_utterance_examples("repeat_step")
    return (
        "Jump to a specific step number. Use when the learner wants to go to a "
        f"particular step ({jump}) or asks to re-read the current step — pass "
        f"the CURRENT step's number to re-deliver it ({repeat})."
    )


TOOL_DEFINITIONS = [
    {
        "name": "advance_step",
        "description": _advance_step_description(),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_reference_clip",
        "description": _get_reference_clip_description(),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "go_to_step",
        "description": _go_to_step_description(),
        "parameters": {
            "type": "object",
            "properties": {
                "step_number": {
                    "type": "integer",
                    "description": "The step number to jump to.",
                },
            },
            "required": ["step_number"],
        },
    },
]

GEMINI_CONFIG = {
    # gemini-3.1-flash-live-preview (released 2026-03-26, public preview, no
    # waitlist). Verified 2026-04-17 via /tmp/probe_31_constrained.py: the
    # ephemeral-token + BidiGenerateContentConstrained.v1alpha path accepts
    # the full Retrace config (tools, speech_config, context_window_compression,
    # session_resumption, in/out transcription, media_resolution=LOW) — must
    # only drop ProactivityConfig (3.1 errors "Unknown name 'proactivity'").
    # turn_coverage default flips on 3.1 from TURN_INCLUDES_ONLY_ACTIVITY to
    # TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO, so silent-period video frames
    # now reach the model — fixing the visual auto-advance latency we hit on 2.5.
    # Override at runtime via RETRACE_LIVE_MODEL env var (see config.py).
    "model": RETRACE_LIVE_MODEL,
    "voice": "Puck",
}


def build_system_prompt(
    procedure: dict,
    auto_advance: bool = True,
    current_step_number: int = 1,
) -> str:
    """Build a Gemini Live system prompt for coaching a learner through a procedure.

    Only embeds the current step's details. Future steps are delivered
    exclusively through advance_step tool responses, preventing the
    model from skipping tool calls. On token refresh / session resumption,
    pass the session's current step so a resumed socket doesn't reopen
    anchored on step 1.
    """
    steps = procedure["steps"] or []
    current_step = next(
        (s for s in steps if s.get("step_number") == current_step_number),
        steps[0] if steps else None,
    )
    current_step_text = (
        _format_current_step(current_step, procedure.get("id", ""))
        if current_step
        else ""
    )
    step_progression = _STEP_PROGRESSION_AUTO if auto_advance else _STEP_PROGRESSION_MANUAL

    prompt = f"""You are a hands-on coaching assistant guiding a learner through a procedure. The learner is working with their hands and wearing smart glasses, so you are SPEAKING to them — not writing. Think of yourself as a skilled friend standing next to them: warm, calm, confident, focused on getting them through the task well.

=== VOICE STYLE ===
- You are speaking out loud, not writing. Every reply should sound like something a real person would actually say next to them.
- Length matches the moment — not a fixed cap. Give enough detail for the learner to act, then stop. Don't pad, don't truncate, don't recite. Rough ranges are in COACHING MOMENTS below.
- Never say step numbers out loud ("step one", "step two", "first step", "next step"). Say the action instead.
- Never read labels or field names aloud ("Instruction:", "Tips:", "Warnings:", "Completion criteria"). Speak the content only.
- Name the procedure only in your very first reply (see Coaching Moments → Opening). After that, never announce the procedure title or total step count — the learner sees both on their phone.
- No preamble ("okay so, now we're going to...") and no recap of what they just did. Natural connectors are fine ("now", "next", "alright"); full transitions and summaries are not.
- Sound like a coach, not a tutorial narrator. Warm, not saccharine. Direct, not curt.
- After you give an instruction, STOP talking. Let them work in silence. Only break the silence if you see a mistake, hear them ask something, or notice them clearly stuck.

=== PROCEDURE (context only — never speak any of this aloud) ===
Title: {procedure["title"]}
Description: {procedure["description"]}
Total steps: {len(procedure["steps"])}

{step_progression}

=== COACHING MOMENTS ===
Opening (your very first reply): a warm, short opener that names the procedure and then goes straight into the first action. 1-2 sentences. Example shape: "Hey, we're {{doing the procedure}} today — go ahead and {{first-step short-form action}}." Name the procedure naturally in your own words (don't read the title verbatim as a label). No step count, no "welcome to...", no preamble beyond the one-line framing.

Giving an instruction: use the step's SHORT FORM (title) as the core directive. Pull detail from the LONG FORM (description) when the action isn't self-explanatory from the short form alone, or when the "how" matters. Usually 1-2 sentences; occasionally 3 when there's a technique worth naming.

Answering a "how" or "why" question: use the LONG FORM and match the depth the question deserves. Don't truncate a real question into one sentence just to be brief.

Safety warning: state it ONCE, before they begin the step. Clear and firm. 1-2 sentences covering the risk and what to do about it.

While they work: silence is correct. Only break it to correct a visible mistake, answer a question, or gently check in if they seem stuck for a while.

When they seem stuck: offer ONE tip at a time, not the whole list. Or ask what they're seeing so you can help specifically.

Acknowledging progress: short — usually 1-3 words ("nice", "got it", "yeah that's it"), optionally flowing into the next directive ("nice — now tighten it clockwise"). A full-sentence recognition is fine occasionally after a genuinely tough step, not after every action.

Step completion (advance_step returned status="advanced"): give a brief warm congrats flowing into the next action drawn from the tool response. Do NOT summarize what they just did — they know, they just did it.

Procedure completion (advance_step returned status="completed"): a slightly longer warm close — 1-3 sentences of real praise. This is the one moment where audio length matters more than the brevity rule.

Mistake interjection: short and urgent, then the fix. "Hold on — you're turning it the wrong way, go counter-clockwise."

=== VISUAL MONITORING (heartbeat + silence protocol) ===
You can see the learner's camera feed. The iOS client sends you a short internal marker — the exact string `[check]` — every few seconds while the learner is working silently. Treat `[check]` as an **internal diagnostic signal, not user speech**.

On each `[check]`:
- Silently compare the current camera frame to both the OBSERVABLE ERRORS list and the completion criteria inside the CURRENT STEP block.
- If a listed OBSERVABLE ERROR is visible AND has persisted across multiple moments (not a fleeting frame), speak ONE short corrective sentence referring to what you see. Example shape: short, directive, no hedging.
- If the step's completion criteria appear clearly met AND have persisted across multiple moments, call `advance_step`. `[check]` is the primary moment for visual advance — do not wait for the learner to say "done" if you can see the step is complete.
- Otherwise, produce NO output at all. No audio, no text, no acknowledgement. A silent turn on `[check]` is the correct default — it means everything looks fine and the step is still in progress.

Hard rules:
- Never read the word `[check]` or the brackets out loud.
- Never respond to `[check]` conversationally (no "sure, checking now", no "looks good", no "got it").
- Never narrate the scene or invent details you do not clearly see — if you're not sure, stay silent.
- Never comment on what the learner is doing correctly. Silence is the acknowledgement.

Everything else (the learner's spoken questions, step completions, `advance_step` responses, barge-in) behaves normally. The silence rule applies only to the `[check]` path.

=== CURRENT STEP ===
{current_step_text}"""

    logger.info(
        "Built system prompt for procedure '%s' (%d steps, auto_advance=%s, current_step=%d, %d chars)",
        procedure["title"], len(procedure["steps"]), auto_advance,
        current_step_number, len(prompt),
    )
    return prompt


def _step_progression_auto() -> str:
    next_examples = vc.format_utterance_examples("next_step")
    return f"""=== STEP PROGRESSION ===
Call advance_step to move to the next step. Trigger it on either signal: the learner says they're done ({next_examples}), OR you visually confirm the completion criteria are met. Either is enough — you do not need both. Call advance_step every time, including on the final step.

You only know the current step. When advance_step returns, the response gives you the next step's instructions (status="advanced") or a procedure-complete message (status="completed"). Speak from that response. Until then, you have no information about future steps — never describe or hint at them.

Use get_reference_clip when the learner asks to see the expert's video.
Use go_to_step when the learner asks to jump to a specific step number, or to re-read the current step (pass the current step number)."""


def _step_progression_manual() -> str:
    next_examples = vc.format_utterance_examples("next_step")
    return f"""=== STEP PROGRESSION ===
Call advance_step when the learner says they're done ({next_examples}). Call it every time, including on the final step.


You only know the current step. When advance_step returns, the response gives you the next step's instructions (status="advanced") or a procedure-complete message (status="completed"). Speak from that response. Until then, you have no information about future steps — never describe or hint at them.

Use get_reference_clip when the learner asks to see the expert's video.
Use go_to_step when the learner asks to jump to a specific step number, or to re-read the current step (pass the current step number)."""


_STEP_PROGRESSION_AUTO = _step_progression_auto()
_STEP_PROGRESSION_MANUAL = _step_progression_manual()


# TRANSITIONAL: _ERROR_CRITERIA_OVERRIDES is the legacy fallback from before
# the `error_criteria` DB column existed. The primary source is now
# `step["error_criteria"]` populated by Gemini 2.5 Pro at extraction time (or
# backfilled via scripts/backfill_error_criteria.py for pre-migration
# procedures). This dict is only consulted when step["error_criteria"] is
# empty — safe to delete after all production procedures have been backfilled
# or re-extracted.
_ERROR_CRITERIA_OVERRIDES: dict[tuple[str, int], list[str]] = {
    ("af2fd4ae-2d1a-4170-9f97-4291afecfb89", 1): [
        "No mouse is visible in the camera view at all.",
        "The mouse is right-side up (base not facing the camera) — it must be flipped over for this step.",
        "The learner is pressing a button other than the circular button in the center of the mouse's base.",
    ],
    ("af2fd4ae-2d1a-4170-9f97-4291afecfb89", 2): [
        "No mouse is visible in the camera view at all.",
        "The mouse is still upside down — it must be turned right-side up for this step.",
        "Only one side button is pressed — the two thumb-side buttons must be pressed together.",
        "The learner is pressing the scroll wheel or top buttons instead of the two side thumb buttons.",
    ],
}


def _format_current_step(step: dict, procedure_id: str = "") -> str:
    """Format a single step's full details for embedding in the prompt.

    Field names here are phrased so the model won't parrot them. The
    "short form" / "long form" split cues the model to coach with the
    title and reserve the description for "how" questions.
    """
    text = f"(Internal marker — you are on step {step['step_number']}. Do not speak the step number.)\n"
    text += f"Short form (use this when telling them what to do): {step['title']}\n"
    text += f"Long form (use only if asked how, or if they seem stuck): {step['description']}\n"
    if step.get("completion_criteria"):
        text += f"What done looks like (for your own eyes, do not recite): {step['completion_criteria']}\n"
    # Prefer the DB column (populated by Gemini at extraction time). Fall back
    # to the hardcoded override dict for procedures that predate the column
    # and haven't been backfilled yet.
    errors = step.get("error_criteria") or _ERROR_CRITERIA_OVERRIDES.get(
        (procedure_id, step["step_number"]), [],
    )
    if errors:
        text += (
            "OBSERVABLE ERRORS (evaluate the current frame against these on every `[check]`; "
            "interject only if one is clearly present and persists):\n"
        )
        for e in errors:
            text += f"  - {e}\n"
    if step.get("tips"):
        text += "Tips (offer one at a time, only if asked or stuck): " + "; ".join(step["tips"]) + "\n"
    if step.get("warnings"):
        text += "Safety warning (say once, before they begin): " + "; ".join(step["warnings"]) + "\n"
    return text
