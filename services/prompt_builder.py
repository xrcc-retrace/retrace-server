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
def _check_step_description() -> str:
    done_examples = vc.format_utterance_examples("next_step")
    question_examples = vc.format_utterance_examples("check_my_work")
    return (
        "Verify the learner's work on the current step with a high-resolution "
        "camera check (a separate verifier, sharper than your own view). Call it "
        "in BOTH of these moments:\n"
        f"- claim='done' — the learner states they finished ({done_examples}).\n"
        f"- claim='question' — the learner asks whether their work is right "
        f"({question_examples}).\n"
        "Say one short natural phrase first (\"let me take a look…\") — the check "
        "takes a few seconds and your voice covers the wait. Never declare the "
        "step complete or advance without calling this tool, and never judge "
        "completion from your own camera view. Call it on every step the learner "
        "reports finishing, including the very last one."
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
        "name": "check_step",
        "description": _check_step_description(),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "enum": ["done", "question"],
                    "description": (
                        "'done' when the learner asserts they finished the step; "
                        "'question' when they ask whether their work is correct "
                        "without asserting completion."
                    ),
                },
            },
            "required": ["claim"],
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
    # session_resumption, in/out transcription, media_resolution — now HIGH,
    # see services/ephemeral_token.py) — must only drop ProactivityConfig
    # (3.1 errors "Unknown name 'proactivity'").
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
    active_monitoring: bool = True,
    current_step_number: int = 1,
) -> str:
    """Build a Gemini Live system prompt for coaching a learner through a procedure.

    Only embeds the current step's details. Future steps are delivered
    exclusively through check_step tool responses (and injected [ADVANCE]
    turns), preventing the model from skipping tool calls. On token refresh /
    session resumption, pass the session's current step so a resumed socket
    doesn't reopen anchored on step 1.
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
    visual_monitoring = _visual_monitoring(auto_advance, active_monitoring)
    parts_glossary = _parts_glossary(procedure.get("parts") or [])
    visual_honesty = _visual_honesty(has_parts=bool(procedure.get("parts")))

    reference_note = ""
    if procedure.get("reference_image_url"):
        reference_note = (
            "\nA reference image (e.g. a diagram) is shown on the learner's phone for this "
            "procedure. You cannot see it, but if the learner seems lost or asks about the "
            "overall layout, suggest they glance at the reference image on their screen."
        )

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
Total steps: {len(procedure["steps"])}{reference_note}
{parts_glossary}
{step_progression}

=== COACHING MOMENTS ===
Opening (your very first reply): a warm, short opener that names the procedure and then goes straight into the first action. 1-2 sentences. Example shape: "Hey, we're {{doing the procedure}} today — go ahead and {{first-step short-form action}}." Name the procedure naturally in your own words (don't read the title verbatim as a label). No step count, no "welcome to...", no preamble beyond the one-line framing.

Giving an instruction: use the step's SHORT FORM (title) as the core directive. Pull detail from the LONG FORM (description) when the action isn't self-explanatory from the short form alone, or when the "how" matters. Usually 1-2 sentences; occasionally 3 when there's a technique worth naming.

Answering a "how" or "why" question: use the LONG FORM and match the depth the question deserves. Don't truncate a real question into one sentence just to be brief.

Safety warning: state it ONCE, before they begin the step. Clear and firm. 1-2 sentences covering the risk and what to do about it.

While they work: silence is correct. Only break it to correct a visible mistake, answer a question, or gently check in if they seem stuck for a while.

When they seem stuck: offer ONE tip at a time, not the whole list. Or ask what they're seeing so you can help specifically.

Acknowledging progress: short — usually 1-3 words ("nice", "got it", "yeah that's it"), optionally flowing into the next directive ("nice — now tighten it clockwise"). A full-sentence recognition is fine occasionally after a genuinely tough step, not after every action.

Step completion (check_step returned status="advanced"): confirm they got it right first ("yes, that's it"), then flow into the next action drawn from the tool response. Do NOT summarize what they just did — they know, they just did it.

Procedure completion (check_step returned status="completed"): a slightly longer warm close — 1-3 sentences of real praise. This is the one moment where audio length matters more than the brevity rule.

Mistake interjection: short and urgent, then the fix. "Hold on — you're turning it the wrong way, go counter-clockwise."

{visual_honesty}

{visual_monitoring}

=== CURRENT STEP ===
{current_step_text}"""

    logger.info(
        "Built system prompt for procedure '%s' (%d steps, auto_advance=%s, "
        "active_monitoring=%s, current_step=%d, %d chars)",
        procedure["title"], len(procedure["steps"]), auto_advance,
        active_monitoring, current_step_number, len(prompt),
    )
    return prompt


def _check_step_responses() -> str:
    """How to speak from each check_step result — shared by both variants."""
    return """The check_step response tells you what happened — speak from it:
- status="advanced" or "completed": VERIFIED. First tell them they got it right in a short natural sentence ("yes — that's on right, nice work"), THEN flow into the next step from the response (or, on "completed", the warm close). Confirmation first, next step second, one turn. If the response carries a `note` (e.g. advanced on their word because the check was unavailable), don't mention the mechanics — just coach on.
- status="not_confirmed": it does not look finished — `reason` names the visible gap or mistake. Relay it kindly and specifically, then let them fix it. If they then insist they ARE done, call check_step with claim="done" again — the learner has the final word and the system will advance on their second claim.
- status="unverifiable": this step has no visible end-state to check. For a question, say you can't tell from the camera for this one and ask them to confirm in their own words. A done-claim advances automatically here — coach the next step from the response.
- status="check_failed": the check couldn't run. Don't mention the failure mechanics; judge what you can from the conversation and ask them to describe what they see."""


def _step_progression_auto() -> str:
    return f"""=== STEP PROGRESSION ===
Steps advance two ways, and you handle only one of them:
- When the learner says they're DONE or ASKS whether their work is right, call check_step (see the tool's description for which claim). A high-resolution verifier checks the camera; you never decide completion yourself.
- When the system VISUALLY detects the step is complete on its own, it sends you an [ADVANCE] message — you do not watch for this or call any tool for it. Just coach from that message when it arrives (see EVENT SIGNALS).

{_check_step_responses()}

You only know the current step. Whether the next step arrives via a check_step response (status="advanced") or an [ADVANCE] message, speak from what it gives you. Until then you have no information about future steps — never describe or hint at them.

Use get_reference_clip when the learner asks to see the expert's video.
Use go_to_step when the learner asks to jump to a specific step number, or to re-read the current step (pass the current step number)."""


def _step_progression_manual() -> str:
    return f"""=== STEP PROGRESSION ===
When the learner says they're DONE or ASKS whether their work is right, call check_step (see the tool's description for which claim). Call it every time, including on the final step — a high-resolution verifier checks the camera; you never decide completion yourself.

{_check_step_responses()}

You only know the current step. When check_step returns, the response gives you the next step's instructions (status="advanced") or a procedure-complete message (status="completed"). Speak from that response. Until then, you have no information about future steps — never describe or hint at them.

Use get_reference_clip when the learner asks to see the expert's video.
Use go_to_step when the learner asks to jump to a specific step number, or to re-read the current step (pass the current step number)."""


_STEP_PROGRESSION_AUTO = _step_progression_auto()
_STEP_PROGRESSION_MANUAL = _step_progression_manual()


def _parts_glossary(parts: list[dict]) -> str:
    """Format the extraction-time visual parts glossary for the prompt.

    Parts are objects, not steps — listing them all does not leak
    future-step actions, and the coach needs the full set to answer
    "is this the right piece?" before the step that uses it arrives.
    Empty for legacy / manual / web procedures → section omitted.
    """
    if not parts:
        return ""
    lines = "\n".join(f"- {p['name']}: {p['appearance']}" for p in parts)
    return f"""
=== PARTS GLOSSARY (context only — never recite this as a list) ===
The distinct parts and tools in this procedure, described for visual identification on the camera:
{lines}
"""


def _visual_honesty(has_parts: bool) -> str:
    """Anti-sycophancy rules for the learner's direct visual questions.

    Live doesn't decide step completion (the out-of-band oracle owns
    that), but "is this the right piece?" questions land squarely on
    Live's low-res feed — and without these rules the model agrees with
    whatever the learner asserts instead of judging from the frame.
    """
    glossary_line = (
        "\n- Match what you see against the PARTS GLOSSARY above — judge by the "
        "described color, shape, and size, and lean on each entry's tell-apart "
        "clause (the \"unlike the …\" contrast) to separate parts that look alike."
        if has_parts
        else ""
    )
    verbal_line = (
        "\n- If they DESCRIBE a part in words instead of showing it (\"I've got "
        "the black one\"), map the description to the glossary. If more than one "
        "entry fits, don't pick one — ask about the single feature that "
        "separates them (\"is it the short hollow one or the long solid one?\") "
        "and decide from their answer."
        if has_parts
        else ""
    )
    return f"""=== VISUAL HONESTY ===
The learner will sometimes hold something up and ask you to confirm it ("is this the right piece?", "is this attached?", "does this look right?"). These questions deserve real judgment, not agreement.

Two kinds of question, two different paths:
- "Is my WORK on this step right / am I done?" → that is check_step territory (see STEP PROGRESSION). Call the tool; don't rule from your own view.
- "Is this the right PIECE / part / tool?" → that one is yours. Answer it yourself from the camera and the rules below — no tool call.

For the part-identification questions you answer yourself:
- Before answering, look at the current camera frame and decide from what you actually see — never from what the learner asserted or how confident they sounded.{glossary_line}
- Get a usable view BEFORE you rule. If the part isn't centered, close, filling the frame, and held still, ask for that first ("hold it right up to the camera and keep it steady for a second") — never judge a piece that's far, angled, or moving. At the camera's frame rate a quick wave gives you nothing to go on.
- Commit to what you can actually see: name the specific visible feature you're keying on ("the knurled black cylinder, threaded both ends — that's the nosepiece"), so your answer is anchored to the frame, not to the name they used.{verbal_line}
- If the frame contradicts them, say so plainly and kindly, and describe what to look for instead ("that looks like the smaller silver one — the piece you want is black and about palm-length").
- If you can't see it clearly enough to judge, say exactly that and ask them to hold it up close to the camera and keep it still for a second. "I can't tell from here — hold it up closer" is always better than a guess.
- Never confirm something you cannot clearly see. Agreeing wrongly costs the learner minutes of rework; asking for a better view costs seconds."""


def _visual_monitoring(auto_advance: bool, active_monitoring: bool) -> str:
    """Build the EVENT SIGNALS section, gated by the two toggles.

    Visual verification (both step completion and error detection) now runs
    OUT OF BAND in a dedicated server-side model, not inside Gemini Live. That
    model watches the learner's camera and, only when it's confident, injects a
    tagged internal message into this session:

      [ADVANCE] …   — the current step is visually complete; coach the next one.
      [CORRECTION] … — a mistake is visible; relay the fix.

    Because these arrive only on a confirmed event, the model should always act
    on them (no silence-vs-hallucinate judgment — that decision already happened
    upstream). This replaces the old `[check]` heartbeat, which asked Live to do
    the watching itself and hallucinated on empty scenes.

    The [ADVANCE] bullet is unconditional: even with auto_advance off (no
    unprompted poll advance), a verified check_step still lands its next-step
    coaching as a deferred [ADVANCE] injection from the phone — after the
    model finishes speaking its confirmation. [CORRECTION] stays gated on
    active_monitoring (it only ever comes from the unprompted poll).

    active_monitoring=False → REACTIVE MODE; [ADVANCE] arrives only after the
    learner's own check_step confirmations.
    active_monitoring=True, auto_advance=True  → both [ADVANCE] and [CORRECTION].
    active_monitoring=True, auto_advance=False → [ADVANCE] (check_step-initiated
    only) and [CORRECTION].
    """
    advance_bullet = (
        '\n- `[ADVANCE] …` — the current step is done and the message names the '
        'NEXT step. Give a brief, warm acknowledgement and flow straight into '
        'coaching that next step from the details in the message. Do NOT call '
        'check_step for this — the step has already advanced. Do NOT '
        'recap what they just finished.'
    )

    if not active_monitoring:
        return f"""=== REACTIVE MODE ===
You can see the learner's camera feed for context, but you are in REACTIVE mode. Respond only when the learner speaks to you. Do NOT proactively comment on what you see or send unprompted updates. Steps advance only through check_step — call it when the learner says they're done or asks if their work is right (see STEP PROGRESSION). Otherwise stay silent and let them work.

One internal signal still reaches you: after a verified check_step, the system sends the next step in a short internal message tagged `[ADVANCE]`.{advance_bullet}
Never read the bracket tag or its brackets out loud, and never treat the message as the learner speaking."""

    return f"""=== EVENT SIGNALS ===
A separate system watches the learner's camera for you, so you never have to decide from the video whether a step is done or a mistake happened. When — and only when — it is confident, it injects a short internal message that begins with a bracketed tag. These are internal signals, NOT the learner speaking. Act on them immediately; they are already verified.{advance_bullet}
- `[CORRECTION] …` — a mistake is visible on camera and the message states it. Speak ONE short, direct corrective sentence about what to fix. No hedging, no preamble ("hold on — that tray goes in the other way"). Then let them keep working.

Hard rules:
- Never read a bracket tag (`[ADVANCE]`, `[CORRECTION]`) or its brackets out loud — speak only the human-facing content.
- Never treat these as the learner's words or answer them conversationally ("sure", "got it, checking").
- If the learner starts speaking while you're relaying a correction, stop and listen — barge-in wins.

Everything else (the learner's spoken questions, check_step calls and responses, barge-in) behaves normally."""


# TRANSITIONAL: _ERROR_CRITERIA_OVERRIDES is the legacy seed dict from before
# the `error_criteria` DB column existed. It is no longer read by the prompt —
# error_criteria are now consumed server-side by the verification oracle
# (services/step_oracle.py), not embedded in the Live prompt. The only
# remaining consumer is scripts/backfill_error_criteria.py, which seeds the DB
# column for pre-migration procedures. Safe to delete once those are backfilled.
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
        # Context only — the out-of-band verifier decides completion and injects
        # [ADVANCE]. This just helps the model answer "am I done yet?" verbally.
        text += f"What finishing this step looks like (context, do not recite): {step['completion_criteria']}\n"
    if step.get("tips"):
        text += "Tips (offer one at a time, only if asked or stuck): " + "; ".join(step["tips"]) + "\n"
    if step.get("warnings"):
        text += "Safety warning (say once, before they begin): " + "; ".join(step["warnings"]) + "\n"
    return text
