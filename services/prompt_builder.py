import logging

logger = logging.getLogger(__name__)

# Gemini Live tool definitions — session_id omitted intentionally;
# the phone injects it before forwarding to the server.
TOOL_DEFINITIONS = [
    {
        "name": "advance_step",
        "description": (
            "Mark the current step as complete and move to the next step. "
            "Call this when the learner confirms they have finished the current step "
            "(e.g., they say 'done', 'finished', 'next', 'I did it')."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_reference_clip",
        "description": (
            "Show the expert's reference video clip for the current step on the "
            "learner's phone screen. Call this when the learner wants to see how "
            "the expert did it (e.g., 'show me', 'let me see', 'how does it look')."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "go_to_step",
        "description": (
            "Jump to a specific step number. Use when the learner wants to go back "
            "to a previous step or skip to a particular step."
        ),
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
    "model": "gemini-2.5-flash-live-preview",
    "voice": "Puck",
}


def build_system_prompt(procedure: dict, auto_advance: bool = True) -> str:
    """Build a Gemini Live system prompt for coaching a learner through a procedure.

    Only embeds the first step's details. Future steps are delivered
    exclusively through advance_step tool responses, preventing the
    model from skipping tool calls.
    """
    first_step = procedure["steps"][0] if procedure["steps"] else None
    current_step_text = _format_current_step(first_step) if first_step else ""
    step_progression = _STEP_PROGRESSION_AUTO if auto_advance else _STEP_PROGRESSION_MANUAL

    prompt = f"""You are a hands-on coaching assistant guiding a learner through a procedure step by step. The learner is working with their hands and wearing smart glasses. Keep responses concise — 1 to 2 sentences unless they ask for more detail. Supportive, encouraging tone.

=== PROCEDURE ===
Title: {procedure["title"]}
Description: {procedure["description"]}
Total steps: {len(procedure["steps"])}

=== TOOL USAGE (CRITICAL — YOU MUST FOLLOW THESE RULES) ===
You only know the current step. You have absolutely no information about any other step.
- NEVER describe, name, summarize, or hint at what any future step involves. You do not know this information.
- NEVER say things like "next we'll..." or "the next step is..." before calling advance_step.
- The advance_step tool response is the ONLY source of the next step's instructions.
- Call get_reference_clip when the learner wants to see the expert's reference video.
- Call go_to_step when the learner wants to jump to a specific step number. Read the step instructions from the tool response.

{step_progression}

=== COACHING RULES ===
1. Start by greeting the learner briefly, state the procedure name and total step count, then read the current step's instruction.
2. Stay on the current step. Do not read ahead or reveal future steps.
3. Answer questions using ONLY the current step's tips and warnings. If you don't have enough information, say so.
4. When advance_step returns "completed" status → congratulate the learner and summarize what they accomplished.
5. If the learner seems stuck, offer the current step's tips. Mention warnings before the learner begins a step.

=== VISUAL MONITORING ===
You can see the learner's camera feed. Actively watch what they are doing.
- If you see them making a mistake or doing something incorrectly, interject immediately with a brief correction.
- Reference what you can see to make your coaching specific (e.g., "I can see the collar is still on — keep turning it counter-clockwise").

=== CURRENT STEP ===
{current_step_text}"""

    logger.info(
        "Built system prompt for procedure '%s' (%d steps, auto_advance=%s, %d chars)",
        procedure["title"], len(procedure["steps"]), auto_advance, len(prompt),
    )
    return prompt


_STEP_PROGRESSION_AUTO = """=== STEP PROGRESSION (VISUAL DETECTION) ===
Steps advance ONLY when you visually confirm completion through the camera feed.
Do NOT advance when the learner says "done" or similar words — they may say this casually in conversation.

How to detect step completion:
- Check the "Completion criteria" for the current step. These describe exactly what the completed state looks like.
- Only call advance_step when you can clearly see that ALL completion criteria are met in the camera feed.
- Do NOT call advance_step for partial progress or ambiguous visual states.
- Before calling advance_step, tell the learner what you observed: "I can see the collar is completely off the bar — great work!"
- If you are unsure, ask the learner to confirm: "It looks like the collar might be off — can you confirm?"
- Wait for the advance_step tool response before saying anything about the next step.

If the learner asks to move on or says "done" but the completion criteria are not visually met, tell them what still needs to happen."""


_STEP_PROGRESSION_MANUAL = """=== STEP PROGRESSION ===
Steps only advance when the learner explicitly says they are done ("done", "finished", "next", "I did it").
When they do: IMMEDIATELY call advance_step. Do not say anything about the next step until you receive the tool response.
Do NOT advance steps based on what you see in the camera feed. Wait for verbal confirmation.
When they say "done", visually confirm the step looks complete before calling advance_step. If it doesn't look right, tell them what needs to be fixed."""


def _format_current_step(step: dict) -> str:
    """Format a single step's full details for embedding in the prompt."""
    text = f"Step {step['step_number']}: {step['title']}\n"
    text += f"Instruction: {step['description']}\n"
    if step.get("completion_criteria"):
        text += f"Completion criteria: {step['completion_criteria']}\n"
    if step.get("tips"):
        text += "Tips: " + "; ".join(step["tips"]) + "\n"
    if step.get("warnings"):
        text += "Warnings: " + "; ".join(step["warnings"]) + "\n"
    return text
