"""Render an InterviewLog into the prompt section the video extractor reads.

Kept separate from gemini_processor.py so the extraction prompt stays a
constant there and this stays testable as a pure function.
"""

from __future__ import annotations

from models.interview import InterviewLog


def format_interview_notes(log: InterviewLog | None) -> str:
    """The `=== EXPERT INTERVIEW NOTES ===` block, or "" when there is nothing
    worth telling the extractor."""
    if log is None:
        return ""
    facts = list(log.facts)
    answered = [q for q in log.qa if q.answer]
    windows = [w for w in log.ai_speech_windows if len(w) == 2 and w[1] > w[0]]
    if not facts and not answered and not windows and not log.briefing:
        return ""

    lines = ["", "=== EXPERT INTERVIEW NOTES ===",
             "While recording, an AI assistant occasionally asked the expert short questions. "
             "What the expert SAID in answer is below — treat it as authoritative over anything "
             "you infer from the image, and never invent a value the expert did not state."]
    if log.briefing:
        lines.append(f"Expert's pre-recording briefing: {log.briefing}")

    if facts:
        lines.append("Timestamped facts (t = seconds into the video; null = said before/after recording):")
        for f in facts:
            t = f"t={f.t:.1f}" if f.t is not None else "t=null"
            part = f" (part: {f.part})" if f.part else ""
            lines.append(f"- [{t}] {f.kind}: {f.value}{part}")
    elif answered:
        lines.append("Questions the assistant asked and the expert's answers:")
        for q in answered:
            t = f"t={q.t_ask:.1f}" if q.t_ask is not None else "t=null"
            lines.append(f"- [{t}] Q: {q.question}  A: {q.answer}")

    if windows:
        spans = ", ".join(f"[{a:.1f}, {b:.1f}]" for a, b in windows[:40])
        lines.append(f"The assistant's own voice is audible at these times (seconds) — it is NOT the expert and "
                     f"is not narration; ignore it: {spans}")

    lines += [
        "How to fold these in:",
        "- Specs/sizes/settings go into the `description` of the step whose time range contains t, AND into the "
        "matching parts-glossary `appearance` when they identify a part.",
        "- judgment / warning / tip / alternative facts go into that step's `tips` or `warnings` (procedure-level "
        "facts with t=null go on the most relevant step, or the first step if truly general).",
        "- Prefer the expert's exact words for numbers and names.",
    ]
    return "\n".join(lines)
