"""Turn the interview's Q/A + narration transcript into structured facts.

Runs once at /finalize (and again, defensively, at extraction time if the
uploaded log carries Q/A but no facts). One Flash call with a structured
schema; never raises — an empty list just means extraction falls back to the
raw Q/A text, which the notes formatter also includes.
"""

from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import GEMINI_API_KEY, RETRACE_PLANNER_MODEL
from models.interview import Fact, InterviewLog

logger = logging.getLogger(__name__)

TIMEOUT_MS = 20_000


class _FactsOutput(BaseModel):
    facts: list[Fact]


_SYSTEM_PROMPT = """You convert an expert's answers (and their narration) into atomic facts for a how-to guide. Each fact is one thing a learner needs that the video alone would not show: a size, setting, quantity, part identity, reason, done-cue, warning, tip, or alternative.

Rules:
- One fact per entry. Keep the expert's numbers and names verbatim ("19 mm", not "about 19").
- `kind`: spec | tool | part | why | judgment | warning | tip | alternative.
- `part`: the part or tool the fact is about, in the expert's own words, or null.
- `t`: copy the question's t_ask when the fact comes from a live answer; copy the narration line's t when it comes from narration; null only for pre-roll / debrief / unknown. Timestamps are how the guide builder attaches a fact to the right step — never drop one that is available.
- `source`: live | debrief | narration | preroll — where the fact came from.
- Never invent. If an answer was vague ("pretty tight"), record it as a judgment fact in the expert's words rather than inventing a number.
- Skip acknowledgements, filler, and anything already obvious from watching.

Output JSON matching the schema."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _render(log: InterviewLog) -> str:
    lines = [f"BRIEFING: {log.briefing or '(none)'}", "", "Q/A:"]
    for q in log.qa:
        t = f"t_ask={q.t_ask:.1f}s" if q.t_ask is not None else "t_ask=null"
        lines.append(f"- [{q.phase} {t} {q.outcome}] Q: {q.question}")
        lines.append(f"    A: {q.answer or '(no answer)'}")
    lines += ["", "NARRATION (planner-transcribed):"]
    for e in log.transcript[-60:]:
        t = f"t={e.get('t'):.0f}s" if e.get("t") is not None else "t=null"
        lines.append(f"- ({t}) {e.get('text', '')}")
    return "\n".join(lines)


async def extract_facts(log: InterviewLog, *, model: str | None = None) -> list[Fact]:
    if not log.qa and not log.transcript:
        return []

    def _call():
        return _get_client().models.generate_content(
            model=model or RETRACE_PLANNER_MODEL,
            contents=[types.Content(parts=[types.Part.from_text(text=_render(log))])],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_FactsOutput,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )

    try:
        response = await asyncio.to_thread(_call)
        parsed = _FactsOutput.model_validate_json(response.text)
    except Exception as exc:
        logger.warning("interview facts: extraction failed (%s)", exc)
        return []
    logger.info("interview facts: %d facts from %d Q/A", len(parsed.facts), len(log.qa))
    return parsed.facts
