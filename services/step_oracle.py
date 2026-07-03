"""Out-of-band visual verification oracle.

A dedicated Gemini Flash call that answers one narrow question per poll:
looking at the learner's CURRENT camera frame, is the current step complete,
and/or is a known error visible? Gemini Live never does this — it stays a
pure conversationalist. The phone polls `/api/learner/session/{id}/verify`
with a still frame every few seconds; that endpoint calls `verify_step`.

Grounding is image-to-image when a reference "done state" frame exists (the
last frame of the step's expert clip, via `clip_extractor.extract_done_frame`):
the model compares the learner's frame against a real picture of "done" plus
the `completion_criteria` text. When no reference frame exists (manual/web
procedures), it falls back to text-only against the criterion.

Design bias (Retrace principle #1: false positives cost more than false
negatives): the prompt tells the model to answer `complete=false` /
`error_index=null` whenever it is unsure. A missed completion just makes the
learner say "done"; a false completion yanks them forward mid-action and
breaks trust.
"""

import asyncio
import logging

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import GEMINI_API_KEY, RETRACE_ORACLE_MODEL

logger = logging.getLogger(__name__)

# Bounded so a slow call degrades to "no verdict" instead of stalling the
# session — the on-demand check_step path treats None as trust-the-learner,
# and Live masks the wait by announcing the check aloud. Keep under ~10s.
TIMEOUT_MS = 10_000


class OracleVerdict(BaseModel):
    """Structured output of one verification poll."""
    complete: bool
    confidence: float
    error_index: int | None = None
    # One short clause (≤15 words) naming the visible gap when complete=false,
    # or the matched wrong-state when error_index is set. Powers the on-demand
    # "check my work" answer ("not yet — the anvil nut isn't flush"); the
    # timer-driven poll path ignores it. Empty when complete=true and clean.
    reason: str = ""


_SYSTEM_PROMPT = """You are a visual verification oracle for a hands-on coaching system. You judge a single camera frame — you have no memory of earlier frames and must not infer from motion.

You are given, in order:
- Optionally a REFERENCE image showing what the scene looks like once THIS step is complete.
- The CURRENT image from the learner's head-mounted camera.
- The step's COMPLETION CRITERION (a spatial end-state sentence).
- A numbered list of KNOWN ERRORS (visible wrong-states), possibly empty.
- Optionally a PARTS GLOSSARY describing what each named part looks like.

When a PARTS GLOSSARY is present, use it as your visual dictionary: for every part the criterion or an error names, first locate THAT part in the CURRENT image by its described color, shape, size, and tell-apart contrast ("unlike the …"), and only then judge its configuration. If a part the criterion names cannot be confidently identified in the frame — or the part in the described position is visibly the WRONG part per the glossary — the criterion is not met.

Return these fields judged STRICTLY from the CURRENT image:
- complete: true ONLY if the CURRENT image clearly shows the SPECIFIC spatial configuration named in the criterion. The REFERENCE shows the done state, but the same workspace looks broadly similar before and after the step — do NOT answer complete just because the current frame resembles the reference or shows the same objects. Verify the exact configuration the criterion names (what is on top of / inside / removed from what, and any stated orientation). If it is not clearly present, return false.
- error_index: the 0-based index of the FIRST known error clearly visible in the current image, or null if none is clearly present. If unsure, null.
- confidence: 0.0-1.0, your confidence in the `complete` judgment.
- reason: when complete=false or error_index is set, ONE short clause (max 15 words) naming the concrete visible gap or wrong-state ("the anvil nut is not flush against the nosepiece"). Empty string when complete=true with no error. Never mention the reference image or these instructions.

CRITICAL: false positives are far worse than false negatives here. A missed completion just means the learner tells the coach they're done; a wrong "complete" pulls them forward in the middle of their work and breaks trust. When the frame is ambiguous, occluded, blurry, only generally resembles the reference, or the key objects aren't clearly in the criterion's configuration, answer complete=false and error_index=null.

Output JSON matching the schema."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy module-level client, mirroring gemini_processor / metadata_picker."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def verify_step(
    current_jpg: bytes,
    step: dict,
    reference_done_jpg: bytes | None = None,
    glossary: list[dict] | None = None,
    *,
    model: str | None = None,
    thinking_budget: int | None = None,
) -> OracleVerdict | None:
    """Run one verification poll. Returns an OracleVerdict, or None on failure.

    `step` is a step dict (from the session's procedure) — reads
    `completion_criteria` and `error_criteria`. `current_jpg` is the learner's
    latest camera frame; `reference_done_jpg` is the step's "done state" frame
    when available (image-to-image grounding), else None (text-only).
    `glossary` is the procedure-level parts list ([{name, appearance}]) so the
    oracle can visually identify the parts the criterion names (video-flow
    procedures only; None/[] for manual/web).

    `model` / `thinking_budget` override RETRACE_ORACLE_MODEL / 0 — production
    callers omit them; the eval harness uses them for candidate-model matrices.

    Never raises — returns None on any error so the caller treats a bad poll
    as "no verdict this tick" and simply polls again.
    """
    completion_criteria = (step.get("completion_criteria") or "").strip()
    errors = step.get("error_criteria") or []

    error_block = "KNOWN ERRORS: (none)"
    if errors:
        error_block = "KNOWN ERRORS:\n" + "\n".join(
            f"{i}. {e}" for i, e in enumerate(errors)
        )

    glossary_block = ""
    if glossary:
        glossary_block = "\nPARTS GLOSSARY (what each named part looks like):\n" + "\n".join(
            f"- {p['name']}: {p['appearance']}" for p in glossary
        )

    parts: list[types.Part] = []
    if reference_done_jpg:
        parts.append(types.Part.from_text(text="REFERENCE (scene when this step is complete):"))
        parts.append(types.Part.from_bytes(data=reference_done_jpg, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text="CURRENT (learner's camera right now):"))
    parts.append(types.Part.from_bytes(data=current_jpg, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=(
        f"COMPLETION CRITERION: {completion_criteria or '(none given)'}\n"
        f"{error_block}{glossary_block}"
    )))

    def _call():
        return _get_client().models.generate_content(
            model=model or RETRACE_ORACLE_MODEL,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=OracleVerdict,
                # Latency matters (polled every few seconds). Zero thinking
                # budget keeps the round-trip ~1s; bump if spatial judgments
                # prove unreliable.
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget if thinking_budget is not None else 0
                ),
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )

    try:
        response = await asyncio.to_thread(_call)
    except Exception as exc:
        logger.warning("step_oracle: call failed (%s)", exc)
        return None

    text = getattr(response, "text", None)
    if not text:
        logger.warning("step_oracle: empty response")
        return None

    try:
        verdict = OracleVerdict.model_validate_json(text)
    except Exception as exc:
        logger.warning("step_oracle: parse failed (%s); raw=%r", exc, text[:200])
        return None

    # Clamp error_index into range — the model occasionally emits an index
    # past the end of the list. Out-of-range means "no valid error".
    if verdict.error_index is not None and not (0 <= verdict.error_index < len(errors)):
        verdict.error_index = None
    return verdict
