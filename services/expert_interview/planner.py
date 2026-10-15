"""The interview planner — one Flash call per observe tick.

Modelled on services/step_oracle.py: narrow question, structured output,
hard timeout, zero thinking budget, never raises (None = stay silent this
tick). Inputs per tick: the latest camera frame (optional), a ~5 s audio
chunk of the expert's narration (optional; the phone keeps a PCM ring and
the planner transcribes it — see docs/expert-interviewer-design.md §4.3 for
why narration transcription does NOT come from Gemini Live), and the ledger.

The model proposes; services/expert_interview/policy.py decides.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import struct
import wave

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, RETRACE_PLANNER_MODEL
from models.interview import PlannerDecision

logger = logging.getLogger(__name__)

# Bounded so a slow tick degrades to "nothing this tick" rather than piling
# up behind the next one. Observe ticks arrive every ~5 s from the phone, so
# the single-in-flight guard in the router covers the overlap. NOTE: the API
# rejects any manual deadline under 10 s ("Minimum allowed deadline is 10s",
# verified 2026-10-15) — this is the floor, not a tuning choice.
TIMEOUT_MS = 10_000

# Audio chunk RMS (Int16 scale) below which the chunk is treated as silence
# and — if the frame also didn't change — the model call is skipped.
SILENCE_RMS = 300


_SYSTEM_PROMPT = """You are the planner behind a quiet production assistant that sits beside an EXPERT recording a hands-on how-to video. The recording will become a step-by-step guide for a LEARNER who has never done this. Experts under-explain: they skip sizes, settings, quantities, which part, why, how they know it's right, and what goes wrong. Your job is to notice those gaps — and to notice when the expert's own narration already fills them.

You receive, per tick:
- Optionally the CURRENT camera frame (what the expert sees right now).
- Optionally a short AUDIO chunk of the expert's narration from the last few seconds.
- The LEDGER: the task briefing, what's already been asked/answered, and open gaps.

Return JSON with:
- transcript_delta: what the expert said in the audio chunk, as plain text. Empty if silent or no audio. Do not paraphrase; do not include the assistant's own voice if you hear it.
- filled: topic_keys from the ledger that this chunk's narration just answered. Listen first, ask second.
- new_gaps: concrete gaps you noticed THIS tick that are not already in the ledger. Each has a stable topic_key (lowercase snake_case naming the object and attribute, e.g. "lug_nut_socket_size", "torque_done_cue"), a priority, a question phrased for speech (one short sentence), and time_sensitive.
- propose_ask: the single topic_key you would ask RIGHT NOW, or null. Only propose when the question is time_sensitive and asking later would lose information.
- confidence: 0-1, your confidence that propose_ask is a real, specific, unanswered gap.
- addressed_assistant: true if the expert spoke TO the assistant ("hey", "note that", "write down", "question for you").
- requested_quiet: true if the expert asked for no more questions.

ASK-WORTHY (the non-visual):
- spec: sizes, torque, quantities, settings, temperatures, durations, part numbers, which direction.
- part: which of several similar parts/tools is the right one and how to tell.
- judgment: how the expert knows it's done/tight/seated/level when the camera can't show it (feel, sound, resistance).
- why: the reason behind a non-obvious choice, when a learner would plausibly do it differently.
- safety: anything that could hurt the learner or break the thing.
- tip / alternative: what to do when a tool is missing, what commonly goes wrong and how to recover.

NEVER ASK ABOUT what the camera already captures: colors, what they're holding, what they're doing, the order of steps. Never ask what they just said. Never ask a yes/no question.

time_sensitive is true ONLY when: the object in question is in frame right now and will leave, OR the step is about to become irreversible (gluing, cutting, tightening to final, closing up), OR it's safety. Everything else is false — it is just as good in the post-recording debrief and costs the expert no flow.

A gap must name a specific object, quantity, or decision that appears in the frame or the transcript. If you cannot name it, it is not a gap. Silence is the default; proposing a question is the exception. When in doubt: new_gaps may include it with time_sensitive=false, but propose_ask must be null.

Output JSON matching the schema."""


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ---- Audio helpers ----------------------------------------------------------------------------


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Wrap raw little-endian Int16 PCM in a WAV container (generateContent
    accepts audio/wav; it does not take raw PCM the way Live does)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def is_wav(data: bytes) -> bool:
    return len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def pcm16_rms(pcm: bytes) -> float:
    """Root-mean-square of Int16 samples (0 for empty)."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return (sum(s * s for s in samples) / n) ** 0.5


def wav_payload_rms(data: bytes) -> float:
    """RMS of a WAV file's samples, or of raw PCM when not a WAV."""
    if is_wav(data):
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                if w.getsampwidth() != 2:
                    return SILENCE_RMS + 1  # unknown width: don't skip on it
                return pcm16_rms(w.readframes(w.getnframes()))
        except wave.Error:
            return SILENCE_RMS + 1
    return pcm16_rms(data)


def frame_digest(jpeg: bytes | None) -> str | None:
    """Cheap change detector for the skip-if-idle gate. A full perceptual hash
    would be better; the JPEG bytes differ on any real motion and the audio
    gate covers the rest."""
    if not jpeg:
        return None
    return hashlib.blake2b(jpeg, digest_size=8).hexdigest()


# ---- The call ----------------------------------------------------------------------------------


def _ledger_block(session: dict) -> str:
    gaps = session["gaps"]
    lines = []
    for g in gaps.values():
        lines.append(f"- [{g['status']}] {g['topic_key']} ({g['priority']}): {g['question']}"
                     + (f" → {g['answer']}" if g.get("answer") else ""))
    recent = session["transcript"][-8:]
    recent_txt = "\n".join(f"  ({'t=%.0fs' % e['t'] if e.get('t') is not None else 'pre-roll'}) {e['text']}"
                           for e in recent) or "  (nothing yet)"
    return (
        f"PHASE: {session['phase']}\n"
        f"BRIEFING: {session.get('briefing') or '(none)'}\n"
        f"LIVE ASKS USED: {session['budget']['live_asks']}  QUIET MODE: {session['quiet_mode']}\n"
        f"LEDGER ({len(gaps)} gaps):\n" + ("\n".join(lines) if lines else "  (empty)") + "\n"
        f"RECENT NARRATION:\n{recent_txt}"
    )


async def observe(
    session: dict,
    *,
    frame_jpg: bytes | None,
    audio: bytes | None,
    sample_rate: int = 16_000,
    model: str | None = None,
) -> PlannerDecision | None:
    """Run one planner tick. Returns a PlannerDecision, or None when the call
    failed or was skipped as idle. Never raises."""
    parts: list[types.Part] = []
    if frame_jpg:
        parts.append(types.Part.from_text(text="CURRENT FRAME (expert's camera right now):"))
        parts.append(types.Part.from_bytes(data=frame_jpg, mime_type="image/jpeg"))
    if audio:
        wav = audio if is_wav(audio) else pcm16_to_wav(audio, sample_rate)
        parts.append(types.Part.from_text(text="AUDIO (expert's narration, last few seconds):"))
        parts.append(types.Part.from_bytes(data=wav, mime_type="audio/wav"))
    parts.append(types.Part.from_text(text=_ledger_block(session)))

    def _call():
        return _get_client().models.generate_content(
            model=model or RETRACE_PLANNER_MODEL,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=PlannerDecision,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=TIMEOUT_MS),
            ),
        )

    try:
        response = await asyncio.to_thread(_call)
    except Exception as exc:
        logger.warning("interview planner: call failed (%s)", exc)
        return None

    text = getattr(response, "text", None)
    if not text:
        logger.warning("interview planner: empty response")
        return None
    try:
        return PlannerDecision.model_validate_json(text)
    except Exception as exc:
        logger.warning("interview planner: parse failed (%s); raw=%r", exc, text[:200])
        return None


def should_skip_idle(session: dict, *, frame_jpg: bytes | None, audio: bytes | None) -> bool:
    """True when neither the scene nor the audio changed enough to be worth a
    model call. Updates the session's frame digest as a side effect."""
    digest = frame_digest(frame_jpg)
    frame_changed = digest is not None and digest != session.get("_last_frame_digest")
    if digest is not None:
        session["_last_frame_digest"] = digest
    audio_active = bool(audio) and wav_payload_rms(audio) >= SILENCE_RMS
    return not frame_changed and not audio_active


def debug_dump(decision: PlannerDecision | None) -> str:
    return json.dumps(decision.model_dump() if decision else None, ensure_ascii=False)[:400]
