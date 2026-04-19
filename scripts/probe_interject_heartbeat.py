"""Empirical probe for the Retrace "Interject" feature on Gemini 3.1 Flash Live.

Answers two load-bearing questions before we commit to the heartbeat-based
design documented in memory/project_interject_feature.md:

  Q1 (trigger): does `realtimeInput.text` mid-session reliably fire a
      generation cycle on 3.1 with no preceding audio activity?
  Q2 (silence): can the model be made to emit a fully silent turn (no
      audio, no text, no tool call) when instructed to — either via plain
      prompt (V1) or via a tool-gate prompt (V2)?

If Q2's V2 silence rate is >= 80% we proceed with the Interject design.
Below 50% it's dead; we fall back to server-side REST polling.

Design: standalone SDK-path harness. Connects via
`client.aio.live.connect()` (raw API key, v1alpha) — not the ephemeral-
token WS path, which is orthogonal to the question under test. Streams a
single static neutral JPEG at 0.5 fps for the full run, fires heartbeat
text turns at a configurable cadence, and classifies each heartbeat's
response window as silent / spoke / called_interject / timed_out.

Usage:
    cd main-local-server-test
    source .venv/bin/activate
    python scripts/probe_interject_heartbeat.py                      # full run
    python scripts/probe_interject_heartbeat.py --phase silence --variant v2
    python scripts/probe_interject_heartbeat.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the repo root importable so `from config import ...` works when
# running this as `scripts/probe_interject_heartbeat.py`.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Monkey-patch websockets.asyncio.client.connect to disable keepalive pings
# before google.genai imports it. The SDK hardcodes the websockets connect
# call with no ping config; the library default of 20s interval / 20s timeout
# is too aggressive for Gemini Live — the server doesn't reliably pong on
# mostly-silent video streams, so the client tears the connection with 1011
# ("keepalive ping timeout") at ~t=35s. Retrace's iOS client uses a native
# URLSessionWebSocketTask which has different keepalive semantics, so this
# workaround is for the Python harness only — production is unaffected.
import websockets.asyncio.client as _ws_client_module  # noqa: E402
_original_ws_connect = _ws_client_module.connect


def _patched_ws_connect(*args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("ping_interval", None)
    kwargs.setdefault("ping_timeout", None)
    return _original_ws_connect(*args, **kwargs)


_ws_client_module.connect = _patched_ws_connect  # type: ignore[assignment]

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from config import FFMPEG_PATH, GEMINI_API_KEY, RETRACE_LIVE_MODEL  # noqa: E402
from services.ephemeral_token import build_live_connect_config  # noqa: E402

logger = logging.getLogger("probe_interject")

# ---------------------------------------------------------------------------
# Terminal formatting (match gemini_live_playground/runner.py)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_BOLD = "\033[1m"


def c(s: str, color: str) -> str:
    return f"{color}{s}{_RESET}" if sys.stdout.isatty() else s


# ---------------------------------------------------------------------------
# SDK compat shim — parameter name for image frames shifted across versions
# ---------------------------------------------------------------------------

_SEND_MEDIA_PARAM: str | None = None


def _detect_send_media_param() -> str:
    """Pick the right send_realtime_input media kwarg for Gemini 3.1 Live.

    The SDK's `media=` path converts to `realtime_input.media_chunks` on the
    wire, which 3.1 rejects with 1007 ("media_chunks is deprecated. Use
    audio, video, or text instead"). Prefer `video=` which maps to
    `realtime_input.video` — the accepted field for image frames on 3.1.
    """
    global _SEND_MEDIA_PARAM
    if _SEND_MEDIA_PARAM is not None:
        return _SEND_MEDIA_PARAM
    try:
        from google.genai.live import AsyncSession  # type: ignore
        sig = inspect.signature(AsyncSession.send_realtime_input)
        params = sig.parameters
    except Exception:
        _SEND_MEDIA_PARAM = "video"
        return _SEND_MEDIA_PARAM
    if "video" in params:
        _SEND_MEDIA_PARAM = "video"
    elif "media" in params:
        _SEND_MEDIA_PARAM = "media"
    else:
        _SEND_MEDIA_PARAM = "video"
    return _SEND_MEDIA_PARAM


async def send_frame(live: Any, jpeg_bytes: bytes) -> None:
    param = _detect_send_media_param()
    blob = types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
    await live.send_realtime_input(**{param: blob})


# ---------------------------------------------------------------------------
# JSONL event log
# ---------------------------------------------------------------------------


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1, encoding="utf-8")

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        rec = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": kind,
            "payload": payload,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Per-heartbeat turn classification
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    index: int
    sent_at: float
    first_event_at: float | None = None
    audio_frame_count: int = 0
    audio_byte_count: int = 0
    transcript: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    turn_complete: bool = False
    timed_out: bool = False

    @property
    def latency_ms(self) -> float | None:
        if self.first_event_at is None:
            return None
        return (self.first_event_at - self.sent_at) * 1000

    @property
    def classification(self) -> str:
        tool_names = {tc.get("name") for tc in self.tool_calls}
        spoke = bool(self.transcript.strip()) or self.audio_frame_count > 0
        if "interject" in tool_names and spoke:
            return "called_interject_AND_spoke"  # gate failure + false positive
        if "interject" in tool_names:
            return "called_interject"
        if "observe_only" in tool_names and spoke:
            return "called_observe_only_AND_spoke"  # gate failure
        if "observe_only" in tool_names:
            return "called_observe_only"
        if self.tool_calls:
            return "called_other_tool"
        if spoke:
            return "spoke"
        return "silent"


# ---------------------------------------------------------------------------
# Neutral frame extraction (first frame of the source video via ffmpeg)
# ---------------------------------------------------------------------------


def ensure_neutral_jpeg(source_video: Path, jpeg_path: Path) -> bytes:
    """Return the bytes of `jpeg_path`, extracting it from `source_video` on first run."""
    if not jpeg_path.exists():
        if not source_video.exists():
            raise FileNotFoundError(
                f"Source video not found: {source_video}. Pass --jpeg or --source-video."
            )
        jpeg_path.parent.mkdir(parents=True, exist_ok=True)
        print(c(
            f"[INFO] extracting neutral frame: {source_video.name} → {jpeg_path.name}",
            _DIM,
        ))
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", str(source_video),
            "-ss", "0.5",           # skip a beat in case frame 0 is a sync frame
            "-vframes", "1",
            "-q:v", "2",            # high quality JPEG
            str(jpeg_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract neutral frame:\n{result.stderr}"
            )
    data = jpeg_path.read_bytes()
    if not data:
        raise RuntimeError(f"Neutral JPEG is empty: {jpeg_path}")
    return data


# ---------------------------------------------------------------------------
# Shared Live-session machinery
# ---------------------------------------------------------------------------


INTERJECT_TOOL = {
    "name": "interject",
    "description": (
        "Call this when you observe a clear, persistent mistake in what the "
        "learner is doing that matches one of the listed observable errors. "
        "Do NOT call this for anything else — no acknowledgements, no check-ins, "
        "no narration. If in doubt, do not call it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "What you observed going wrong (brief, 3–10 words).",
            },
            "correction": {
                "type": "string",
                "description": (
                    "One short directive sentence to speak to the learner "
                    "(e.g., 'Stop — turn it clockwise, the other way')."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium"],
                "description": "high if unambiguous, medium if fairly sure.",
            },
        },
        "required": ["reason", "correction", "confidence"],
    },
}

# V3 companion to `interject` — the "structured silence" channel. With
# tool_config.function_calling_config.mode = ANY, the model is forced to call
# one of the declared tools on every turn. observe_only is the escape valve
# so nothing-to-report turns have a structurally-mandated silent path (we
# never speak the result) rather than leaking free-form audio.
OBSERVE_ONLY_TOOL = {
    "name": "observe_only",
    "description": (
        "Call this on every turn where you do NOT see one of the listed "
        "observable errors. This is the default path when the learner appears "
        "on track or the scene is unchanged. The server will silently ACK this "
        "call and take no action — the learner hears nothing. Use this liberally; "
        "prefer it over interject whenever uncertain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "assessment": {
                "type": "string",
                "description": (
                    "One short phrase describing what you currently see (for "
                    "server-side logging only; the learner will not hear this)."
                ),
            },
        },
        "required": ["assessment"],
    },
}


async def _frame_pump(
    live: Any,
    jpeg_bytes: bytes,
    stop_event: asyncio.Event,
    interval_s: float,
    send_lock: asyncio.Lock,
    log: EventLog,
) -> None:
    """Stream the neutral JPEG at 1/interval_s fps until stop_event is set."""
    frame_index = 0
    while not stop_event.is_set():
        try:
            async with send_lock:
                await send_frame(live, jpeg_bytes)
            frame_index += 1
            if frame_index == 1 or frame_index % 10 == 0:
                log.write("frame_sent", {"index": frame_index})
        except Exception as exc:
            log.write("frame_error", {"index": frame_index, "error": repr(exc)})
            print(c(f"[WARN] frame pump error: {exc!r}", _YELLOW))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            continue


async def _receiver(
    live: Any,
    out_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    log: EventLog,
) -> None:
    """Continuously drain responses from the Live session into `out_queue`."""
    try:
        async for response in live.receive():
            if stop_event.is_set():
                break
            await out_queue.put(response)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.write("receiver_error", {"error": repr(exc)})
        print(c(f"[WARN] receiver error: {exc!r}", _YELLOW))


def _ingest_response_into_turn(response: Any, turn: TurnResult, now: float) -> None:
    """Pull whatever is in a single `response` into the current TurnResult."""
    saw_something = False

    if getattr(response, "tool_call", None):
        for fc in response.tool_call.function_calls:
            args = dict(fc.args) if fc.args else {}
            turn.tool_calls.append({
                "id": fc.id, "name": fc.name, "args": args,
            })
            saw_something = True

    sc = getattr(response, "server_content", None)
    if sc:
        if getattr(sc, "output_transcription", None) and sc.output_transcription.text:
            turn.transcript += sc.output_transcription.text
            saw_something = True
        mt = getattr(sc, "model_turn", None)
        if mt and mt.parts:
            for part in mt.parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    turn.audio_frame_count += 1
                    turn.audio_byte_count += len(inline.data)
                    saw_something = True
                text = getattr(part, "text", None)
                if text:
                    turn.transcript += text
                    saw_something = True
        if getattr(sc, "turn_complete", False):
            turn.turn_complete = True
            saw_something = True

    if saw_something and turn.first_event_at is None:
        turn.first_event_at = now


async def _ack_tool_calls(
    live: Any,
    response: Any,
    send_lock: asyncio.Lock,
    log: EventLog,
) -> None:
    """Send an empty-ack toolResponse for every function call in this response.

    Function calling is synchronous on 3.1 — the model blocks all further
    output until we send a toolResponse. Without this, the next heartbeat's
    turn never generates (stuck waiting for the prior tool to resolve).
    """
    tc = getattr(response, "tool_call", None)
    if not tc:
        return
    responses = []
    for fc in tc.function_calls:
        # Canned ack. Real Retrace dispatcher returns step data / corrections;
        # the probe just needs the model to unstick and continue.
        responses.append(types.FunctionResponse(
            id=fc.id, name=fc.name, response={"status": "ack"},
        ))
        log.write("tool_ack", {"id": fc.id, "name": fc.name})
    async with send_lock:
        await live.send_tool_response(function_responses=responses)


async def _drain_turn(
    event_queue: asyncio.Queue,
    turn: TurnResult,
    *,
    wait_for_turn_complete_s: float,
    quiet_grace_s: float,
    live: Any,
    send_lock: asyncio.Lock,
    log: EventLog,
) -> None:
    """Collect events for one heartbeat turn.

    Strategy: wait up to `wait_for_turn_complete_s` for the first event. If
    nothing arrives in that window → silent turn. If something arrives, keep
    reading until `turn_complete` fires OR `quiet_grace_s` elapses without
    new events. Acks any tool calls as they arrive so the session doesn't
    stall waiting for a synchronous function response.
    """
    loop = asyncio.get_running_loop()
    deadline_initial = loop.time() + wait_for_turn_complete_s

    # Wait for first event
    while turn.first_event_at is None:
        remaining = deadline_initial - loop.time()
        if remaining <= 0:
            turn.timed_out = True
            return
        try:
            response = await asyncio.wait_for(event_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            turn.timed_out = True
            return
        _ingest_response_into_turn(response, turn, loop.time())
        await _ack_tool_calls(live, response, send_lock, log)
        if turn.turn_complete:
            return

    # Drain tail until turn_complete or quiet grace expires
    while not turn.turn_complete:
        try:
            response = await asyncio.wait_for(
                event_queue.get(), timeout=quiet_grace_s,
            )
        except asyncio.TimeoutError:
            return  # No more events coming
        _ingest_response_into_turn(response, turn, loop.time())
        await _ack_tool_calls(live, response, send_lock, log)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


@dataclass
class PhaseParams:
    label: str
    system_prompt: str
    tools: list[dict]
    heartbeats: list[float]        # seconds from phase start
    frame_interval_s: float = 2.0  # 0.5 fps
    wait_for_first_event_s: float = 5.0
    quiet_grace_s: float = 1.5
    # If True, lock the session into tool-only output via
    # tool_config.function_calling_config.mode=ANY. The model cannot emit
    # free-form audio/text — it MUST call one of the declared tools per turn.
    force_tool_mode: bool = False


async def run_phase(params: PhaseParams, jpeg_bytes: bytes, model: str, log: EventLog) -> list[TurnResult]:
    log.write("phase_start", {
        "label": params.label,
        "heartbeats": params.heartbeats,
        "frame_interval_s": params.frame_interval_s,
        "tool_count": len(params.tools),
    })
    print(c(f"\n=== PHASE: {params.label} ({len(params.heartbeats)} heartbeats) ===", _BOLD))

    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1alpha"},
    )
    cfg = build_live_connect_config(
        system_instruction=params.system_prompt,
        tools=params.tools,
        voice="Puck",
    )
    # Harness sends no audio input — disable the audio-only transcription
    # configs so they don't hang around as unused state on 3.1.
    cfg.input_audio_transcription = None

    if params.force_tool_mode:
        allowed = [t["name"] for t in params.tools]
        cfg.tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=allowed,
            ),
        )
        log.write("tool_mode_forced", {"mode": "ANY", "allowed": allowed})
        print(c(f"[INFO] tool_config.mode=ANY allowed={allowed}", _DIM))

    send_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    event_queue: asyncio.Queue = asyncio.Queue()
    turns: list[TurnResult] = []

    try:
        async with client.aio.live.connect(model=model, config=cfg) as live:
            log.write("live_connected", {"model": model})
            print(c(f"[INFO] connected to {model}", _DIM))

            receiver = asyncio.create_task(
                _receiver(live, event_queue, stop_event, log),
                name="receiver",
            )
            pump = asyncio.create_task(
                _frame_pump(live, jpeg_bytes, stop_event, params.frame_interval_s, send_lock, log),
                name="frame_pump",
            )

            loop = asyncio.get_running_loop()
            phase_start = loop.time()

            # Drain anything that arrives during setup (e.g., initial turn_complete
            # if the model decides to speak on connection, which would be unusual
            # without a first turn).
            await asyncio.sleep(1.0)
            while not event_queue.empty():
                drained = await event_queue.get()
                log.write("pre_heartbeat_event", {"type": type(drained).__name__})

            for i, t_offset in enumerate(params.heartbeats, start=1):
                target_t = phase_start + t_offset
                sleep_s = target_t - loop.time()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                # Drain any trailing events from the prior turn (e.g., late
                # usage_metadata after turn_complete) so they don't get
                # mis-attributed to this heartbeat's drain.
                drained_stale = 0
                while not event_queue.empty():
                    try:
                        event_queue.get_nowait()
                        drained_stale += 1
                    except asyncio.QueueEmpty:
                        break
                if drained_stale:
                    log.write("stale_events_drained", {"heartbeat_index": i, "count": drained_stale})

                sent_at = loop.time()
                async with send_lock:
                    await live.send_realtime_input(text="[check]")
                log.write("heartbeat_sent", {"index": i, "t_offset_s": t_offset})
                print(c(f"[HB {i}/{len(params.heartbeats)}] sent at t={t_offset:.1f}s", _CYAN))

                turn = TurnResult(index=i, sent_at=sent_at)
                await _drain_turn(
                    event_queue, turn,
                    wait_for_turn_complete_s=params.wait_for_first_event_s,
                    quiet_grace_s=params.quiet_grace_s,
                    live=live,
                    send_lock=send_lock,
                    log=log,
                )
                turns.append(turn)
                log.write("heartbeat_result", {
                    "index": i,
                    "classification": turn.classification,
                    "latency_ms": turn.latency_ms,
                    "audio_frames": turn.audio_frame_count,
                    "audio_bytes": turn.audio_byte_count,
                    "transcript": turn.transcript,
                    "tool_calls": turn.tool_calls,
                    "turn_complete": turn.turn_complete,
                    "timed_out": turn.timed_out,
                })

                # Human-readable line
                klass = turn.classification
                color = {
                    "silent": _DIM,
                    "spoke": _YELLOW,
                    "called_interject": _MAGENTA,
                    "called_observe_only": _GREEN,
                    "called_interject_AND_spoke": _RED,
                    "called_observe_only_AND_spoke": _RED,
                    "called_other_tool": _MAGENTA,
                }.get(klass, _RESET)
                lat = f"{turn.latency_ms:.0f}ms" if turn.latency_ms is not None else "—"
                extra = ""
                if turn.transcript:
                    snippet = turn.transcript.strip().replace("\n", " ")
                    if len(snippet) > 90:
                        snippet = snippet[:87] + "..."
                    extra = f' "{snippet}"'
                if turn.tool_calls:
                    names = ",".join(tc["name"] for tc in turn.tool_calls)
                    extra += f" [tools={names}]"
                print(c(
                    f"  → {klass:18s} latency={lat:>7s} "
                    f"audio={turn.audio_frame_count}f/{turn.audio_byte_count}B"
                    f"{extra}",
                    color,
                ))

            stop_event.set()
            await asyncio.sleep(0.1)
            for task in (pump, receiver):
                task.cancel()
            for task in (pump, receiver):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    except Exception as exc:
        log.write("phase_error", {"label": params.label, "error": repr(exc)})
        print(c(f"[ERROR] phase {params.label} failed: {exc!r}", _RED))
        raise
    finally:
        log.write("phase_end", {"label": params.label, "turns": len(turns)})

    return turns


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    p = Path(__file__).parent / "prompts" / name
    return p.read_text(encoding="utf-8")


def summarize_trigger(turns: list[TurnResult]) -> None:
    fired = len(turns)
    responded = sum(1 for t in turns if t.first_event_at is not None)
    lats = [t.latency_ms for t in turns if t.latency_ms is not None]
    median_lat = sorted(lats)[len(lats) // 2] if lats else None
    audio = sum(1 for t in turns if t.audio_frame_count > 0)
    tool = sum(1 for t in turns if t.tool_calls)

    print(c("\n=== PHASE 1 TRIGGER — SUMMARY ===", _BOLD))
    print(f"  Heartbeats fired:           {fired}")
    print(f"  Produced any response:      {responded}/{fired} "
          f"({100*responded/fired:.0f}%)" if fired else "  (no data)")
    if median_lat is not None:
        print(f"  Median time-to-first-event: {median_lat:.0f} ms")
    print(f"  Responses with audio:       {audio}")
    print(f"  Responses with tool calls:  {tool}")
    pass_gate = responded >= max(1, int(0.75 * fired))
    verdict = c("PASS", _GREEN) if pass_gate else c("FAIL", _RED)
    print(f"  Trigger gate (>=75% respond): {verdict}")
    if not pass_gate:
        print(c(
            "  ⚠ realtimeInput.text may not be reliably triggering generation on 3.1. "
            "The whole heartbeat design is at risk. Investigate before proceeding.",
            _YELLOW,
        ))


def summarize_silence(variant: str, turns: list[TurnResult]) -> dict:
    n = len(turns)
    by_class: dict[str, int] = {}
    for t in turns:
        by_class[t.classification] = by_class.get(t.classification, 0) + 1

    silent = by_class.get("silent", 0)
    spoke = by_class.get("spoke", 0)
    called_interject = by_class.get("called_interject", 0)
    called_observe_only = by_class.get("called_observe_only", 0)
    interject_and_spoke = by_class.get("called_interject_AND_spoke", 0)
    observe_and_spoke = by_class.get("called_observe_only_AND_spoke", 0)
    called_other = by_class.get("called_other_tool", 0)

    # For V3 (tool_config.mode=ANY), "silence" is called_observe_only —
    # structurally-mandated silent channel. For V1/V2, silence is the natural
    # no-output turn.
    if variant == "v3":
        effective_silent = called_observe_only
        spoke_any = spoke + interject_and_spoke + observe_and_spoke
        false_positive_corrections = called_interject + interject_and_spoke
        silence_rate = effective_silent / n if n else 0
    else:
        effective_silent = silent
        spoke_any = spoke
        false_positive_corrections = called_interject
        silence_rate = effective_silent / n if n else 0

    print(c(f"\n=== PHASE 2 SILENCE ({variant.upper()}) — SUMMARY ===", _BOLD))
    print(f"  Heartbeats:                  {n}")
    if variant == "v3":
        print(f"  called_observe_only (✓):     {called_observe_only}")
        print(f"  called_interject (FP):       {called_interject}")
        if interject_and_spoke:
            print(c(f"  interject+spoke (GATE FAIL): {interject_and_spoke}", _RED))
        if observe_and_spoke:
            print(c(f"  observe+spoke   (GATE FAIL): {observe_and_spoke}", _RED))
        if spoke:
            print(c(f"  spoke only       (GATE FAIL): {spoke}", _RED))
        if silent:
            print(c(f"  silent (no tool - mode=ANY violation): {silent}", _YELLOW))
    else:
        print(f"  silent:                      {silent}")
        print(f"  spoke (audio/text):          {spoke}")
        print(f"  called_interject (FP):       {called_interject}")
    if called_other:
        print(f"  called_other_tool:           {called_other}")
    print(f"  effective silence rate:      {100*silence_rate:.1f}%")
    return {
        "variant": variant,
        "n": n,
        "silent": silent,
        "spoke": spoke_any,
        "called_interject": called_interject,
        "called_observe_only": called_observe_only,
        "interject_and_spoke": interject_and_spoke,
        "observe_and_spoke": observe_and_spoke,
        "silence_rate": silence_rate,
        "effective_silent": effective_silent,
    }


def print_final_verdict(silence_results: list[dict]) -> None:
    if not silence_results:
        return
    print(c("\n=== FINAL DECISION GATE ===", _BOLD))
    best = max(silence_results, key=lambda r: r["silence_rate"])
    rate = best["silence_rate"]
    print(f"  Best variant: {best['variant'].upper()} with "
          f"{100*rate:.1f}% silence rate ({best['silent']}/{best['n']})")
    if rate >= 0.80:
        print(c(
            "  VIABLE — Interject design can proceed (silence rate ≥ 80%).",
            _GREEN,
        ))
    elif rate >= 0.50:
        print(c(
            "  MARGINAL — silence rate 50–80%. Iterate on prompt wording or "
            "lower heartbeat cadence before committing.",
            _YELLOW,
        ))
    else:
        print(c(
            "  DEAD — silence rate < 50%. The heartbeat design cannot keep "
            "the coach quiet. Fall back to server-side REST polling.",
            _RED,
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Empirical probe for Interject heartbeat + silence on Gemini 3.1 Flash Live.",
    )
    p.add_argument(
        "--phase",
        choices=["trigger", "silence", "both"],
        default="both",
    )
    p.add_argument(
        "--variant",
        choices=["v1", "v2", "v3", "both"],
        default="both",
        help=(
            "Which silence prompt variants to run. v1=plain, v2=prompt-only "
            "tool gate, v3=structural tool gate (tool_config.mode=ANY + "
            "observe_only escape valve). 'both' runs v1+v2."
        ),
    )
    p.add_argument(
        "--heartbeat-interval", type=float, default=15.0,
        help="Seconds between heartbeats in the silence phase (default 15).",
    )
    p.add_argument(
        "--silence-duration", type=float, default=120.0,
        help="Total seconds for the silence phase (default 120).",
    )
    p.add_argument(
        "--jpeg", type=Path,
        default=BACKEND_ROOT / "test_data" / "neutral.jpg",
        help="Path to the neutral JPEG (auto-generated from --source-video if missing).",
    )
    p.add_argument(
        "--source-video", type=Path,
        default=BACKEND_ROOT / "test_data" / "mac-input-reset.MOV",
        help="Source video to extract neutral.jpg from when --jpeg is missing.",
    )
    p.add_argument(
        "--model", default=RETRACE_LIVE_MODEL,
        help=f"Gemini Live model id (default {RETRACE_LIVE_MODEL}).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="JSONL event log path (default: tmp/heartbeat_probe_<ts>.jsonl).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    if not GEMINI_API_KEY:
        print(c("GEMINI_API_KEY not set — add to .env or export.", _RED))
        return 2

    jpeg_bytes = ensure_neutral_jpeg(args.source_video, args.jpeg)

    output = args.output or (
        BACKEND_ROOT / "tmp" / f"heartbeat_probe_{int(time.time())}.jsonl"
    )
    log = EventLog(output)
    log.write("run_start", {
        "model": args.model,
        "phase": args.phase,
        "variant": args.variant,
        "heartbeat_interval_s": args.heartbeat_interval,
        "silence_duration_s": args.silence_duration,
        "jpeg_path": str(args.jpeg),
        "jpeg_bytes": len(jpeg_bytes),
        "send_media_param": _detect_send_media_param(),
    })
    print(c(
        f"[INFO] model={args.model} jpeg={args.jpeg.name} "
        f"({len(jpeg_bytes):,} B) log={output}",
        _DIM,
    ))

    silence_results: list[dict] = []

    try:
        if args.phase in ("trigger", "both"):
            # Fire heartbeats at 5, 15, 30, 45s (relative to phase start)
            params = PhaseParams(
                label="trigger",
                system_prompt=_load_prompt("interject_trigger.txt"),
                tools=[],  # no tools for trigger phase
                heartbeats=[5.0, 15.0, 30.0, 45.0],
            )
            turns = await run_phase(params, jpeg_bytes, args.model, log)
            summarize_trigger(turns)
            # Quick sanity: if no heartbeat produced any response, don't bother with silence
            any_response = any(t.first_event_at is not None for t in turns)
            if not any_response and args.phase == "both":
                print(c(
                    "\n[ABORT] No heartbeat in the trigger phase produced a response. "
                    "Skipping silence phase — something is wrong with the trigger "
                    "mechanism before we can measure silence. Inspect the log.",
                    _RED,
                ))
                return 1

        if args.phase in ("silence", "both"):
            # Evenly spaced heartbeats across the silence duration
            n_beats = max(1, int(args.silence_duration // args.heartbeat_interval))
            beats = [args.heartbeat_interval * (i + 1) for i in range(n_beats)]

            variants: list[str]
            if args.variant == "both":
                variants = ["v1", "v2"]
            else:
                variants = [args.variant]

            for v in variants:
                prompt_file = f"interject_silence_{v}.txt"
                if v == "v1":
                    tools = []
                    force = False
                elif v == "v2":
                    tools = [INTERJECT_TOOL]
                    force = False
                elif v == "v3":
                    tools = [INTERJECT_TOOL, OBSERVE_ONLY_TOOL]
                    force = True
                else:
                    tools = []
                    force = False
                params = PhaseParams(
                    label=f"silence_{v}",
                    system_prompt=_load_prompt(prompt_file),
                    tools=tools,
                    heartbeats=beats,
                    force_tool_mode=force,
                )
                turns = await run_phase(params, jpeg_bytes, args.model, log)
                result = summarize_silence(v, turns)
                silence_results.append(result)

        print_final_verdict(silence_results)
        log.write("run_end", {"silence_results": silence_results})
        return 0
    finally:
        log.close()
        print(c(f"\n[INFO] event log: {output}", _DIM))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        print(c("\n[INFO] interrupted", _DIM))
        return 130


if __name__ == "__main__":
    sys.exit(main())
