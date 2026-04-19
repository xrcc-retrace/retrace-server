"""Raw-WebSocket probe of Gemini 3.1 Flash Live with tool_config.mode=ANY.

The Python SDK's LiveConnectConfig has `extra='forbid'` and no `tool_config`
field, so the SDK path is blocked. But the wire protocol accepts it. This
probe:

  1. Hand-crafts a BidiGenerateContentSetup with toolConfig.
     functionCallingConfig.mode=ANY and allowed_function_names=[
     interject, observe_only].
  2. Streams a static neutral JPEG at 0.5 fps.
  3. Fires N heartbeat `realtimeInput.text="[check]"` turns at the given
     cadence, acking each tool call so the synchronous call doesn't stall
     the session.
  4. Classifies each turn as called_observe_only / called_interject /
     gate_failure (audio or text emitted) / silent (no tool call — which
     should be impossible under mode=ANY).

Usage:
    python scripts/probe_live_tool_config.py                   # 8 beats @ 5s
    python scripts/probe_live_tool_config.py --beats 12 --interval 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import websockets  # type: ignore

from config import GEMINI_API_KEY

MODEL_DEFAULT = "gemini-3.1-flash-live-preview"


def _ws_url(model: str) -> str:
    return (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1alpha."
        "GenerativeService.BidiGenerateContent"
        f"?key={GEMINI_API_KEY}"
    )


SYSTEM_PROMPT = (
    "You are a silent visual coach watching a learner through a camera feed.\n\n"
    "=== CURRENT STEP ===\n"
    "STEP: Turn the valve clockwise until the indicator aligns with the green mark.\n"
    "OBSERVABLE ERRORS:\n"
    "- The learner is rotating the valve counterclockwise instead of clockwise.\n"
    "- The learner is gripping the wrong object (not the valve handle).\n"
    "- The valve has been rotated past the green mark.\n\n"
    "=== OUTPUT PROTOCOL ===\n"
    "On every turn you MUST call exactly one tool.\n"
    "- observe_only(assessment): the default. Call when the learner is on track, "
    "nothing is happening, the scene doesn't contain the task at all, or you are "
    "uncertain. The learner hears nothing.\n"
    "- interject(correction): only when you observe a clear, persistent deviation "
    "from the OBSERVABLE ERRORS list. Provide a short directive sentence.\n\n"
    "Prefer observe_only strongly. [check] markers are internal diagnostics — not "
    "user speech. Never treat them as conversation."
)

TOOL_DECLS = [
    {
        "name": "interject",
        "description": "Call on observed valve counterclockwise rotation or listed errors.",
        "parameters": {
            "type": "object",
            "properties": {"correction": {"type": "string"}},
            "required": ["correction"],
        },
    },
    {
        "name": "observe_only",
        "description": "Call when nothing is wrong. Default path. Learner hears nothing.",
        "parameters": {
            "type": "object",
            "properties": {"assessment": {"type": "string"}},
            "required": ["assessment"],
        },
    },
]


@dataclass
class TurnResult:
    index: int
    sent_at: float
    first_event_at: float | None = None
    tool_calls: list[dict] = field(default_factory=list)
    audio_bytes: int = 0
    text: str = ""
    turn_complete: bool = False

    @property
    def latency_ms(self) -> float | None:
        if self.first_event_at is None:
            return None
        return (self.first_event_at - self.sent_at) * 1000

    @property
    def classification(self) -> str:
        names = {tc["name"] for tc in self.tool_calls}
        spoke = self.audio_bytes > 0 or bool(self.text.strip())
        if "interject" in names and spoke:
            return "GATE_FAIL_interject+spoke"
        if "observe_only" in names and spoke:
            return "GATE_FAIL_observe+spoke"
        if spoke and not self.tool_calls:
            return "GATE_FAIL_spoke_only"
        if "interject" in names:
            return "called_interject"
        if "observe_only" in names:
            return "called_observe_only"
        if self.tool_calls:
            return "called_other"
        return "silent_no_output"


async def _frame_pump(ws, jpeg_b64: str, stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        try:
            msg = {"realtimeInput": {"video": {
                "data": jpeg_b64, "mimeType": "image/jpeg",
            }}}
            await ws.send(json.dumps(msg))
        except Exception:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _ingest(event: dict, turn: TurnResult, now: float) -> list[dict]:
    """Return tool calls that need to be acked."""
    acked = []
    if tc := event.get("toolCall"):
        for fc in tc.get("functionCalls", []):
            turn.tool_calls.append({
                "id": fc.get("id"),
                "name": fc.get("name"),
                "args": fc.get("args", {}),
            })
            acked.append(fc)
            if turn.first_event_at is None:
                turn.first_event_at = now
    if sc := event.get("serverContent"):
        mt = sc.get("modelTurn", {})
        for part in mt.get("parts", []):
            if inline := part.get("inlineData"):
                data = inline.get("data", "")
                try:
                    turn.audio_bytes += len(base64.b64decode(data))
                except Exception:
                    turn.audio_bytes += len(data)
                if turn.first_event_at is None:
                    turn.first_event_at = now
            if text := part.get("text"):
                turn.text += text
                if turn.first_event_at is None:
                    turn.first_event_at = now
        if ot := sc.get("outputTranscription"):
            if ot.get("text"):
                turn.text += ot["text"]
                turn.audio_bytes += 1  # marker
                if turn.first_event_at is None:
                    turn.first_event_at = now
        if sc.get("turnComplete"):
            turn.turn_complete = True
            if turn.first_event_at is None:
                turn.first_event_at = now
    return acked


async def _ack(ws, function_calls: list[dict]) -> None:
    if not function_calls:
        return
    responses = [
        {"id": fc.get("id"), "name": fc.get("name"), "response": {"status": "ack"}}
        for fc in function_calls
    ]
    await ws.send(json.dumps({"toolResponse": {"functionResponses": responses}}))


async def _drain(ws, turn: TurnResult, *, first_timeout: float, grace: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + first_timeout
    while turn.first_event_at is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            return
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            event = json.loads(text)
        except Exception:
            continue
        acked = _ingest(event, turn, loop.time())
        await _ack(ws, acked)
        if turn.turn_complete:
            return
    while not turn.turn_complete:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=grace)
        except asyncio.TimeoutError:
            return
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        try:
            event = json.loads(text)
        except Exception:
            continue
        acked = _ingest(event, turn, loop.time())
        await _ack(ws, acked)


async def main(args: argparse.Namespace) -> int:
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not set")
        return 2

    neutral_jpeg = (BACKEND_ROOT / "test_data" / "neutral.jpg").read_bytes()
    jpeg_b64 = base64.b64encode(neutral_jpeg).decode("ascii")

    setup = {
        "setup": {
            "model": f"models/{args.model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "mediaResolution": "MEDIA_RESOLUTION_LOW",
            },
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "tools": [{"functionDeclarations": TOOL_DECLS}],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [t["name"] for t in TOOL_DECLS],
                },
            },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "contextWindowCompression": {
                "triggerTokens": 100_000,
                "slidingWindow": {"targetTokens": 40_000},
            },
            "sessionResumption": {},
        },
    }

    log_path = BACKEND_ROOT / "tmp" / f"tool_config_probe_{int(time.time())}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8", buffering=1)
    def log(kind: str, payload: dict) -> None:
        import datetime
        log_fh.write(json.dumps({
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
            "kind": kind, "payload": payload,
        }) + "\n")

    print(f"[INFO] model={args.model} beats={args.beats} interval={args.interval}s log={log_path}")

    async with websockets.connect(
        _ws_url(args.model), max_size=None, ping_interval=None,
    ) as ws:
        await ws.send(json.dumps(setup))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            print("[FAIL] no setupComplete within 10s")
            return 1
        text0 = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if "setupComplete" not in text0:
            print(f"[FAIL] unexpected first message: {text0[:300]}")
            return 1
        log("setup_complete", {})
        print("[OK] setupComplete received (toolConfig.mode=ANY accepted)")

        stop = asyncio.Event()
        pump = asyncio.create_task(
            _frame_pump(ws, jpeg_b64, stop, args.frame_interval),
        )

        # Warm-up — let a few frames land in context before the first heartbeat
        await asyncio.sleep(2.0)

        turns: list[TurnResult] = []
        loop = asyncio.get_running_loop()
        phase_start = loop.time()

        for i in range(1, args.beats + 1):
            target = phase_start + args.interval * i
            sleep_s = target - loop.time()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            sent_at = loop.time()
            await ws.send(json.dumps({"realtimeInput": {"text": "[check]"}}))
            log("heartbeat_sent", {"index": i, "t_offset": args.interval * i})

            turn = TurnResult(index=i, sent_at=sent_at)
            await _drain(
                ws, turn,
                first_timeout=args.interval - 0.5,
                grace=1.0,
            )
            turns.append(turn)
            log("heartbeat_result", {
                "index": i,
                "classification": turn.classification,
                "latency_ms": turn.latency_ms,
                "tool_calls": turn.tool_calls,
                "audio_bytes": turn.audio_bytes,
                "text": turn.text,
                "turn_complete": turn.turn_complete,
            })

            klass = turn.classification
            lat = f"{turn.latency_ms:.0f}ms" if turn.latency_ms is not None else "—"
            tools_str = ""
            if turn.tool_calls:
                first = turn.tool_calls[0]
                args_str = json.dumps(first.get("args", {}))
                if len(args_str) > 90:
                    args_str = args_str[:87] + "..."
                tools_str = f" {first['name']}({args_str})"
            prefix = "✓" if klass == "called_observe_only" else (
                "!" if klass.startswith("GATE_FAIL") else "?"
            )
            print(f"  [{prefix} HB {i}/{args.beats}] {klass:32s} lat={lat:>7s}{tools_str}")

        stop.set()
        pump.cancel()
        try:
            await pump
        except (asyncio.CancelledError, Exception):
            pass

    # Summary
    by_class: dict[str, int] = {}
    for t in turns:
        by_class[t.classification] = by_class.get(t.classification, 0) + 1
    print(f"\n=== SUMMARY ===")
    print(f"  Total heartbeats:         {len(turns)}")
    for k in sorted(by_class.keys()):
        print(f"  {k}:".ljust(34) + f"{by_class[k]}")
    gate_fail = sum(v for k, v in by_class.items() if k.startswith("GATE_FAIL"))
    observe_only = by_class.get("called_observe_only", 0)
    interject = by_class.get("called_interject", 0)
    print(f"\n  Gate failures (any audio leak): {gate_fail}")
    print(f"  called_observe_only (desired):  {observe_only}/{len(turns)} "
          f"({100*observe_only/len(turns):.0f}%)")
    if interject:
        print(f"  called_interject (false positive on static MacBook scene): {interject}")

    log_fh.close()
    print(f"\n  Log: {log_path}")
    return 0 if gate_fail == 0 else 1


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--beats", type=int, default=8)
    p.add_argument("--interval", type=float, default=5.0,
                   help="Seconds between heartbeats (default 5)")
    p.add_argument("--frame-interval", type=float, default=2.0,
                   help="Seconds between video frames (default 2 = 0.5 fps)")
    p.add_argument("--model", default=MODEL_DEFAULT)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_cli())))
