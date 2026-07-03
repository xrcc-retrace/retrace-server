"""PlaygroundSession — drives a Gemini Live session for behavioral tuning.

Mirrors the production iOS coaching pipeline: same model, system prompt,
tool definitions, voice, compression, proactivity, and transcription
(via the shared `services.ephemeral_token.build_live_connect_config`
helper). Differences from production:

  - Text-only I/O. The Live session still negotiates AUDIO modality so the
    model behaves identically to production; we render replies via the
    output_audio_transcription stream. Pass `text_modality=True` to
    switch to ["TEXT"] for raw-text experiments.

  - Direct API key, no ephemeral token. Server-side script holds the key,
    so there's no point in a single-use mint cycle.

  - Tool calls dispatch in-process against `services.session.handle_tool_call`.
    No FastAPI server required.

  - No video frames. Visual-verification tuning (auto_advance) happens via
    simulated text turns ("looks right from here") instead of camera feed.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from services.database import get_procedure
from services.ephemeral_token import build_live_connect_config
from services.prompt_builder import GEMINI_CONFIG, TOOL_DEFINITIONS, build_system_prompt
from services.session import (
    create_session,
    delete_session,
    get_context_summary,
    get_session,
    handle_tool_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario config
# ---------------------------------------------------------------------------


@dataclass
class PromptOverrides:
    step_progression: str | None = None
    append: str | None = None
    step_completion_criteria: dict[int, str] = field(default_factory=dict)


@dataclass
class PlaygroundConfig:
    procedure_id: str
    auto_advance: bool = True
    active_monitoring: bool = True
    voice: str = "Puck"
    overrides: PromptOverrides = field(default_factory=PromptOverrides)
    scripted_turns: list[str] = field(default_factory=list)
    log_file: Path | None = None
    text_modality: bool = False


def load_scenario(path: Path) -> PlaygroundConfig:
    data = yaml.safe_load(path.read_text()) or {}
    if "procedure_id" not in data:
        raise ValueError(f"{path}: missing required field 'procedure_id'")

    raw_overrides = data.get("prompt_overrides") or {}
    raw_criteria = raw_overrides.get("step_completion_criteria") or {}
    overrides = PromptOverrides(
        step_progression=raw_overrides.get("step_progression"),
        append=raw_overrides.get("append"),
        step_completion_criteria={int(k): str(v) for k, v in raw_criteria.items()},
    )

    log_file = data.get("log_file")
    return PlaygroundConfig(
        procedure_id=str(data["procedure_id"]),
        auto_advance=bool(data.get("auto_advance", True)),
        active_monitoring=bool(data.get("active_monitoring", True)),
        voice=str(data.get("voice", "Puck")),
        overrides=overrides,
        scripted_turns=[str(t) for t in (data.get("scripted_turns") or [])],
        log_file=Path(log_file) if log_file else None,
        text_modality=bool(data.get("text_modality", False)),
    )


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------


def _apply_criteria_overrides(proc: dict, overrides: PromptOverrides) -> dict:
    """Return a deep copy of `proc` with per-step completion_criteria swapped."""
    if not overrides.step_completion_criteria:
        return proc
    proc = copy.deepcopy(proc)
    for step in proc.get("steps", []):
        new = overrides.step_completion_criteria.get(step.get("step_number"))
        if new is not None:
            step["completion_criteria"] = new
    return proc


# Match the whole "=== STEP PROGRESSION ... ===" block up to (but not
# including) the next section header. The build_system_prompt template
# always places "=== COACHING RULES ===" right after, so we anchor on that.
_STEP_PROGRESSION_RE = re.compile(
    r"=== STEP PROGRESSION[\s\S]*?(?=\n\n=== COACHING RULES ===)",
    re.MULTILINE,
)


def _apply_prompt_overrides(prompt: str, overrides: PromptOverrides) -> str:
    if overrides.step_progression:
        new_prompt, n = _STEP_PROGRESSION_RE.subn(
            overrides.step_progression.rstrip(), prompt, count=1
        )
        if n == 0:
            logger.warning(
                "step_progression override: no '=== STEP PROGRESSION' block "
                "found in built prompt; check prompt_builder.py template",
            )
        else:
            prompt = new_prompt
    if overrides.append:
        prompt = prompt.rstrip() + "\n\n" + overrides.append.strip()
    return prompt


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
# Terminal rendering
# ---------------------------------------------------------------------------


_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"


def c(s: str, color: str) -> str:
    return f"{color}{s}{_RESET}" if sys.stdout.isatty() else s


def _compact(args: dict) -> str:
    return json.dumps(args, separators=(",", ":")) if args else ""


def _summarize_tool_result(name: str, result: dict) -> str:
    if "error" in result:
        return c(f"error={result['error']}", _RED)
    if name in ("advance_step", "check_step"):
        # check_step in the playground hits the server degrade handler (no
        # camera frame in-process): done-claims advance verbally, questions
        # come back unverifiable.
        if result.get("status") == "completed":
            return result.get("message", "completed")
        if result.get("status") == "advanced":
            return f'step {result["new_step"]}: "{result["step_title"]}"'
        if result.get("status") == "unverifiable":
            return "unverifiable (no camera in playground)"
    if name == "go_to_step":
        return f'step {result.get("current_step")}: "{result.get("step_title")}"'
    if name == "get_reference_clip":
        return f'step {result.get("step_number")} clip={result.get("clip_url")}'
    return json.dumps(result, separators=(",", ":"))[:200]


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


class PlaygroundSession:
    """Drives one Gemini Live session for behavioral tuning.

    Lifecycle: `await session.run(turn_source)` opens the Live socket,
    pulls turns from `turn_source` (an async iterable of strings), sends
    each as a user turn, and streams back AI replies + tool calls until
    the iterable is exhausted (REPL → user typed /quit; replay → list
    drained). Tool calls dispatch in-process; everything is logged to a
    JSONL file you can diff between tuning runs.
    """

    def __init__(self, cfg: PlaygroundConfig):
        self.cfg = cfg
        self.session_id: str | None = None
        self.system_prompt: str = ""
        self.log: EventLog | None = None
        self._live = None  # google.genai AsyncSession
        self._last_total_tokens: int = 0
        self._last_band: int = 0

    # -- setup --------------------------------------------------------------

    def _prepare(self) -> None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set — add to .env or export")

        proc = get_procedure(self.cfg.procedure_id)
        if not proc:
            raise RuntimeError(
                f"Procedure {self.cfg.procedure_id!r} not found in data/retrace.db",
            )

        proc = _apply_criteria_overrides(proc, self.cfg.overrides)

        session = create_session(
            self.cfg.procedure_id,
            auto_advance=self.cfg.auto_advance,
            active_monitoring=self.cfg.active_monitoring,
            voice=self.cfg.voice,
        )
        if not session:
            raise RuntimeError("create_session returned None")
        # services.session.create_session re-reads the procedure from SQLite,
        # so any per-step completion_criteria overrides we just applied are
        # only on our local copy. Patch them into the in-memory session
        # store so the tool-call dispatcher (advance_step, go_to_step) sees
        # the overridden criteria when it builds its responses.
        session["procedure"] = proc
        self.session_id = session["session_id"]

        self.system_prompt = _apply_prompt_overrides(
            build_system_prompt(
                proc,
                auto_advance=self.cfg.auto_advance,
                active_monitoring=self.cfg.active_monitoring,
                current_step_number=session["current_step"],
            ),
            self.cfg.overrides,
        )

        log_path = self.cfg.log_file or (
            Path(__file__).parent / "logs" / f"run-{int(time.time())}.jsonl"
        )
        self.log = EventLog(log_path)
        self.log.write("session_start", {
            "session_id": self.session_id,
            "procedure_id": self.cfg.procedure_id,
            "procedure_title": proc.get("title"),
            "voice": self.cfg.voice,
            "auto_advance": self.cfg.auto_advance,
            "active_monitoring": self.cfg.active_monitoring,
            "text_modality": self.cfg.text_modality,
            "total_steps": session["total_steps"],
            "system_prompt": self.system_prompt,
            "applied_overrides": {
                "step_progression": bool(self.cfg.overrides.step_progression),
                "append": bool(self.cfg.overrides.append),
                "step_completion_criteria": list(
                    self.cfg.overrides.step_completion_criteria.keys()
                ),
            },
        })

    # -- main loop ----------------------------------------------------------

    async def run(self, turn_source: Callable[[PlaygroundSession], AsyncIterator[str]]) -> None:
        self._prepare()

        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )

        cfg = build_live_connect_config(
            system_instruction=self.system_prompt,
            tools=TOOL_DEFINITIONS,
            voice=self.cfg.voice,
        )
        if self.cfg.text_modality:
            cfg.response_modalities = ["TEXT"]
        # Proactivity tells the model to decide whether to respond based on
        # audio cues. With text-only input there's nothing to be proactive
        # about, and on the native-audio model it can cause the socket to
        # close with 1008 mid-session. Strip it for the playground; the
        # production iOS path (audio + video) keeps it via the shared
        # build_live_connect_config default.
        cfg.proactivity = None
        # input_audio_transcription is also audio-only and has no effect on
        # text input — drop it to keep the config minimal for text I/O.
        cfg.input_audio_transcription = None

        model = GEMINI_CONFIG["model"]
        self._print_banner(model)

        try:
            async with client.aio.live.connect(model=model, config=cfg) as live:
                self._live = live
                async for text in turn_source(self):
                    await self._send_turn(text)
                    await self._drain_turn()
        finally:
            await self._teardown()

    def _print_banner(self, model: str) -> None:
        print(c(
            f"[INFO] model={model} voice={self.cfg.voice} "
            f"auto_advance={self.cfg.auto_advance} "
            f"modality={'TEXT' if self.cfg.text_modality else 'AUDIO+transcript'}",
            _DIM,
        ))
        print(c(f"[INFO] session_id={self.session_id}", _DIM))
        print(c(f"[INFO] log={self.log.path}", _DIM))
        print(c("── SYSTEM PROMPT ──", _DIM))
        print(c(self.system_prompt, _DIM))
        print(c("── end prompt ──", _DIM))

    # -- send / receive -----------------------------------------------------

    async def _send_turn(self, text: str) -> None:
        self.log.write("user_turn", {"text": text})
        print(c(f"> {text}", _CYAN))
        await self._live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    async def _drain_turn(self) -> None:
        """Read responses until turn_complete=True (or socket closes)."""
        ai_buf: list[str] = []
        learner_buf: list[str] = []
        async for response in self._live.receive():
            # Tool call → dispatch in-process, return result back to Gemini.
            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    await self._handle_function_call(fc)
                continue

            sc = response.server_content
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    learner_buf.append(sc.input_transcription.text)
                if sc.output_transcription and sc.output_transcription.text:
                    ai_buf.append(sc.output_transcription.text)
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        text = getattr(part, "text", None)
                        if text:
                            ai_buf.append(text)
                if sc.interrupted:
                    print(c("[BARGE-IN] model interrupted", _YELLOW))
                    self.log.write("interrupted", {})
                if sc.turn_complete:
                    self._flush_turn(learner_buf, ai_buf)
                    return

            if response.session_resumption_update:
                handle = response.session_resumption_update.new_handle
                if handle:
                    print(c(f"[INFO] resumption handle {handle[:12]}…", _DIM))
                    self.log.write("resumption_update", {"new_handle": handle})

            if response.go_away:
                tl = response.go_away.time_left
                print(c(f"[WARN] goAway time_left={tl}", _YELLOW))
                self.log.write("go_away", {"time_left": str(tl)})

            if response.usage_metadata and response.usage_metadata.total_token_count:
                self._track_tokens(int(response.usage_metadata.total_token_count))

    def _flush_turn(self, learner_buf: list[str], ai_buf: list[str]) -> None:
        if learner_buf:
            t = "".join(learner_buf).strip()
            if t:
                print(c(f"(learner heard: {t})", _DIM))
                self.log.write("input_transcript", {"text": t})
        if ai_buf:
            t = "".join(ai_buf).strip()
            if t:
                print(c(f"AI: {t}", _GREEN))
                self.log.write("ai_turn", {"text": t})
                return
        # Either AUDIO mode without transcript yet, or empty turn.
        print(c("AI: [no text — audio modality without transcript captured]", _DIM))

    async def _handle_function_call(self, fc: Any) -> None:
        args = dict(fc.args) if fc.args else {}
        self.log.write("tool_call", {"id": fc.id, "name": fc.name, "args": args})
        result = handle_tool_call(self.session_id, fc.name, args)
        self.log.write("tool_result", {"id": fc.id, "name": fc.name, "result": result})
        print(c(
            f"[TOOL] {fc.name}({_compact(args)}) → {_summarize_tool_result(fc.name, result)}",
            _MAGENTA,
        ))
        await self._live.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=fc.id, name=fc.name, response=result),
            ],
        )

    def _track_tokens(self, total: int) -> None:
        # Log every 10k crossed
        band = total // 10_000
        if band > self._last_band:
            print(c(f"[INFO] tokens={total:,}", _DIM))
            self.log.write("usage", {"total_token_count": total})
            self._last_band = band
        # Detect compression: a sudden drop of >30k between samples means
        # Gemini fired its sliding-window compaction (trigger 100k → target 40k).
        if self._last_total_tokens and total + 30_000 < self._last_total_tokens:
            print(c(
                f"[INFO] context compressed: {self._last_total_tokens:,} → {total:,}",
                _DIM,
            ))
            self.log.write(
                "compression",
                {"before": self._last_total_tokens, "after": total},
            )
            self._last_band = band  # avoid re-logging the band on the way back up
        self._last_total_tokens = total

    # -- teardown -----------------------------------------------------------

    async def _teardown(self) -> None:
        if self.session_id:
            delete_session(self.session_id)
        if self.log:
            self.log.write("session_end", {"session_id": self.session_id})
            self.log.close()


# ---------------------------------------------------------------------------
# Built-in turn sources
# ---------------------------------------------------------------------------


def scripted_turn_source(turns: list[str]) -> Callable[[PlaygroundSession], AsyncIterator[str]]:
    """Return an async generator that yields each scripted turn in order."""
    async def gen(_session: PlaygroundSession) -> AsyncIterator[str]:
        for t in turns:
            yield t
            await asyncio.sleep(0.2)  # let receive loop settle between turns
    return gen


# Re-export so playground.py can build slash commands without re-importing.
__all__ = [
    "PlaygroundConfig",
    "PlaygroundSession",
    "PromptOverrides",
    "EventLog",
    "load_scenario",
    "scripted_turn_source",
    "get_session",
    "get_context_summary",
    "handle_tool_call",
    "c",
    "_DIM",
    "_CYAN",
    "_MAGENTA",
    "_YELLOW",
    "_GREEN",
    "_RED",
]
