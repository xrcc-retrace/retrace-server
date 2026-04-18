"""CLI + REPL frontend for the Gemini Live behavioral tuning playground.

Usage:
    python -m gemini_live_playground --procedure-id <uuid>
    python -m gemini_live_playground --scenario gemini_live_playground/scenarios/sample.yaml
    python -m gemini_live_playground --scenario … --replay        # non-interactive
    python -m gemini_live_playground --procedure-id … --text-modality

REPL slash commands:
    /state           print the in-memory session dict (current_step, history)
    /jump <n>        synthesize a go_to_step tool call locally (skip the model)
    /summary         inject services.session.get_context_summary as a user turn
    /save <name>     write current scenario back to scenarios/<name>.yaml
    /quit | /q       clean shutdown

Set GEMINI_API_KEY in main-local-server-test/.env (already used by the server).
The FastAPI server does NOT need to be running — tool calls dispatch in-process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import yaml

from gemini_live_playground.runner import (
    _CYAN,
    _DIM,
    _GREEN,
    _MAGENTA,
    _RED,
    _YELLOW,
    PlaygroundConfig,
    PlaygroundSession,
    PromptOverrides,
    c,
    get_context_summary,
    get_session,
    handle_tool_call,
    load_scenario,
    scripted_turn_source,
)


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------


def _handle_slash(cmd: str, session: PlaygroundSession) -> str | None:
    """Run a slash command. Returns a string to inject as a user turn, or None."""
    parts = cmd.strip().split(None, 1)
    name = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if name == "/state":
        s = get_session(session.session_id)
        if not s:
            print(c("session not found", _RED))
            return None
        snapshot = {
            "current_step": s["current_step"],
            "total_steps": s["total_steps"],
            "status": s["status"],
            "step_history": s["step_history"],
        }
        print(c(json.dumps(snapshot, indent=2, default=str), _DIM))
        return None

    if name == "/jump":
        try:
            n = int(arg)
        except ValueError:
            print(c("usage: /jump <step_number>", _RED))
            return None
        result = handle_tool_call(session.session_id, "go_to_step", {"step_number": n})
        session.log.write("manual_jump", {"step": n, "result": result})
        print(c(f"[MANUAL] go_to_step({n}) → {result}", _MAGENTA))
        # Don't inject as a user turn — this is a side-channel mutation,
        # the model won't know about it until the next time it queries state.
        return None

    if name == "/summary":
        summary = get_context_summary(session.session_id)
        if summary is None:
            print(c("session not found", _RED))
            return None
        print(c(f"[INFO] injecting context summary as user turn", _DIM))
        return f"[Session resumed — context summary]\n{summary}"

    if name == "/save":
        if not arg:
            print(c("usage: /save <name>", _RED))
            return None
        out = _save_scenario(session, arg.strip())
        print(c(f"[INFO] saved scenario → {out}", _DIM))
        return None

    print(c(f"unknown command: {name}", _RED))
    return None


def _save_scenario(session: PlaygroundSession, name: str) -> Path:
    cfg = session.cfg
    payload: dict = {
        "procedure_id": cfg.procedure_id,
        "auto_advance": cfg.auto_advance,
        "voice": cfg.voice,
    }
    overrides: dict = {}
    if cfg.overrides.step_progression:
        overrides["step_progression"] = cfg.overrides.step_progression
    if cfg.overrides.append:
        overrides["append"] = cfg.overrides.append
    if cfg.overrides.step_completion_criteria:
        overrides["step_completion_criteria"] = dict(cfg.overrides.step_completion_criteria)
    if overrides:
        payload["prompt_overrides"] = overrides
    if cfg.text_modality:
        payload["text_modality"] = True
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# Turn sources
# ---------------------------------------------------------------------------


def repl_turn_source():
    """Async generator that yields user turns from stdin until /quit or EOF."""
    async def gen(session: PlaygroundSession) -> AsyncIterator[str]:
        print(c(
            "[INFO] REPL ready. Type a message to coach, or /state /jump <n> "
            "/summary /save <name> /quit",
            _DIM,
        ))
        while True:
            try:
                text = await asyncio.to_thread(input, c("> ", _CYAN))
            except (EOFError, KeyboardInterrupt):
                print()
                return
            text = text.strip()
            if not text:
                continue
            if text in ("/quit", "/q", "/exit"):
                return
            if text.startswith("/"):
                injected = _handle_slash(text, session)
                if injected is not None:
                    yield injected
                continue
            yield text
    return gen


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> PlaygroundConfig:
    if args.scenario:
        cfg = load_scenario(Path(args.scenario))
    else:
        if not args.procedure_id:
            raise SystemExit(
                "either --scenario <yaml> or --procedure-id <uuid> is required",
            )
        cfg = PlaygroundConfig(procedure_id=args.procedure_id)

    # CLI flags override scenario file
    if args.procedure_id:
        cfg.procedure_id = args.procedure_id
    if args.voice:
        cfg.voice = args.voice
    if args.manual:
        cfg.auto_advance = False
    if args.text_modality:
        cfg.text_modality = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m gemini_live_playground",
        description="Behavioral tuning REPL for Retrace Gemini Live coaching sessions",
    )
    parser.add_argument("--scenario", help="Path to a scenario YAML")
    parser.add_argument("--procedure-id", help="Procedure UUID from data/retrace.db (overrides scenario)")
    parser.add_argument("--voice", help="Override voice (Puck, Charon, Kore, ...)")
    parser.add_argument("--manual", action="store_true", help="Force manual mode (auto_advance=False)")
    parser.add_argument("--text-modality", action="store_true", help="Set response_modalities=['TEXT']")
    parser.add_argument("--replay", action="store_true", help="Run scripted_turns then exit (no REPL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _build_config(args)

    if args.replay:
        if not cfg.scripted_turns:
            print(c("[ERROR] --replay requires scripted_turns in the scenario YAML", _RED))
            sys.exit(2)
        turn_source = scripted_turn_source(cfg.scripted_turns)
    else:
        turn_source = repl_turn_source()

    session = PlaygroundSession(cfg)
    try:
        asyncio.run(session.run(turn_source))
    except KeyboardInterrupt:
        print(c("\n[INFO] interrupted", _DIM))


if __name__ == "__main__":
    main()
