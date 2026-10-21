"""Voice-command registry. Single source of truth: voice_commands.yaml.

Loaded once at import time. Callers (prompt builders, future API endpoints) ask
for prompt utterances by command id and trust this module to keep the phrasing
in sync with what the designer edited in the YAML.

Designer workflow is documented in the YAML header; this module is the code
contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

# The YAML lives next to this package (main-local-server-test/voice_commands.yaml),
# one directory above services/.
_YAML_PATH = Path(__file__).resolve().parent.parent / "voice_commands.yaml"

_ALLOWED_MODES = {"learner", "troubleshoot", "expert_interview", "global"}
_ALLOWED_HANDLER_TYPES = {"server_tool", "client_tool", "hybrid"}
_ALLOWED_CONFIRMATIONS = {"none", "soft", "destructive_hold"}

# Modes that include a given command. A "global" command surfaces in both
# learner and troubleshoot contexts. The expert interviewer deliberately
# excludes globals — its mic is only open inside answer windows, so the
# learner-side help/mute vocabulary doesn't apply.
_MODE_INCLUSIONS = {
    "learner": {"learner", "global"},
    "troubleshoot": {"troubleshoot", "global"},
    "expert_interview": {"expert_interview"},
}


def _load() -> list[dict]:
    with _YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    commands = data.get("commands") or []
    seen: set[str] = set()
    for cmd in commands:
        cid = cmd.get("id")
        if not cid or not isinstance(cid, str):
            raise ValueError(f"voice_commands.yaml: command missing string id: {cmd!r}")
        if cid in seen:
            raise ValueError(f"voice_commands.yaml: duplicate command id {cid!r}")
        seen.add(cid)
        mode = cmd.get("mode")
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"voice_commands.yaml: {cid!r} has invalid mode {mode!r}")
        htype = cmd.get("handler_type")
        if htype not in _ALLOWED_HANDLER_TYPES:
            raise ValueError(f"voice_commands.yaml: {cid!r} has invalid handler_type {htype!r}")
        conf = cmd.get("confirmation")
        if conf not in _ALLOWED_CONFIRMATIONS:
            raise ValueError(f"voice_commands.yaml: {cid!r} has invalid confirmation {conf!r}")
        if not cmd.get("prompt_utterances"):
            raise ValueError(f"voice_commands.yaml: {cid!r} has empty prompt_utterances")
        if len(cmd["prompt_utterances"]) > 5:
            raise ValueError(
                f"voice_commands.yaml: {cid!r} has {len(cmd['prompt_utterances'])} "
                "prompt_utterances (max 5 — prompt budget)"
            )
    logger.info("voice_commands.yaml loaded: %d commands", len(commands))
    return commands


_COMMANDS: list[dict] = _load()
_BY_ID: dict[str, dict] = {cmd["id"]: cmd for cmd in _COMMANDS}


def all_commands() -> list[dict]:
    """Return every command (in YAML order)."""
    return list(_COMMANDS)


def commands_for_mode(mode: str) -> list[dict]:
    """Return commands available in the given session mode (includes globals)."""
    if mode not in _MODE_INCLUSIONS:
        raise ValueError(f"unknown mode {mode!r} (expected: learner | troubleshoot | expert_interview)")
    wanted = _MODE_INCLUSIONS[mode]
    return [cmd for cmd in _COMMANDS if cmd["mode"] in wanted]


def get(command_id: str) -> dict:
    """Look up a single command by id; raises KeyError if absent."""
    return _BY_ID[command_id]


def prompt_utterances(command_id: str) -> list[str]:
    """Return the (≤5) utterances shipped to Gemini for the given command.

    Prompt-builder sites pass the command id; utterances come from the YAML.
    Editing the YAML is the only way to change these.
    """
    return list(_BY_ID[command_id]["prompt_utterances"])


def format_utterance_examples(command_id: str) -> str:
    """Render prompt utterances as a comma-separated quoted list for inline use.

    Example: `"done", "next", "I'm done", "move on", "finished"`.
    """
    return ", ".join(f'"{u}"' for u in prompt_utterances(command_id))


def utterance_join(command_ids: Iterable[str], separator: str = " / ") -> str:
    """Render a union of several commands' prompt utterances as a single phrase.

    Useful when a prompt block wants to list trigger phrases for a family of
    commands (e.g. all step-advance utterances together).
    """
    parts: list[str] = []
    for cid in command_ids:
        parts.extend(prompt_utterances(cid))
    return separator.join(f'"{p}"' for p in parts)
