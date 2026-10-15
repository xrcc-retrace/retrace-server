#!/usr/bin/env python3
"""Validate voice_commands.yaml.

Run before committing any YAML edit:

    python validate_voice_commands.py

Exits 0 on success, non-zero on schema/budget violation. Also prints a summary
of commands whose `target` references a function or file that doesn't yet exist
(marked in the YAML with the `NEW:` prefix) — use that list as an engineering
TODO; it does NOT fail the build.

The heavy-lifting schema check lives inside services/voice_commands.py (runs at
import time — the server won't start with a malformed YAML). This script wraps
that plus a few extra design-time checks:

  - Every utterance appears in at most one command (prevents phrase collisions).
  - Every prompt_utterance is also present in the utterances list (so the
    shipped phrasing is always a subset of the designer's coverage list).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict


def _normalize(phrase: str) -> str:
    """Lowercase, strip trailing punctuation, collapse whitespace."""
    p = phrase.lower().strip()
    p = re.sub(r"[^\w\s]", "", p)  # drop punctuation (commas, apostrophes, etc.)
    p = re.sub(r"\s+", " ", p)
    return p

try:
    from services import voice_commands as vc
except Exception as e:  # pragma: no cover — surfaces the YAML error to the user
    print(f"FAIL: voice_commands.yaml failed to load: {e}", file=sys.stderr)
    sys.exit(1)


def _check_phrase_collisions(commands: list[dict]) -> list[str]:
    """Warn when the same phrase appears in multiple commands' utterances.

    Collisions are sometimes deliberate — "expand" can mean expand the clip or
    expand the step detail, depending on what's on screen. The command's
    `disambiguation` field is where that's resolved. We warn so the designer
    can double-check, but this does not fail the build.
    """
    warnings: list[str] = []
    seen: dict[str, list[str]] = defaultdict(list)
    for cmd in commands:
        for u in cmd.get("utterances", []):
            seen[_normalize(u)].append(cmd["id"])
    for phrase, owners in seen.items():
        if len(owners) > 1:
            unique_owners = sorted(set(owners))
            if len(unique_owners) > 1:
                warnings.append(
                    f"phrase {phrase!r} appears in multiple commands: "
                    f"{', '.join(unique_owners)} — ensure each has a "
                    "disambiguation note"
                )
    return warnings


def _check_prompt_subset(commands: list[dict]) -> list[str]:
    """Each prompt_utterance should match an entry in utterances (modulo punctuation).

    Template prompt_utterances containing "N" (e.g. "go to step N") are treated
    as patterns and skipped — they're intentionally unrendered forms that
    Gemini fills in from the user's speech.
    """
    problems: list[str] = []
    for cmd in commands:
        full = {_normalize(u) for u in cmd.get("utterances", [])}
        for pu in cmd.get("prompt_utterances", []):
            # Skip template placeholders ("go to step N" etc).
            if re.search(r"\bN\b", pu):
                continue
            if _normalize(pu) not in full:
                problems.append(
                    f"{cmd['id']}: prompt_utterance {pu!r} is not in the full "
                    "utterances list (after normalizing punctuation)"
                )
    return problems


def _summarize_gaps(commands: list[dict]) -> list[dict]:
    """Commands whose target starts with 'NEW:' — still need implementation."""
    return [c for c in commands if str(c.get("target", "")).startswith("NEW:")]


def main() -> int:
    commands = vc.all_commands()
    print(f"Loaded {len(commands)} commands.")

    warnings = _check_phrase_collisions(commands)
    problems = _check_prompt_subset(commands)

    if warnings:
        print("\nWarnings (intentional collisions are OK if disambiguation is set):")
        for w in warnings:
            print(f"  - {w}")

    if problems:
        print("\nSchema violations:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    gaps = _summarize_gaps(commands)
    if gaps:
        print(f"\nCommands still needing implementation ({len(gaps)}):")
        by_handler: dict[str, list[dict]] = defaultdict(list)
        for c in gaps:
            by_handler[c["handler_type"]].append(c)
        for htype in sorted(by_handler):
            print(f"  {htype}:")
            for c in by_handler[htype]:
                print(f"    - {c['id']:<28} target: {c['target']}")
        print(
            "\nNote: `NEW:` is informational, not a failure. The YAML is valid "
            "as a design artifact even before the referenced code exists."
        )

    # Spot-check that each mode has a reasonable command set.
    for mode in ("learner", "troubleshoot", "expert_interview"):
        n = len(vc.commands_for_mode(mode))
        print(f"{mode} mode: {n} commands available")

    print("\nOK: voice_commands.yaml is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
