"""Troubleshoot mode — conversational diagnostic session, handoff to Learner.

Mode-isolated: all troubleshoot-specific prompts, tool handlers, and session
helpers live under this package. Shared infrastructure (session store,
ephemeral tokens, database, Gemini Live service) stays shared.

Importing this package registers tool handlers with the shared session
dispatcher (see handlers.py).
"""

from services.troubleshoot import handlers  # noqa: F401 — side-effect: register handlers
