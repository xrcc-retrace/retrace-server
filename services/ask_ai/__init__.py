"""Ask-AI mode — lightweight audio-only learner voice search over the library.

Importing the package registers its tool handlers with the shared dispatcher
(same pattern as services.troubleshoot).
"""

from services.ask_ai import handlers  # noqa: F401  (import triggers registration)
