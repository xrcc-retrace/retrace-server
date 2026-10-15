"""Expert Interviewer mode — a Gemini Live session that runs alongside expert
recording and asks the questions a learner would have needed.

Mode-parallel to services.troubleshoot / services.ask_ai. No tool handlers in
v1 (facts are transcript-derived at finalize), so importing the package has
no registration side effect; the router imports the submodules it needs.

See docs/expert-interviewer-design.md (repo root) for the architecture.
"""
