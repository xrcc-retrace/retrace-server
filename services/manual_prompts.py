"""Prompt templates for PDF manual → structured SOP extraction.

Two variants share the same output schema (`GeminiManualProcedureOutput`):

- `EXPERT_MANUAL_PROMPT`: extract the **end-to-end** procedure described
  in the manual. Used when an expert uploads a PDF from the iOS Expert
  tab or when a fixture is auto-loaded on startup. No diagnostic
  scoping — the full procedure is the deliverable.

- `TROUBLESHOOT_MANUAL_PROMPT`: scope tightly to a specific fault path
  the user is hitting, using a focus_problem + diagnostic_summary the
  troubleshoot Live model has already collected. Used when troubleshoot
  mode falls back to a manual after no library match (legacy path —
  the live troubleshoot flow now uses web_search_for_fix instead).

Both prompts ask Gemini to populate `source_page` (1-indexed) AND a
per-step `source_bbox` (page-percent rectangle) so the ingest pipeline
can crop just the figure illustrating each step. If a step has no figure
in the manual (e.g. a purely textual instruction), Gemini returns an
all-zero bbox and the pipeline skips the image for that step entirely
— there is no full-page fallback.

Per-field shape rules (title format, description length, tips/warnings,
completion_criteria, error_criteria) live in
`models.procedure.GeminiManualProcedureStep` field descriptions — the
prompt only carries flow-distinctive content (full-procedure vs
fault-scoped, source_page/bbox cropping rules).
"""


EXPERT_MANUAL_PROMPT = """You are an expert procedure analyst. Given this product's manufacturer manual (a PDF), extract the COMPLETE end-to-end procedure described in the manual so an AI coach can teach a first-time learner step-by-step.

Product: {product_name}

Rules:
1. Extract the full procedure as documented in the manual. Do not narrow to a single fault path.
2. Step ordering follows the manual's own sequence — do not reorder.
3. For each step's `source_bbox`: crop ONLY the small illustration positioned with this step's numbered text — NOT the whole page, NOT a neighboring step's figure. Add about 5% padding on each side so the crop has visual breathing room. If this step has no figure (e.g. a purely textual instruction like "wait 30 seconds") or you cannot confidently isolate one figure for it, return {{x:0, y:0, w:0, h:0}} and the pipeline will skip the image for this step. Do NOT guess — a wrong bbox is worse than no image.
4. Title the procedure: "{product_name} — <2-5 word task summary>". Pick the most natural English summary (e.g., "Descaling", "Filter Replacement", "First-time Setup").

A separate vision model later confirms each step by comparing the learner's live camera frame to `completion_criteria`, so write that field as a single PRESENT-TENSE spatial/LED end-state asserting what IS visible (positive presence/configuration, never absence, removal, motion, or "no longer" — the model sees a state, not a change) and set `visual_verifiability` (reliable / marginal / none) per its field description — use `none` when the step has no single-frame visual consequence (a button press, a wait, fine alignment).

Field-level shape requirements (per-step title, description, tips, warnings, completion_criteria, error_criteria, visual_verifiability, step cap) live in the response schema (GeminiManualProcedureOutput) — follow the field descriptions there.

Output JSON matching the GeminiManualProcedureOutput schema."""


TROUBLESHOOT_MANUAL_PROMPT = """You are a service-manual analyst. Given this product's manufacturer manual (a PDF), extract a focused repair procedure for a specific reported problem.

Focus problem: {focus_problem}

Diagnostic summary (what was already ruled in/out during conversation with the user):
{diagnostic_summary}

Rules:
1. Scope tightly to the focus problem — skip sections unrelated to this specific fault path.
2. For each step's `source_bbox`: crop ONLY this step's illustration on source_page, NOT the whole page or a neighboring step's figure. Add about 5% padding on each side. If this step has no figure or you cannot confidently isolate one, return {{x:0, y:0, w:0, h:0}} and the pipeline will skip the image. Do NOT guess.
3. Title the procedure: "{product_name} — {focus_title}" where focus_title summarizes the problem in 3-5 words.

A separate vision model later confirms each step by comparing the learner's live camera frame to `completion_criteria`, so write that field as a single PRESENT-TENSE spatial/LED end-state asserting what IS visible (positive presence/configuration, never absence, removal, motion, or "no longer" — the model sees a state, not a change) and set `visual_verifiability` (reliable / marginal / none) per its field description — use `none` when the step has no single-frame visual consequence (a button press, a wait, fine alignment).

Field-level shape requirements (per-step title, description, tips, warnings, completion_criteria, error_criteria, visual_verifiability, step cap) live in the response schema (GeminiManualProcedureOutput) — follow the field descriptions there.

Output JSON matching the GeminiManualProcedureOutput schema."""
