"""Prompt templates for PDF manual → structured SOP extraction.

Two variants share the same output schema (`GeminiProcedureOutput`):

- `EXPERT_MANUAL_PROMPT`: extract the **end-to-end** procedure described
  in the manual. Used when an expert uploads a PDF from the iOS Expert
  tab or when a fixture is auto-loaded on startup. No diagnostic
  scoping — the full procedure is the deliverable.

- `TROUBLESHOOT_MANUAL_PROMPT`: scope tightly to a specific fault path
  the user is hitting, using a focus_problem + diagnostic_summary the
  troubleshoot Live model has already collected. Used when troubleshoot
  mode falls back to a manual after no library match.

Both prompts ask Gemini to populate `source_page` (1-indexed) AND a
per-step `source_bbox` (page-percent rectangle) so the ingest pipeline
can crop just the figure illustrating each step. If a step has no figure
in the manual (e.g. a purely textual instruction), Gemini returns an
all-zero bbox and the pipeline skips the image for that step entirely
— there is no full-page fallback.
"""


EXPERT_MANUAL_PROMPT = """You are an expert procedure analyst. Given this product's manufacturer manual (a PDF), extract the COMPLETE end-to-end procedure described in the manual so an AI coach can teach a first-time learner step-by-step.

Product: {product_name}

Rules:
1. Extract the full procedure as documented in the manual. Do not narrow to a single fault path.
2. Cap at 15 steps. Prefer fewer when the manual itself uses fewer.
3. Step ordering follows the manual's own sequence — do not reorder.
4. For each step, return:
   - step_number (1-indexed, contiguous starting at 1).
   - title — short and action-oriented ("Fill the water tank", "Press and hold the button until the LED blinks orange").
   - description — 1-3 sentences a first-time learner can follow. Concrete actions, not paraphrased platitudes.
   - timestamp_start, timestamp_end — start at 0.0 and increment by ~30 seconds per step. These are placeholders for API compatibility; the source is a PDF, not a video.
   - tips — short technique notes the manual calls out (when to be gentle, which direction to turn, helpful timing). Empty list if none.
   - warnings — safety callouts the manual highlights. Empty list if none.
   - completion_criteria — ONE single observable end-state sentence written for VLM verification from a head-mounted camera. RULES:
     * Prefer high-signal cues: LED color or blink-state changes, lid open vs closed, container present vs absent, water level visibly above or below a marker, lever locked vs unlocked.
     * AVOID: fine alignment, sub-millimeter offsets, counting events, torque, anything requiring text recognition or measuring elapsed time.
     * If the manual describes an LED state machine (steady, blinking orange, alternating, etc.), use those LED states as the completion criterion — they are the cleanest visual signal.
   - error_criteria — exactly THREE single-sentence observable visual errors the learner might make on this step. Same hard rules as completion_criteria: visible state in a single frame, named object plus named wrong state, no judgment words ("incorrectly", "wrongly"), no cognitive claims ("learner forgets"), no counting. Each error must be mutually distinct.
   - source_page — the 1-indexed PDF page that CONTAINS this step's figure. If the step spans multiple pages, pick the one with the most relevant figure, photo, or LED diagram. If this step has NO figure in the manual at all, set source_page to 0.
   - source_bbox — the bounding box of the figure illustrating THIS specific step on `source_page`, expressed as percentages of page dimensions (0–100): {{x, y, w, h}} where (x,y) is the TOP-LEFT corner and (w,h) is the width and height. Crop ONLY the small illustration positioned with this step's numbered text — NOT the whole page, NOT a neighboring step's figure. Add about 5% padding on each side so the crop has visual breathing room. If this step has no figure (e.g. a purely textual instruction like "wait 30 seconds") or you cannot confidently isolate one figure for it, return {{x:0, y:0, w:0, h:0}} and the pipeline will skip the image for this step. Do NOT guess — a wrong bbox is worse than no image.

5. Title the procedure: "{product_name} — <2-5 word task summary>". Pick the most natural English summary (e.g., "Descaling", "Filter Replacement", "First-time Setup").
6. Description: one or two sentences naming the product and the task, with no marketing language.

Also produce two top-level iconography fields representing the MAIN PHYSICAL OBJECT (the appliance / device, not the action):
- `icon_symbol`: a single Iconify icon ID in the form "<prefix>:<name>" — e.g. "lucide:refrigerator", "material-symbols:humidifier-large", "phosphor:speaker-high", "tabler:washing-machine". Prefer these packs in order: lucide, material-symbols, phosphor, tabler. The icons render white on a dark surface — pick line-style monochrome icons, not multi-color packs (avoid emojione, twemoji, fxemoji).
- `icon_emoji`: a single emoji that represents the same object as a fallback (e.g. 💧 humidifier, 🧊 fridge, 📱 iPhone).

Also pick a `category` for this procedure — exactly one value from this closed list:
- "Assembly" — joining parts or components into a working unit for the first time.
- "Electrical" — wiring, terminating, or configuring electrical/electronic systems.
- "Calibration" — setting, adjusting, or verifying that a machine or instrument meets a specified output.
- "Maintenance" — scheduled upkeep that keeps a working system working: cleaning, lubricating, descaling, replacing consumables.
- "Repair" — diagnosing and fixing a system that has failed or degraded below spec.
- "Inspection" — verifying conformance to a standard without making any change.
- "Other" — only when none of the above fits.

Precedence: scheduled/preventive change → Maintenance; unplanned/corrective change → Repair; adjustment to spec → Calibration; check-only → Inspection. Pick "Other" only as a last resort.

Output JSON matching the GeminiProcedureOutput schema."""


TROUBLESHOOT_MANUAL_PROMPT = """You are a service-manual analyst. Given this product's manufacturer manual (a PDF), extract a focused repair procedure for a specific reported problem.

Focus problem: {focus_problem}

Diagnostic summary (what was already ruled in/out during conversation with the user):
{diagnostic_summary}

Produce a Standard Operating Procedure with these rules:
1. Scope tightly to the focus problem — skip sections unrelated to this specific fault path.
2. Cap the procedure at 15 steps. Prefer fewer steps if that's enough to fix the problem.
3. Each step must have:
   - A short, action-oriented title (e.g., "Unplug the unit", "Remove the water tank").
   - A 1-3 sentence description that a first-time learner can follow.
   - A `completion_criteria` string describing the observable end-state (what "done" looks like) in a single sentence, written for VLM verification: prefer LED states, lid/tray/container presence, water level cues. Avoid fine alignment, counting, or text recognition.
   - 2-5 `error_criteria` strings — observable error states that, if visible, should cause an AI coach to interject. Same rules as completion_criteria: single declarative sentence, specific object plus specific wrong configuration, visible in a single frame, no judgment words, no abstract faults.
   - source_page — the 1-indexed PDF page that CONTAINS this step's figure. If this step has no figure in the manual at all, set source_page to 0.
   - source_bbox — the bounding box of the figure illustrating THIS specific step on `source_page`, in page-percent coordinates (0–100): {{x, y, w, h}} where (x,y) is the TOP-LEFT corner. Crop ONLY this step's illustration, NOT the whole page or a neighboring step's figure. Add about 5% padding on each side. If this step has no figure or you cannot confidently isolate one, return {{x:0, y:0, w:0, h:0}} and the pipeline will skip the image. Do NOT guess.
4. Include tips for technique and safety warnings where the manual calls them out.
5. Use timestamps starting from 0.0 and ticking up in reasonable ~30-second increments per step. The manual is a PDF, not a video — timestamps here are just for API compatibility.

Title the procedure: "{product_name} — {focus_title}" where focus_title summarizes the problem in 3-5 words.

Description: one or two sentences naming the product and the problem being addressed.

Also produce two top-level iconography fields representing the MAIN PHYSICAL OBJECT (the appliance / device, not the action):
- `icon_symbol`: a single Iconify icon ID in the form "<prefix>:<name>" — e.g. "lucide:refrigerator", "material-symbols:humidifier-large", "phosphor:speaker-high", "tabler:washing-machine". Prefer these packs in order: lucide, material-symbols, phosphor, tabler. The icons render white on a dark surface — pick line-style monochrome icons, not multi-color packs (avoid emojione, twemoji, fxemoji).
- `icon_emoji`: a single emoji that represents the same object as a fallback (e.g. 💧 humidifier, 🧊 fridge, 📱 iPhone).

Also pick a `category` for this procedure — exactly one value from this closed list: "Assembly", "Electrical", "Calibration", "Maintenance" (scheduled upkeep — cleaning, lubricating, descaling, consumables), "Repair" (unplanned/corrective fix), "Inspection" (check-only), or "Other" (last resort). Troubleshoot-mode SOPs almost always land in "Repair" or "Maintenance" depending on whether the cause is a failure or scheduled upkeep.

Output JSON matching the GeminiProcedureOutput schema."""
