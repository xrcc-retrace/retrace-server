# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

**Retrace Backend Server** — the intelligence layer of the Retrace AI coaching system built for the XRCC Berlin 2026 hackathon.

> One-liner: "Record an expert once. Coach every learner forever."

An expert records a hands-on task via smart glasses (or an iPhone); this server processes the video into a structured procedure with per-step reference clips. When a learner starts a coaching session, the server assembles a system prompt, mints a short-lived Gemini Live token, and returns everything the phone needs to open a direct WebSocket to Gemini — so real-time voice + vision never hop through this server.

Three-tier architecture:

```
Glasses / iPhone  ↔  Phone (iOS)  ↔  Server (this repo)
                          │
                          ↕  (voice + video, direct)
                      Gemini Live API
```

The iOS companion app lives in a separate repo (`retrace-ios`). This server also hosts a static SOP review dashboard at `/` for inspecting and editing procedures without the phone.

## Prerequisites

- Python 3.11+
- `ffmpeg` (`brew install ffmpeg`) — used for clip extraction + duration probing
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put GEMINI_API_KEY in .env, or pass explicitly:
python main.py                           # reads .env
GEMINI_API_KEY=<key> python main.py      # explicit
```

Server listens on `http://0.0.0.0:8000`.

- SOP review dashboard → `http://localhost:8000/`
- From an iOS device on the same Wi-Fi → Bonjour discovery finds the server automatically (see below).

## Architecture

Three main flows: **expert** (record → SOP), **learner** (real-time coaching), and **troubleshoot** (mode-parallel session that diagnoses a broken product and hands off into coaching).

### Expert flow

```
iOS app --[POST multipart video]--> /api/expert/upload
       <--[202 + procedure_id]------
                                    │ (background asyncio.create_task)
                                    ▼
                      ffprobe authoritative duration
                                    │
                                    ▼
                      Gemini Files API upload + poll
                                    │
                                    ▼
                  Gemini 2.5 Pro structured-output JSON
                                    │
                                    ▼
                  ffmpeg per-step clip extraction
                                    │
                                    ▼
                  SQLite (procedures + steps) on disk
```

- Upload returns `202` immediately with `{id, status: "processing"}`.
- Poll `/api/procedures/{id}` until `status` flips to `completed`, `completed_partial` (some clips failed), or `failed`.
- Gemini-reported step timestamps are clamped to the true ffprobe duration before clipping.
- Per-step clip extraction is fault-tolerant — a ffmpeg failure for one step doesn't fail the whole procedure; that step lands with `clip_status="failed"` and the procedure as a whole is marked `completed_partial`.

### Learner flow

```
iOS app --[POST procedure_id, auto_advance, voice]--> /api/learner/session/start
       <--[session + current-step prompt + ephemeral token]----
                                    │
                                    ▼
iOS app <=== direct WebSocket (BidiGenerateContentConstrained) ===> Gemini Live
                                    │
                                    │ (Gemini issues function calls)
                                    ▼
iOS app --[POST tool_name, arguments]--> /api/learner/session/{id}/tool-call
       <--[tool result]-----------------
                                    │ (phone sends result back to Gemini as toolResponse)
                                    │
   (~every 4s, in parallel)         ▼
iOS app --[POST camera JPEG]------> /api/learner/session/{id}/verify
       <--[skip|pending|error|advanced|completed]---   (Flash oracle; advances server-side)
                                    │
       phone injects [ADVANCE] / [CORRECTION] turn into the Live socket
```

- The assembled system prompt embeds **only the current step** — future-step details arrive only via the injected `[ADVANCE]` turn (the check_step toolResponse deliberately carries no next-step content). This prevents the model from narrating ahead.
- Tools: `check_step(claim: done|question)` (client-intercepted — see below), `get_reference_clip` (no args), `go_to_step(step_number)`. `advance_step` is **no longer in Live's toolset** — it survives only as a server dispatcher handler for the iOS HUD swipe and legacy tokens.
- **The oracle is the sole advancement authority.** Two paths into it: (1) the unprompted ~4s poll (Active mode only), and (2) the learner-initiated `check_step` tool — Live calls it when the learner says "done" (claim="done") or asks "is this right?" (claim="question"); the phone intercepts the call (a frame can't ride /tool-call), POSTs its cached camera JPEG to `/verify` with `mode=on_demand`, and synthesizes the toolResponse. Works in Passive mode too. On a confirmed advance the phone defers the `[ADVANCE]` injection until Live finishes speaking its confirmation ("yes, that's right" → then next step). Semantics: insist-once (a second consecutive rejected done-claim advances `method="verbal"`), trust-on-failure (oracle error/timeout on a done-claim advances verbally), `visual_verifiability=none` done-claims advance verbally without the oracle. Live never watches the camera for completion.
- Sessions are **in-memory only** (a `dict` keyed by session id). They die with the process — no DB table.
- When the phone sees Gemini's `goAway` frame (~60 s warning) or a socket error, it calls `POST /session/{id}/token?handle=<sessionResumptionUpdate.new_handle>` to mint a token that reopens the prior session with compressed context intact. On reconnect it injects the result of `GET /session/{id}/context-summary` as a user turn to re-orient the model on the real current step.

### Troubleshoot flow

Mode-parallel to learner — own router, own session dict, own prompt builder, but reuses the same Live config + tool dispatcher.

```
iOS app --[POST voice?]--> /api/troubleshoot/session/start
       <--[session + troubleshoot system prompt + ephemeral token]----
                                   │
                                   ▼
iOS app <=== direct Gemini Live WebSocket ===> Gemini
                                   │
                                   │ tool calls:
                                   │   identify_product → confirm_identification
                                   │   → search_procedures (Gemini-ranked over the library)
                                   │     ↳ no_match → web_search_for_fix
                                   │       (Gemini Pro + GoogleSearch grounding → structured procedure JSON
                                   │        with cited URLs; no PDF anymore)
                                   │   → handoff_to_learner (POST /session/{id}/handoff)
                                   ▼
              learner session opens on the chosen procedure
```

- No procedure is required to start a troubleshoot session. The phase machine progresses `identify → diagnose → fetching-manual → ready-to-coach` (legacy state names; `fetching-manual` now covers `web_search_for_fix`).
- `web_search_for_fix` replaced the old `fetch_manual` PDF flow — the troubleshoot path no longer ingests PDFs at runtime. PDF ingest still exists, but as an expert-side flow under `/api/expert/manuals/*`.
- `handoff_to_learner` (renamed from `handoff_to_coaching`) promotes the chosen procedure into a normal learner session and returns the full `SessionStartResponse` so the iOS client can open a coaching socket without a second round-trip.
- A startup fixture loader ingests a bundled Nespresso Vertuo Next descaling manual on a fresh DB so the demo library isn't empty.

### Expert interviewer flow

A Gemini Live session that runs *alongside* expert recording and asks the questions a learner would have needed (socket size, why, how you know it's tight). Full architecture + edge cases: `docs/expert-interviewer-design.md`.

```
iOS app --[POST voice, transport, task_hint]--> /api/expert/interview/session/start
       <--[session + interviewer prompt + ephemeral token (NO tools) + policy consts]--
                                   │
   every ~5 s while recording      ▼
iOS app --[frame JPEG + 5 s audio chunk + video_t]--> /session/{id}/observe
                                   │  Flash PLANNER: transcribes the chunk, keeps a gap ledger,
                                   │  marks gaps the narration filled, proposes a question
                                   │  policy.py (pure Python) approves/vetoes: time-sensitive only,
                                   │  60 s min interval, 6 live asks max, 0.7 confidence floor, dedup
       <--[{ask?: {question, inject_text, ttl}}]--
                                   │  phone's pause gate waits for silence, then injects
                                   │  "[ASK] …" into Live → Live voices it → mic opens for the answer
iOS app --[question_id, transcript, outcome]--> /session/{id}/answer
   at Stop:  POST /session/{id}/debrief   → leftover gaps, priority-ordered, + "[DEBRIEF]" inject text
   before upload:  POST /session/{id}/finalize → server view of the log with Flash-extracted facts
iOS app --[video + interview (JSON)]--> /api/expert/upload
                                   │  log persisted to data/interviews/{procedure_id}.json
                                   │  extraction prompt += EXPERT INTERVIEW NOTES (facts, Q/A, AI speech windows)
                                   │  clip_extractor ducks the AI's voice (-22 dB) in every clip it intersects
```

- **Live never decides when to speak.** It only hears the mic inside an *answer window* (right after it asks) — if it heard the narration it would reply to every pause, and 3.1 tokens can't carry `proactivity`. The planner is the only thing that hears the whole narration (as audio chunks it transcribes itself).
- **The phone owns the log.** The server session is a planner cache; the phone uploads its `InterviewLog` with the video so a server restart between Stop and Upload loses nothing. Re-extraction (`reprocess.py`) re-reads the on-disk log, so facts survive a pipeline-version bump.
- **Policy is code, not prompt.** `services/expert_interview/policy.py` is pure and covered by `tests/test_interview_policy.py`; the Flash planner only proposes.
- **No tools in v1.** Facts are derived from the Q/A transcript at finalize. `ephemeral_token.build_live_connect_config` emits no `tools` block when the list is empty (an empty `Tool` is rejected by the API).

### Bonjour discovery

On startup the server advertises `_retrace._tcp.local.` on port 8000 via zeroconf. The iOS app's `BonjourDiscovery` populates its cached `serverBaseURL` automatically, with a manual-override UI as the fallback. No hardcoded IPs needed when both devices are on the same Wi-Fi.

## Project Layout

```
main-local-server-test/
├── main.py                    # FastAPI app entry, CORS, Bonjour (lifespan), health, static dashboard
├── config.py                  # Paths, GEMINI_API_KEY, GEMINI_MODEL, RETRACE_LIVE_MODEL, RETRACE_ORACLE_MODEL (default gemini-2.5-flash), FFMPEG_PATH (auto-creates data dirs on import)
├── requirements.txt
├── models/
│   ├── procedure.py           # Per-flow Gemini structured-output schemas (GeminiVideoProcedureOutput / GeminiManualProcedureOutput / GeminiWebProcedureOutput off a shared _BaseProcedureStep) + Procedure / ProcedureStep (API models, with display_summary (condensed ≤160-char lens-card text), completion_criteria, error_criteria, visual_verifiability, clip_status, icon_symbol/icon_emoji/category, sources, parts). visual_verifiability is the `reliable|marginal|none` oracle gate (VisualVerifiability). Plus ProcedureSource (cited web URLs), ProcedurePart (visual parts glossary — video flow only), and StepBoundingBox. source_type ∈ {video, manual, web}
│   ├── session.py             # Pydantic models for session start/state, StepCompletion (method: "visual" | "verbal"), tool call req/res, ephemeral token response
│   ├── interview.py           # Expert-interviewer models: session start/observe/answer/debrief wire shapes, PlannerDecision (Flash output), Ask (policy output), InterviewLog (phone-owned, strict — extra="forbid" + required version)
│   └── troubleshoot.py        # Troubleshoot-mode request/response models (session start, state, handoff, manual upload/status, identified product, candidate procedure, diagnostic log entry)
├── routers/
│   ├── expert.py              # POST /api/expert/upload (multipart `video` + optional `interview` JSON log) — async background pipeline, timestamp clamping, per-step status bookkeeping
│   ├── expert_interview.py    # /api/expert/interview/* — start/token/observe/answer/debrief/finalize/state. Planner tick lives in `observe` (single in-flight, idle-skip gate)
│   ├── expert_manuals.py      # POST /api/expert/manuals/upload + GET /api/expert/manuals/{id} (PDF → SOP) + GET /api/manual_images/{procedure_id}/step_{n}.png (rendered manual page used as the step "clip")
│   ├── icons.py               # GET /api/icons/{prefix}/{name}.png — Iconify icons rendered as white-on-transparent PNGs, disk-cached
│   ├── learner.py             # Voices list + session lifecycle (start, tool-call forwarding, state, context summary, token refresh with optional ?handle=)
│   ├── procedures.py          # CRUD for procedures / steps + clip serving
│   └── troubleshoot.py        # Mode-parallel to learner. Importing this router imports services.troubleshoot, which registers its tool handlers with the shared dispatcher.
├── services/
│   ├── gemini_processor.py    # Uploads video to Gemini Files API, polls, generates structured procedure JSON via Gemini 2.5 Pro (steps + procedure-level `parts` visual glossary for live piece disambiguation). Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step MP4 extraction via ffmpeg subprocess (H.264+AAC, faststart, bt709 colr tags, bf=0 to keep DTS monotonic)
│   ├── video_probe.py         # ffprobe wrapper — authoritative duration read on upload
│   ├── database.py            # SQLite (WAL). Tables: procedures, steps. JSON-serialized tips/warnings. Lightweight ALTER-TABLE migrations for new columns (source_type on procedures, etc.).
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher (shared by learner + troubleshoot; handlers register at import time)
│   ├── prompt_builder.py      # Current-step-only system prompt. Live is conversation-only: an "EVENT SIGNALS" block tells it to act on injected [ADVANCE]/[CORRECTION] turns (produced by the out-of-band oracle), not to watch the camera itself. Embeds a PARTS GLOSSARY (procedure.parts — all parts listed; objects don't leak future steps) + a VISUAL HONESTY anti-sycophancy block (judge "is this the right piece?" from the frame/glossary; ask for a closer view rather than guess; never confirm on assertion alone). advance_step is verbal-only. Tool definitions + Gemini config. (The old "[check] heartbeat / VISUAL MONITORING" block was removed.) _ERROR_CRITERIA_OVERRIDES is now only read by scripts/backfill_error_criteria.py — the prompt no longer embeds error_criteria (the oracle consumes them server-side).
│   ├── step_oracle.py         # Out-of-band visual verification. gemini-2.5-flash (RETRACE_ORACLE_MODEL) compares the learner's current frame vs the step's reference "done" frame + completion_criteria + error_criteria → {complete, confidence, error_index}. Biased against false positives. Backs POST /session/{id}/verify.
│   ├── ephemeral_token.py     # Mints v1alpha auth_tokens locked to model + system prompt + tools + voice + transcription + compression + resumption handle
│   ├── procedure_search.py    # Gemini Flash-ranked search over the existing procedure library by product + symptom (no separate index)
│   ├── metadata_picker.py     # Flash (gemini-2.5-flash, thinking_budget=0) call that picks a procedure's icon_symbol + category from a closed enum. Used by all four generation flows (expert video, expert PDF, troubleshoot PDF, web search).
│   ├── manual_ingest.py       # PDF → structured SOP via Gemini Files API. Shared by expert + troubleshoot via mode={"expert","troubleshoot"}. Renders source_page → PNG in MANUAL_IMAGES_DIR; writes path into step.clip_url so manual procedures plug into existing clip endpoints unchanged.
│   ├── manual_prompts.py      # Mode-specific Gemini prompts for manual_ingest
│   ├── pdf_images.py          # PDF page → PNG rasterization
│   ├── icon_renderer.py       # Iconify SVG → white-on-transparent PNG via cairosvg, disk-cached by Iconify ID
│   ├── voice_commands.py      # Loads voice_commands.yaml at import; prompt builders look up phrasing by command id
│   ├── fixtures.py            # Startup loader that ingests the bundled Nespresso Vertuo Next descaling manual on a fresh DB. Scheduled non-blocking from main.py's lifespan.
│   ├── expert_interview/      # Expert interviewer mode (no tool handlers — nothing registers on import)
│   │   ├── session.py         # Ledger (gaps), Q/A log, transcript, live-ask budget on the shared session dict; to_log() → InterviewLog
│   │   ├── planner.py         # One Flash call per observe tick (RETRACE_PLANNER_MODEL): transcribe chunk + propose gaps. Idle-skip gate (frame digest + audio RMS). PCM→WAV helper
│   │   ├── policy.py          # PURE deterministic decisions: dedup, budget, min interval, confidence floor, time-sensitivity, debrief ordering, inject-text formats. Unit-tested
│   │   ├── prompt.py          # Interviewer Live persona (speaks only on [BRIEFING]/[ASK]/[DEBRIEF] turns); INTERVIEW_TOOL_DEFINITIONS = []
│   │   ├── facts.py           # Flash: Q/A + transcript → atomic Fact list (at finalize, or at extraction if missing)
│   │   ├── notes.py           # format_interview_notes(log) → the EXPERT INTERVIEW NOTES prompt section ("" when nothing to say)
│   │   └── store.py           # data/interviews/{id}.json read/write, duck_windows(), strict parse_upload()
│   └── troubleshoot/
│       ├── session.py         # In-memory troubleshoot store, phase transitions (identify → diagnose → fetching-manual → ready-to-coach), diagnostic log
│       ├── prompt.py          # Troubleshoot-specific system prompt
│       ├── handlers.py        # Tool handlers: identify_product, confirm_identification, search_procedures, web_search_for_fix, handoff_to_learner. Registers with the shared dispatcher at import time.
│       └── web_research.py    # Two-step Gemini Pro pipeline: GoogleSearch grounding (free-form research brief) + structured-output (procedure JSON with cited source URLs). Backs web_search_for_fix; replaced the old fetch_manual flow.
├── static/
│   ├── index.html             # SOP review dashboard
│   └── voice_previews/        # Pre-generated WAV previews of Gemini voices
├── scripts/
│   ├── generate_voice_previews.py   # Regenerate the WAV previews
│   ├── backfill_error_criteria.py   # Backfill steps.error_criteria from prompt_builder._ERROR_CRITERIA_OVERRIDES
│   ├── probe_interject_heartbeat.py # Empirical probe for the "Interject" heartbeat feature on 3.1 Live
│   ├── probe_live_tool_config.py    # Raw-WebSocket probe of tool_config.mode=ANY (the SDK config object forbids it)
│   └── prompts/                     # Prompt fixtures for the interject probes (interject_silence_v1/v2/v3.txt, interject_trigger.txt)
├── gemini_live_playground/    # Interactive REPL (`python -m gemini_live_playground`) for behavioral tuning against the production prompt/tools/config. Text I/O via output-transcription stream; tool calls dispatch in-process against services.session. No FastAPI server required. Scenarios in scenarios/*.yaml.
├── reprocess_clips.py         # Re-run clip extraction for completed procedures without hitting Gemini again (after tweaking ffmpeg flags)
├── reset_data.py              # Wipe all user-generated data (DB rows, uploads, clips) — interactive by default, --yes / --dry-run / --force flags. Preserves code, .venv, .env, voice_previews.
├── backfill_icons.py          # One-shot: render and cache Iconify PNGs for procedures predating the icon system
├── backfill_summaries.py      # One-shot: Flash-condense step descriptions into display_summary (lens glance text) for procedures predating the field — the deliberate alternative to an extraction-version bump
├── validate_voice_commands.py # Lint voice_commands.yaml (schema, uniqueness, referenced-from-code checks)
├── voice_commands.yaml        # Single source of truth for prompt utterances; designer-editable
├── tests/
│   └── test_interview_policy.py  # stdlib unittest: interview policy gates, dedup, debrief ordering, ducking filter rebasing, notes formatter, strict log parsing
├── test_live_session.py       # Ad-hoc integration test that drives a Gemini Live session against a running server
├── probe_live_models.py       # Which Live model IDs bidiGenerateContent accepts on this API key, plus per-model probes of proactive_audio and video input (curated candidate list)
├── probe_list_models.py       # Full models.list() across v1alpha/v1beta/v1 — authoritative source of truth when probe_live_models.py's candidate list comes up empty
├── probe_all_models.py        # Same dump with capability filters (generateContent, bidiGenerateContent, embedContent, …)
├── probe_retrace_config.py    # End-to-end verification of the production ephemeral-token config against a real Live session
├── probe_31_config.py         # Bisect which LiveConnectConfig field the current model rejects — used when a model upgrade breaks the baseline config
├── probe_gemini_live_system_instruction.py  # Canary probe: verify Gemini Live honors the system_instruction baked into ephemeral tokens (motivated by the open Google Developers forum report)
├── probe_tool_reliability.py  # HIT/MISS rate for advance_step tool calls over many trials — decide if tool calling is reliable enough or if we need the out-of-band transcript fallback
├── probe_diagnostic_reliability.py  # Analog of probe_tool_reliability.py for troubleshoot-mode tool calls (identify_product, confirm_identification, search_procedures, web_search_for_fix, handoff_to_learner)
├── fixtures/                  # Bundled manuals; only nespresso_vertuo_next_descaling.pdf is wired into the startup loader (levoit_lv600hh_manual.pdf also present but unused)
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4   # plus on-demand step_{n}.jpg (first-frame poster) and step_{n}_done.jpg (last-frame "done state", oracle reference) cached beside each clip
│   ├── manual_images/{procedure_id}/step_{n}.png   # MANUAL_IMAGES_DIR — rendered manual-page PNGs for manual-sourced procedures
│   ├── interviews/{procedure_id}.json              # INTERVIEWS_DIR — phone-uploaded expert-interview logs (re-read on every re-extraction)
│   └── retrace.db             # SQLite (WAL mode)
└── uploads/                   # Source expert videos ({uuid}.{ext})
```

## API Reference

### Health & static

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/` | SOP review dashboard. |
| `GET` | `/api/uploads/{filename}` | Serve original source video for review playback. |
| `GET` | `/static/voice_previews/{name}.wav` | Pre-generated voice preview. |

### Expert — `/api/expert`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/expert/upload` | Multipart `video` upload, plus optional `interview` part (JSON `InterviewLog` from the expert interviewer — strict: unknown keys or a missing `version` are a 400, so a stray JSON object can't masquerade as an empty interview). Returns `202 {id, status: "processing", has_interview}`; background task runs Gemini analysis + clip extraction. Poll `/api/procedures/{id}` for final status. |

### Expert interviewer — `/api/expert/interview`

Runs alongside recording; see "Expert interviewer flow" above and `docs/expert-interviewer-design.md`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/expert/interview/session/start` | Body: `{voice?, transport?, task_hint?}`. Returns session id, interviewer prompt, **empty** tool list, Gemini config, ephemeral token, and the `policy` constants (min interval, max live asks, TTL, answer-window lengths, silence threshold) the phone mirrors. |
| `POST` | `/api/expert/interview/session/{id}/token[?handle=...]` | Resumption-aware token refresh, same semantics as the other modes. |
| `GET` | `/api/expert/interview/session/{id}/briefing-inject` | `{inject_text}` for the optional one-question pre-roll briefing. |
| `POST` | `/api/expert/interview/session/{id}/observe` | Multipart: `frame` (JPEG, optional), `audio` (WAV or raw PCM16 mono, optional), `sample_rate`, `video_t` (seconds; null in pre-roll), `phase` (`preroll`/`recording`/`debrief`), `speaking` (phone VAD). Returns `status` (`ok` / `quiet` / `skipped` / `planner_unavailable` / `finalized`), an optional `ask {question_id, topic_key, priority, question, ttl_seconds, inject_text}`, counters, and `open_answer_window` when the expert addressed the assistant. Single in-flight tick; idle ticks (unchanged frame + silent audio) skip the model call. |
| `POST` | `/api/expert/interview/session/{id}/answer` | Body: `{question_id, transcript, outcome: answered\|unanswered\|skipped\|interrupted\|deferred, t_ask?, t_close?}`. Closes the gap; starts the min-interval clock; "stop asking" in the transcript flips quiet mode. |
| `POST` | `/api/expert/interview/session/{id}/debrief` | Body: `{max_questions?}`. Priority-ordered leftover gaps + `[DEBRIEF]` inject text. Idempotent. |
| `POST` | `/api/expert/interview/session/{id}/finalize` | Server view of the `InterviewLog` with Flash-extracted `facts`. The phone merges this with its own copy (which also carries `ai_speech_windows`) and uploads the result. |
| `GET` | `/api/expert/interview/session/{id}/state` | Phase, quiet mode, budget, full gap ledger, Q/A. |
| `DELETE` | `/api/expert/interview/session/{id}` | End and discard. |
| `GET` | `/api/expert/interview/sessions` | List active interview sessions. |
| `POST` | `/api/expert/manuals/upload` | Multipart PDF manual upload. Returns `202 {manual_id}`; background task runs `services.manual_ingest.process_pdf(mode="expert")` → new procedure with `source_type="manual"`. Per-step manual-page PNG is written to `step.clip_url` so the iOS client renders it where it would normally render a clip. |
| `GET` | `/api/expert/manuals/{manual_id}` | Poll ingest status + (when ready) the resulting `procedure_id`. |
| `GET` | `/api/manual_images/{procedure_id}/step_{n}.png` | Rendered manual-page PNG used in place of an MP4 clip for manual-sourced procedures. |

### Icons — `/api/icons/{prefix}/{name}.png`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/icons/{prefix}/{name}.png` | Iconify icon (e.g. `/api/icons/lucide/refrigerator.png`) rasterized to a white-on-transparent PNG via `cairosvg`, cached on disk. A miss serves a fallback icon; a malformed prefix/name returns `400`. |

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all (newest first). Includes `status` (`processing` / `completed` / `completed_partial` / `failed`) and `step_count`. |
| `GET` | `/api/procedures/{id}` | One procedure with all steps (includes `completion_criteria`, `clip_status`). |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, and/or reorder steps via `step_order: [old_step_number, ...]`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields (`title`, `description`, `display_summary`, `tips`, `warnings`, `timestamp_start`, `timestamp_end`, `error_criteria`, `visual_verifiability`). |
| `DELETE` | `/api/procedures/{id}` | Delete procedure + cascade steps. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |
| `GET` | `/api/clips/{id}/step_{n}.jpg` | On-demand, disk-cached first-frame JPEG poster of the step clip (for the lens-side `MWDATDisplay.Image` thumbnail, which can't take video frames). |
| `GET` | `/api/clips/{id}/step_{n}_done.jpg` | On-demand, disk-cached **last-frame** JPEG ("done state") of the step clip — the verification oracle's image-to-image reference. Pre-generated at extraction; regenerated on demand. |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices (`{name, description, preview_url}`). iOS reads this instead of hardcoding. |
| `POST` | `/api/learner/session/start` | Body: `{procedure_id, auto_advance?, active_monitoring?, voice?}`. Returns session id, full procedure, assembled (current-step-only) system prompt, tool definitions, Gemini config, ephemeral token. `active_monitoring` (default True) is the master switch — when False the phone doesn't poll `/verify` and the coach is fully reactive (no interjection, no visual advance, verbal-only). `auto_advance` (default True) gates visual step advance, effective only when `active_monitoring` is also True. **The iOS app collapses these into one Active/Passive toggle and always sends them equal**; the "active + manual" middle state is reachable only via the playground/probes (`--manual`). |
| `POST` | `/api/learner/session/{id}/verify` | Multipart body: JPEG file part `frame`, plus optional `mode` (`poll` default / `on_demand`) and `claim` (`done`/`question`, required iff `on_demand`). Poll mode: dwell + toggle gates, returns `skip`/`pending`/`error`(`+error_text`)/`advanced`/`completed`. On-demand mode (backs the client-intercepted `check_step` tool; bypasses `active_monitoring`/`auto_advance`/dwell/dedup, keeps confidence floor + race guard): adds `not_confirmed`(`+reason`), `unverifiable`, `check_failed`, `already_completed`; `advanced`/`completed` carry `method` (`visual`/`verbal`) + optional `note`. Insist-once + trust-on-failure semantics live here. The phone injects the outcome into Live as `[ADVANCE]`/`[CORRECTION]` (on-demand advances defer the injection until Live's confirmation finishes). |
| `POST` | `/api/learner/session/{id}/token[?handle=...]` | Mint a fresh ephemeral token. Pass the latest `sessionResumptionUpdate.new_handle` as `?handle=` to bind the new token to the prior session — the next WebSocket continues with compressed context intact. The regenerated system prompt re-anchors on the session's current step, so a resumed socket doesn't reopen on step 1. |
| `POST` | `/api/learner/session/{id}/tool-call` | Body: `{tool_name, arguments}`. Executes the handler and returns `{result: {...}}`. The phone unwraps `result` and hands it back to Gemini via `toolResponse`. |
| `GET` | `/api/learner/session/{id}/state` | Current step, status, step history (each entry includes `method: "visual" | "verbal"`). |
| `GET` | `/api/learner/session/{id}/context-summary` | Compact text blob the phone injects as a user message after context compression or reconnect to re-orient the model. |
| `DELETE` | `/api/learner/session/{id}` | End and discard the session. |

### Troubleshoot — `/api/troubleshoot`

Mode-parallel to learner — same Live config and dispatcher, no procedure required at start.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/troubleshoot/session/start` | Body: `{voice?}`. Returns session id, troubleshoot system prompt, tool definitions, ephemeral token. |
| `POST` | `/api/troubleshoot/session/{id}/token[?handle=...]` | Resumption-aware token refresh, same semantics as the learner endpoint. |
| `POST` | `/api/troubleshoot/session/{id}/tool-call` | Tools: `identify_product`, `confirm_identification`, `search_procedures`, `web_search_for_fix`, `handoff_to_learner`. |
| `GET`  | `/api/troubleshoot/session/{id}/state` | Current phase, identified product, candidate procedures, diagnostic log. |
| `POST` | `/api/troubleshoot/session/{id}/handoff` | Promote a chosen `procedure_id` into a fresh learner session; returns the full `SessionStartResponse`. |
| `DELETE` | `/api/troubleshoot/session/{id}` | End and discard. |
| `GET` | `/api/troubleshoot/sessions` | List active troubleshoot sessions. |

## Gemini Live config

These settings are locked into every ephemeral token so the phone can't change them:

- **Model:** `gemini-3.1-flash-live-preview` (released 2026-03-26, public preview) — verified 2026-04-17 via the production ephemeral-token + `BidiGenerateContentConstrained.v1alpha` path. Supports video input, tools, context compression, session resumption, in/out transcription. Override via `RETRACE_LIVE_MODEL` env var for fast rollback to 2.5 native-audio. `turn_coverage` default flips to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO` on 3.1, so silent-period video frames now reach the model.
- **Voice:** server-authoritative list (Puck / Charon / Kore / Fenrir / Aoede / Zephyr / Enceladus / Leda). Default `Puck`.
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. (The "~25 min then ~12 min" cadence in older notes was derived under `MEDIA_RESOLUTION_LOW`; the config now runs `MEDIA_RESOLUTION_MEDIUM`, so per-frame token burn sits between the LOW and HIGH regimes — treat the old minute figures as unverified.)
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. iOS persists the latest; on reconnect, `POST /session/{id}/token?handle=...` bakes it into `SessionResumptionConfig` so the next WebSocket continues where the prior one left off.
- **Transcription:** both input (learner) and output (AI reply) audio are transcribed so the iOS activity feed renders text.
- **Media resolution:** `MEDIA_RESOLUTION_MEDIUM` (`services/ephemeral_token.py`) — balances visual-verification detail against token thrift. Raising/lowering this directly moves the compression cadence.
- **Token uses:** `uses=1`. Matches documented single-use semantics. The iOS client mints a fresh token for every connection attempt (initial + retry + resumption), so this is sufficient.
- **Expiry format:** ISO8601 with **millisecond** precision (`timespec="milliseconds"`). iOS's `ISO8601DateFormatter.withFractionalSeconds` caps at 3 fractional digits — Python's default 6-digit microsecond output silently fails to parse and forces spurious refreshes.

## Data Storage

- `uploads/` — source expert videos, named `{uuid}.{ext}`. Keep the original extension so ffprobe reads the container correctly.
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips.
- `data/retrace.db` — SQLite (WAL). Two tables:
  - `procedures(id, title, description, total_duration, source_video, created_at, status, error_message, source_type, icon_symbol, icon_emoji, category, sources, parts)` — the trailing columns are added via lightweight ALTER-TABLE migrations. `parts` is a JSON array of `{name, appearance}` visual-glossary entries (video flow only; `'[]'` elsewhere).
  - `steps(id, procedure_id, step_number, title, description, display_summary, timestamp_start, timestamp_end, tips, warnings, completion_criteria, error_criteria, visual_verifiability, clip_url, clip_status)` — `error_criteria` is a JSON array of observable error states (see Interject below); `visual_verifiability` is `reliable|marginal|none` (the oracle gate; legacy rows default `reliable`); `display_summary` is the condensed (≤160 char) lens-card text — produced by all three extraction flows, backfilled for legacy rows by `backfill_summaries.py`, and never fed to the Live prompt (Live keeps the full description).
  - `tips` and `warnings` are JSON-encoded arrays.
  - `UNIQUE(procedure_id, step_number)` — reorder logic uses a two-pass negative-then-positive trick to avoid unique-constraint violations.
- Sessions are **in-memory only** (a `dict` in `services/session.py`). They die with the process.

## Testing

Mostly manual testing via curl and the dashboard at `/`. The one unit-test suite covers the expert-interview policy (pure Python, no Gemini):

```bash
python -m unittest tests.test_interview_policy -v
```

```bash
# Health
curl http://localhost:8000/health

# Upload (returns 202 immediately; poll /api/procedures/{id} until status != processing)
curl -X POST -F "video=@test_data/some_video.mp4" http://localhost:8000/api/expert/upload

# List procedures
curl http://localhost:8000/api/procedures

# Start a learner session
curl -X POST -H "Content-Type: application/json" \
  -d '{"procedure_id":"<id>"}' http://localhost:8000/api/learner/session/start
```

Sample videos live under `test_data/`. `test_live_session.py` drives a full Live session against a running server. The probe scripts (`probe_live_models.py`, `probe_list_models.py`, `probe_retrace_config.py`, `probe_31_config.py`, `probe_gemini_live_system_instruction.py`, `probe_tool_reliability.py`) are the source of truth for "which models/features work right now on this API key" — keep them around after a model upgrade rather than rewriting from scratch. `gemini_live_playground/` is the fastest way to iterate on prompt/tool behavior without rebuilding the iOS app.

## Critical Architecture Decisions

1. **Voice + video go direct**, phone → Gemini Live. Routing them through this server adds 100-200 ms per exchange and breaks conversational feel. The server is strictly the control plane.
2. **Visual verification runs out-of-band in a Gemini Flash oracle**, not inside Gemini Live — and the oracle is the **sole advancement authority**. Live stays conversation-only (it keeps a low-res feed for the learner's part-identification questions). Two paths into the oracle: (a) the unprompted ~4s poll (`mode=poll`, Active mode only — dwell reliable=1/marginal=2, toggles enforced, `[CORRECTION]` interjection); (b) the learner-initiated `check_step(claim=done|question)` Live tool, client-intercepted on the phone and routed to `/verify mode=on_demand` with the cached frame — works in Passive mode, bypasses dwell/toggles, keeps the 0.6 confidence floor + race guard. `services/step_oracle.py` (`RETRACE_ORACLE_MODEL`, 10s timeout) compares the frame against `completion_criteria` — grounded on `step_N_done.jpg` — plus `error_criteria` and the parts glossary, returning `{complete, confidence, error_index, reason}`. On confirm the server calls `advance_step(method="visual")`; the phone injects `[ADVANCE]`/`[CORRECTION]` into Live (on-demand advances defer the injection until Live finishes voicing its confirmation). Learner authority: insist-once (second rejected done-claim advances verbally), trust-on-failure (oracle unavailable on a done-claim advances verbally), `visual_verifiability=none` done-claims advance verbally with no oracle call. The Live `advance_step` tool was removed (the server handler survives for the HUD swipe + legacy tokens); the old `[check]` heartbeat is long gone.
3. **Audio + video share one Live session.** Context compression + session resumption unlock indefinite duration. The "2-minute audio+video" Gemini limit is a context-fill limit, not a hard timer.
4. **Current-step-only system prompt.** `build_system_prompt` embeds just the current step's details; future steps are delivered exclusively through `advance_step` tool responses. Prevents the model from narrating ahead. On resumption, the prompt is re-anchored on `session["current_step"]` so a reopened socket doesn't act like step 1.
5. **Tool calling is the model↔server channel.** Gemini issues function calls; the phone intercepts, injects `session_id`, and forwards to `/tool-call`. Server handler result is returned to the phone wrapped as `{result: {...}}`; the phone unwraps before handing back to Gemini.
6. **Ephemeral tokens** solve API key distribution. The phone never sees the raw key. Tokens are locked to a specific model / prompt / tool set / voice / transcription / compression / resumption handle, so a compromised token can't be repurposed.
7. **Bonjour is the canonical discovery path.** The Wi-Fi IP can change; hardcoded IPs would rot across hackathon networks. Manual override exists as a fallback in iOS Settings.
8. **Per-step clip failures are tolerated.** Gemini occasionally produces step timestamp ranges that ffmpeg can't cut. The procedure is saved as `completed_partial` with failed steps marked `clip_status="failed"` rather than erroring the whole upload.

## Visual Verification Reliability

Design steps around what VLMs can actually detect:

- **Reliable:** Object presence/absence, large state changes (lid open/closed, tray in/out).
- **Marginal:** Small object state, correct orientation.
- **Unreliable:** Fine alignment, sub-millimeter changes, torque/fit.

The `completion_criteria` string on each step is what the **verification oracle** (`services/step_oracle.py`) matches the learner's frame against — image-to-image, grounded on the expert clip's last frame (`step_N_done.jpg`). Write criteria that describe an **observable single-frame spatial end-state** ("the glasses are on top of the keyboard"), not an action. The extraction prompt in `gemini_processor.py` asks Pro to phrase them that way and to set `visual_verifiability` (`reliable|marginal|none`) from the spatial delta between the prior step's end and this step's end — `none` marks steps with no single-frame visual consequence (counting/timing/torque), which stay verbal-only.

**Proactive error interjection (Interject).** Each step also carries `error_criteria` — up to three observable *wrong-state* descriptions (empty is fine; the old "exactly three" quota was relaxed to stop paraphrase padding). The oracle tests the current frame against them on each `/verify` poll and, on a clear match, returns `error_text`; the phone injects a `[CORRECTION]` turn into Live. Interjection stays alive even on verbal-only steps. Same VLM-detectability rules: phrase each error as a clearly visible state ("the cap is still on", "the tray is upside down"). `error_criteria` is produced by Gemini 2.5 Pro at extraction; `scripts/backfill_error_criteria.py` seeds legacy rows. (Earlier in-band `[check]` heartbeat experiments live in `scripts/probe_interject_heartbeat.py` + `scripts/prompts/` — historical, that path was replaced by the oracle.)

**Procedure metadata.** `services/metadata_picker.py` makes a cheap Flash call (`thinking_budget=0`) at the end of every generation flow to pick an `icon_symbol` (Iconify id, served via `/api/icons/...`) and a `category` (closed `ProcedureCategory` enum) for the procedure.

## Gotchas

- **`config.py` creates data dirs on import.** Anything that imports `config` (main, routers, services) will touch disk. This is intentional for first-run setup — don't work around it by deferring imports.
- **Gemini Files API uploads must be deleted** or they count against your storage quota. `gemini_processor.process_expert_video` does this in a `finally` block.
- **ffmpeg flags matter for iOS HD playback.** `-bf 0` keeps PTS == DTS (iOS rejects negative DTS); `-x264opts colorprim=bt709:transfer=bt709:colormatrix=bt709` writes the SEI tags VideoToolbox needs or it complains with `verify_image_parameters` errors. Don't remove these.
- **Don't commit `.env`.** Put real API keys there and let `python-dotenv` load them.
- **Session memory leak hazard.** The iOS client fires a best-effort `DELETE /session/{id}` on session end. If that request is lost, the in-memory dict grows until the server restarts. It's a hackathon tradeoff — don't "fix" by persisting sessions unless the Live-session lifecycle also moves to DB.
- **Prompt changes require re-minting tokens.** System prompts are baked into the ephemeral token. Updating `prompt_builder.py` only affects sessions started after the change; in-flight tokens still carry the old prompt.
- **`generate_content` rejects deadlines under 10 s.** `HttpOptions(timeout=…)` below 10_000 ms fails with `400 "Manually set deadline 8s is too short. Minimum allowed deadline is 10s"` (verified 2026-10-15). The oracle's and planner's 10_000 are the floor, not a tuning knob — don't lower them.
- **`echo "$json"` in zsh mangles JSON.** zsh's `echo` expands `\n` escapes inside strings; pipe curl straight to a file or `python -m json.tool` when inspecting API responses.
