<div align="center">

<img src="https://github.com/xrcc-retrace/.github/raw/main/profile/assets/retrace-banner.svg" alt="Retrace" width="100%" />

# Retrace — Backend Server

### Record an expert once. Coach every learner forever.

The FastAPI backend for **Retrace**, an AI coaching system built for the **XRCC Berlin 2026** hackathon.
The control plane that turns expert videos into structured procedures, mints ephemeral Gemini Live tokens for the phone, and dispatches tool calls during a coaching session.

🥈 **2nd place — XRCC Berlin 2026.** Development closed out in July 2026; this repo is preserved as a complete, working snapshot.

[**▶ Watch the demo**](https://www.youtube.com/watch?v=azK-CLpyRQg) &nbsp;·&nbsp; [**🎬 Full walkthrough video**](https://www.youtube.com/watch?v=CqUj5HtP0QE) &nbsp;·&nbsp; [**🌐 Live API**](https://retracexrcc.duckdns.org/health) &nbsp;·&nbsp; [**🏛 Org overview**](https://github.com/xrcc-retrace) &nbsp;·&nbsp; [**📱 iOS repo**](https://github.com/xrcc-retrace/retrace-ios)

</div>

---

## What it does

Retrace turns one expert demonstration into hands-free AI coaching for everyone else. This backend is the intelligence layer behind the product's headline features:

- **Record once, coach forever.** An expert films a task once (smart glasses or iPhone); Gemini 2.5 Pro turns the video into a structured procedure — ordered steps, per-step reference clips, tips, warnings, and machine-checkable *completion criteria*.
- **A coach that watches your hands.** During a session, a server-side visual oracle compares the learner's live camera frame against each step's expected end-state and advances the procedure only when the work is *verifiably* done. The learner never taps "next".
- **"Am I done?" — actually checked.** When the learner claims completion or asks "is this right?", the oracle verifies the claim against the live frame and the coach confirms or corrects based on visual evidence, not politeness.
- **Catches mistakes before you compound them.** Every step carries observable wrong-states; if the learner's frame matches one, the coach interjects with a correction unprompted.
- **Fixes things nobody recorded.** Troubleshoot mode identifies a broken product on camera, searches the procedure library, and — if nothing matches — synthesizes a fresh, citation-backed repair procedure from live web research, then coaches it.
- **Any manual becomes a coach.** Upload a PDF manual and it becomes a coachable procedure, with the source page rendered as each step's visual reference.

The rest of this README covers how each of these is built.

---

## Where this sits in Retrace

```
  Capture          Phone (iOS)                   Backend (this repo)
┌─────────┐      ┌──────────────────┐       ┌────────────────────┐
│ Ray-Ban │      │ Capture          │       │ Procedure pipeline │
│ Meta    │ ───► │ orchestrator     │ ◄───► │ Session control    │
│ glasses │      │ Gemini Live      │  REST │ Tool dispatcher    │
│   OR    │      │ client           │       │ Token minting      │
│ iPhone  │      │ Token actor      │       │ Bonjour discovery  │
└─────────┘      └────────┬─────────┘       └────────┬───────────┘
                          │                          │
                          │ direct WebSocket         │ Files API
                          │ (BidiGenerateContent     │ + Pro REST
                          │  Constrained, v1alpha)   │
                          ▼                          ▼
                ┌───────────────────────────────────────────┐
                │           Gemini API (Google)             │
                │  Live (3.1 Flash) · 2.5 Pro · 2.5 Flash   │
                └───────────────────────────────────────────┘
```

The phone holds a **direct WebSocket** to Gemini Live — voice and video never hop through this server. The backend handles the offline intelligence: extracting structured procedures from expert videos, slicing per-step clips, building current-step-only system prompts, minting short-lived ephemeral tokens, dispatching tool calls, and recovering sessions through context compression and resumption.

> Full system architecture, the cross-cutting design decisions, and the iOS counterpart live in the **[org profile README](https://github.com/xrcc-retrace)**. This README focuses on the backend.

---

## Highlights

- **Three flows in one server.** Expert ingest (video → SOP), Learner coaching (current-step prompt + ephemeral token), Troubleshoot (no-procedure-required diagnose-then-coach with web grounding).
- **Structured-output procedure extraction** — Gemini 2.5 Pro returns a typed `GeminiProcedureOutput` (title, ordered steps, per-step `completion_criteria`, tips, warnings). Falls back to 2.5 Flash on 503. Timestamps are clamped against the true `ffprobe` duration before clipping so Gemini-imagined ranges can't break ffmpeg.
- **Per-step clip extraction** — H.264+AAC, faststart, BT.709 color tags, `bf=0` for monotonic DTS — the flag set iOS VideoToolbox actually accepts. Per-step ffmpeg failures land the procedure as `completed_partial` instead of failing the whole upload.
- **Direct-to-Gemini ephemeral tokens** — `v1alpha auth_tokens` minted server-side, locked to a specific model + system prompt + tool set + voice + transcription + compression + resumption handle. The phone never sees the API key.
- **Indefinite session length** — context-window compression (trigger 100k → target 40k) plus session resumption via `sessionResumptionUpdate.new_handle`. The "2-minute audio+video" Gemini cap is a context-fill limit, not a hard timer.
- **Current-step-only system prompt** — only the current step is embedded; future steps reach the model exclusively through injected `[ADVANCE]` turns (the `check_step` tool response deliberately carries zero next-step content, so Live can't narrate ahead). Re-anchors on the session's current step on token refresh / resumption.
- **Out-of-band verification oracle** — visual step completion is decided by a separate `gemini-2.5-flash` oracle (`services/step_oracle.py`), **not** inside Gemini Live. The phone polls `POST /session/{id}/verify` with a camera frame every ~4 s; the oracle compares it against the step's `completion_criteria` + `error_criteria`, grounded image-to-image on the expert clip's last frame (`step_N_done.jpg`), biased against false positives. On a confirmed completion the server advances itself (`method="visual"`); on a visible mistake it returns a correction. The phone injects the result into Live as an `[ADVANCE]`/`[CORRECTION]` turn — always a verified event, never a hallucinated one. A per-step `visual_verifiability` gate (`reliable|marginal|none`) decides whether the oracle runs at all; `none` steps (counting/timing/torque) stay verbal-only.
- **On-demand verification (`check_step`)** — the same oracle answers the learner directly. When they say "done" or ask "is this right?", Live calls `check_step`; the phone intercepts it (a frame can't ride the tool-call body), POSTs its latest cached frame to `/verify mode=on_demand`, and synthesizes the tool response from the verdict. Insist-once (a second rejected done-claim advances verbally) and trust-on-failure (oracle outage advances verbally) keep the learner in charge.
- **Evals, not vibes** — an `evals/` harness scores the oracle and the extractor against **self-supervised ground truth** harvested from the expert clips' own first/last frames. Prompt iterations were accepted on measured false-positive/recall movement — one tightening pass drove completion false-positives from 31% to 0%. See [`evals/README.md`](evals/README.md).
- **Proactive error interjection (Interject)** — each step carries `error_criteria` (observable wrong-states) that the same `/verify` poll tests every cycle; a clear match returns `error_text` the phone speaks as a `[CORRECTION]`. Stays alive even on verbal-only steps.
- **Procedure metadata** — `services/metadata_picker.py` runs a cheap Flash call (`thinking_budget=0`) at the end of every generation flow to pick an Iconify `icon_symbol` + a closed-enum `category` for each procedure.
- **Shared tool dispatcher** — both Learner and Troubleshoot register handlers with a single dispatcher at import time. The phone forwards tool calls; the server returns `{result: {...}}` for the phone to hand back to Gemini as `toolResponse`.
- **Web-grounded Troubleshoot** — `web_search_for_fix` runs a two-step Gemini Pro pipeline (GoogleSearch grounding + structured-output JSON) that returns a fresh procedure with cited URLs. Replaced an earlier brittle PDF-fetch path.
- **PDF manual ingest** — `/api/expert/manuals/upload` runs the same structured-output extraction over a PDF and rasterizes each step's source page to a PNG, written into `step.clip_url` so manual-sourced procedures plug into the same clip-serving endpoints as video-sourced ones.
- **Iconify icon library** — `GET /api/icons/{prefix}/{name}.png` serves any Iconify ID as a white-on-transparent PNG via cairosvg, disk-cached. The iOS HUD references icons without bundling assets.
- **Bonjour LAN discovery** — `_retrace._tcp.local.` advertised on port 8000; iOS auto-discovers when on the same Wi-Fi.
- **Probe scripts + interactive playground** — `probe_*.py` family covers model enumeration, capability bisection, end-to-end config verification, tool-call HIT/MISS measurement, and system-prompt honoring canaries. `python -m gemini_live_playground` is a REPL against the same model + prompt + tools + compression + voice as production.

---

## The three flows

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
                  (with per-step completion_criteria)
                                    │
                                    ▼
                  ffmpeg per-step clip extraction
                                    │
                                    ▼
                  SQLite (procedures + steps) + clips on disk
```

Upload returns `202` immediately. Poll `/api/procedures/{id}` until `status` is `completed`, `completed_partial` (some clips failed), or `failed`. Per-step ffmpeg failures don't fail the whole procedure — the step is marked `clip_status="failed"` and the procedure is `completed_partial`.

### Learner flow

```
iOS app --[POST procedure_id, auto_advance, voice]--> /api/learner/session/start
       <--[session + current-step prompt + ephemeral token]----
                                    │
                                    ▼
iOS app <=== direct Gemini Live WebSocket (BidiGenerateContentConstrained) ===> Gemini
                                    │
                                    │ (function calls: check_step† / get_reference_clip / go_to_step)
                                    ▼
iOS app --[POST tool_name, arguments]--> /api/learner/session/{id}/tool-call
       <--[{result: {...}}]-------------
                                    │ (phone unwraps result, sends back to Gemini as toolResponse)
                                    ▼
                       Gemini continues speaking / listening

   (~every 4 s, in parallel — Active mode only)
iOS app --[POST camera JPEG]-----------> /api/learner/session/{id}/verify
       <--[skip | pending | error | advanced | completed]---
                                    │ (Flash oracle vs completion_criteria + error_criteria;
                                    │  server advances itself on a confirmed completion)
                                    ▼
       phone injects an [ADVANCE] / [CORRECTION] turn into the Live socket
```

† `check_step` is **client-intercepted**: a camera frame can't ride the tool-call body, so when the learner says "done" (`claim="done"`) or asks "is this right?" (`claim="question"`), the phone POSTs its cached frame to `/verify mode=on_demand` and synthesizes the `toolResponse` itself.

The oracle is the **sole advancement authority** — Live never decides completion from the video, and `advance_step` is no longer in its toolset (the server handler survives for the HUD swipe). Visual auto-advance + error interjection run through the ~4 s `/verify` poll above; the iOS app collapses this into one **Active/Passive** toggle (Active polls; Passive is on-demand-only) and always sends `active_monitoring` and `auto_advance` equal. Learner authority is preserved: a second consecutive rejected done-claim advances verbally (insist-once), and an oracle outage on a done-claim advances verbally too (trust-on-failure).

On `goAway` or socket error the iOS client calls `POST /session/{id}/token?handle=<sessionResumptionUpdate.new_handle>` to mint a token that reopens the prior session with compressed context intact, then injects `GET /session/{id}/context-summary` as a user turn so the model re-orients on the real current step.

### Troubleshoot flow

```
iOS app --[POST voice?]--> /api/troubleshoot/session/start
       <--[session + troubleshoot prompt + ephemeral token]----
                                    │
                                    ▼
iOS app <=== direct Gemini Live WebSocket ===> Gemini
                                    │
                                    │ tools: identify_product → confirm_identification
                                    │        → search_procedures (Gemini-ranked over the library)
                                    │          ↳ no_match → web_search_for_fix
                                    │            (Gemini Pro + GoogleSearch grounding → procedure JSON
                                    │             with cited URLs; no PDF runtime path)
                                    │        → handoff_to_learner
                                    ▼
              POST /api/troubleshoot/session/{id}/handoff
              opens a normal learner session on the chosen procedure
```

Mode-parallel to learner — own router, own session dict, own prompt builder, but reuses the same Gemini Live config + tool dispatcher. PDF ingest still exists, but only as an expert-side flow at `/api/expert/manuals/*`.

---

## Critical architecture decisions

The cross-cutting Retrace decisions live in the **[org profile README](https://github.com/xrcc-retrace)**. The ones specific to this server:

1. **Voice + video go direct** — phone → Gemini Live, never through this server. The 100–200 ms server hop kills conversational feel. The backend is strictly the control plane.
2. **Visual verification runs out-of-band in a Gemini Flash oracle** — not inside Gemini Live. Live stays conversation-only (it keeps a low-res feed for the learner's visual questions but never decides completion — inline detection hallucinated on empty scenes). The phone polls `POST /session/{id}/verify` with a camera frame every ~4 s; `services/step_oracle.py` (`gemini-2.5-flash`, `RETRACE_ORACLE_MODEL`) compares it against the step's `completion_criteria` — grounded image-to-image on the expert clip's last frame (`step_N_done.jpg`) — plus its `error_criteria`, biased against false positives. On a confirmed completion the server advances itself (`method="visual"`); on a visible mistake it returns `error_text`. The phone injects an `[ADVANCE]`/`[CORRECTION]` turn into Live. A per-step `visual_verifiability` gate (`reliable|marginal|none`) decides whether the oracle runs (dwell: reliable=1, marginal=2 consecutive frames; `none`=verbal-only). The endpoint enforces the toggles: advance only when `active_monitoring AND auto_advance AND verifiability!=none`; interjection stays alive whenever `active_monitoring`. The same oracle backs the learner-initiated `check_step` tool via `/verify mode=on_demand` (client-intercepted with the cached frame; bypasses toggles and dwell, keeps the confidence floor) — with insist-once and trust-on-failure semantics so a skeptical oracle can never trap the learner. The old inline `advance_step` visual path + `[check]` heartbeat are gone.
3. **Audio + video share one Live session.** Context compression + session resumption unlock indefinite duration. The "2-minute audio+video" Gemini limit is a context-fill limit, not a hard timer.
4. **Current-step-only system prompt.** Future-step details are delivered exclusively through injected `[ADVANCE]` turns — the `check_step` tool response deliberately carries zero next-step content, so the model can't narrate ahead. On token refresh / resumption, the prompt is re-anchored on `session["current_step"]` so a reopened socket doesn't act like step 1.
5. **Tool calling is the model↔server channel.** Gemini issues function calls; the phone injects `session_id` and forwards them to `/tool-call` — except `check_step`, which the phone handles on-device (a camera frame can't ride the tool-call body). Server handler result is wrapped as `{result: {...}}`; the phone unwraps before handing back to Gemini.
6. **Ephemeral tokens** solve API key distribution. The phone never sees the raw key. Tokens are locked to a specific model / prompt / tool set / voice / transcription / compression / resumption handle / media resolution — a compromised token can't be repurposed.
7. **Bonjour over hardcoded IPs.** Wi-Fi IPs change across hackathon networks; manual override exists in the iOS Settings as a fallback.
8. **Per-step clip failures are tolerated.** Gemini occasionally produces step timestamp ranges ffmpeg can't cut. The procedure is saved as `completed_partial` with failed steps marked `clip_status="failed"` rather than erroring the whole upload.
9. **Manual procedures share the clip-serving path.** PDF ingest writes a rendered PNG to `step.clip_url` so the iOS client renders manual-sourced steps with the same code path as video-sourced ones — no client-side branching.
10. **Mode-based modularity.** New modes get their own router + session dict + prompt builder + handlers (see `services/troubleshoot/`). We don't extend `prompt_builder.py` or `session.py` inline with per-mode variants.

---

## Quick start

### Prerequisites

- **Python 3.11+**
- **`ffmpeg`** (`brew install ffmpeg`) — for clip extraction + duration probing
- A **[Gemini API key](https://aistudio.google.com/apikey)**

### Local development

```bash
git clone https://github.com/xrcc-retrace/main-local-server-test.git
cd main-local-server-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put GEMINI_API_KEY in .env, or pass explicitly:
python main.py                              # reads .env
GEMINI_API_KEY=<key> python main.py         # explicit
```

Server listens on `http://0.0.0.0:8000`.

- SOP review dashboard → `http://localhost:8000/`
- From an iOS device on the same Wi-Fi → Bonjour auto-discovers the server.

### Environment variables

| Name | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key. |
| `RETRACE_ADVERTISE_BONJOUR` | `1` | Set to `0` on cloud deploys (VPC doesn't support multicast; suppresses noisy startup). |
| `RETRACE_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | Override the Live model for a fast rollback. |
| `RETRACE_ORACLE_MODEL` | `gemini-2.5-flash` | Override the out-of-band visual-verification oracle model. |

---

## API reference

### Health & static

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/` | SOP review dashboard. |
| `GET` | `/api/uploads/{filename}` | Serve the original source video for review playback. |
| `GET` | `/static/voice_previews/{name}.wav` | Pre-generated voice preview. |

### Expert — `/api/expert`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/expert/upload` | Multipart video upload. Returns **202** immediately with `{id, status}`; Gemini analysis + clip extraction run in a background `asyncio` task. Poll `/api/procedures/{id}` for `status: processing / completed / completed_partial / failed`. |
| `POST` | `/api/expert/manuals/upload` | Multipart PDF upload. Returns **202** with `{manual_id}`; background task runs `services.manual_ingest.process_pdf(mode="expert")` → new procedure with `source_type="manual"`. Per-step manual-page PNG is written to `step.clip_url`. |
| `GET`  | `/api/expert/manuals/{manual_id}` | Poll ingest status. When ready, returns the resulting `procedure_id`. |
| `GET`  | `/api/manual_images/{procedure_id}/step_{n}.png` | Rendered manual-page PNG used in place of an MP4 clip for manual-sourced procedures. |

### Icons — `/api/icons/{prefix}/{name}.png`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/icons/{prefix}/{name}.png` | Iconify icon (e.g. `/api/icons/lucide/refrigerator.png`) rendered to a white-on-transparent PNG via `cairosvg`, disk-cached. |

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all (newest first) with `status`, `error_message`, `step_count`. |
| `GET` | `/api/procedures/{id}` | Get one with all steps (includes `completion_criteria`, `clip_status`). |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, or reorder steps via `step_order: [old_step_number, …]`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields. |
| `DELETE` | `/api/procedures/{id}` | Delete procedure + cascade steps. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |
| `GET` | `/api/clips/{id}/step_{n}.jpg` | On-demand, disk-cached first-frame JPEG poster (lens thumbnail — the lens can't take raw video frames). |
| `GET` | `/api/clips/{id}/step_{n}_done.jpg` | On-demand, disk-cached **last-frame** "done state" JPEG — the verification oracle's image-to-image reference. |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices with `preview_url`. |
| `POST` | `/api/learner/session/start` | Body: `{procedure_id, auto_advance?, active_monitoring?, voice?}`. Returns session id, full procedure, the assembled **current-step-only** system prompt, tool definitions, Gemini config, and an ephemeral token. `active_monitoring` (default `true`) is the master switch — when `false` the phone doesn't poll `/verify` (fully reactive, verbal-only). `auto_advance` (default `true`) gates visual advance, effective only when `active_monitoring` is also `true`. The iOS app collapses both into one Active/Passive toggle and always sends them equal. |
| `POST` | `/api/learner/session/{id}/verify` | Multipart body with one JPEG part `frame`, plus optional `mode` (`poll` default / `on_demand`) and `claim` (`done`/`question`, required with `on_demand`). Poll mode returns `{status}`: `skip` (verbal-only step / not auto-advancing), `pending`, `error` (`+error_text`), `advanced` (session advanced server-side), or `completed`. On-demand mode backs the client-intercepted `check_step` tool — bypasses toggles/dwell, keeps the confidence floor, adds `not_confirmed`/`unverifiable`/`check_failed`; insist-once + trust-on-failure semantics live here. The phone injects the outcome into Live as `[ADVANCE]`/`[CORRECTION]`. |
| `POST` | `/api/learner/session/{id}/token[?handle=...]` | Mint a fresh ephemeral token. Pass `?handle=<sessionResumptionUpdate.new_handle>` to bind it to the prior session so the next WebSocket continues with compressed context preserved. Re-anchors the prompt on the session's current step. |
| `POST` | `/api/learner/session/{id}/tool-call` | Forward a Gemini function call (`get_reference_clip`, `go_to_step`; `advance_step` survives here only for the iOS HUD swipe — it's no longer in Live's toolset); server executes and returns `{result: {...}}`. |
| `GET` | `/api/learner/session/{id}/state` | Current step, status, history (each entry carries `method: "visual" | "verbal"`). |
| `GET` | `/api/learner/session/{id}/context-summary` | Compact text the phone injects as a user message after compression or reconnect to re-orient the model. |
| `DELETE` | `/api/learner/session/{id}` | End and discard the session. |

### Troubleshoot — `/api/troubleshoot`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/troubleshoot/session/start` | Body: `{voice?}`. No procedure required at start. |
| `POST` | `/api/troubleshoot/session/{id}/token[?handle=...]` | Resumption-aware token refresh. |
| `POST` | `/api/troubleshoot/session/{id}/tool-call` | Tools: `identify_product`, `confirm_identification`, `search_procedures`, `web_search_for_fix`, `handoff_to_learner`. |
| `GET`  | `/api/troubleshoot/session/{id}/state` | Current phase, identified product, candidate procedures, diagnostic log. |
| `POST` | `/api/troubleshoot/session/{id}/handoff` | Promote a chosen `procedure_id` into a fresh learner session; returns the full `SessionStartResponse`. |
| `DELETE` | `/api/troubleshoot/session/{id}` | End and discard. |
| `GET`  | `/api/troubleshoot/sessions` | List active troubleshoot sessions. |

---

## Project layout

```
main-local-server-test/
├── main.py                    # FastAPI app entry, CORS, Bonjour (lifespan), health, static dashboard
├── config.py                  # Paths, GEMINI_API_KEY, GEMINI_MODEL, FFMPEG_PATH (auto-creates data dirs on import)
├── requirements.txt
├── models/
│   ├── procedure.py           # GeminiProcedureOutput (structured-output schema) + Procedure / ProcedureStep (with completion_criteria, clip_status, source_type)
│   ├── session.py             # Session start/state, StepCompletion (method: visual|verbal), tool call req/res, ephemeral token response
│   └── troubleshoot.py        # Troubleshoot session start/state, handoff, manual upload/status, identified product, candidate procedure, diagnostic log entry
├── routers/
│   ├── expert.py              # POST /api/expert/upload — async pipeline, timestamp clamping, per-step bookkeeping
│   ├── expert_manuals.py      # POST /api/expert/manuals/upload + GET /api/expert/manuals/{id} + GET /api/manual_images/{procedure_id}/step_{n}.png
│   ├── icons.py               # GET /api/icons/{prefix}/{name}.png — Iconify icon serving (white-on-transparent PNG, cached)
│   ├── learner.py             # Voice list + session lifecycle (start, tool-call, state, context-summary, token refresh with ?handle=)
│   ├── procedures.py          # CRUD for procedures / steps / clips
│   └── troubleshoot.py        # Mode-parallel to learner; importing this router imports services.troubleshoot which registers handlers with the shared dispatcher
├── services/
│   ├── gemini_processor.py    # Uploads video to Gemini Files API, polls, generates structured procedure JSON via Gemini 2.5 Pro. Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step ffmpeg extraction (H.264+AAC, faststart, bf=0, bt709 colr tags for iOS VideoToolbox)
│   ├── video_probe.py         # ffprobe authoritative duration
│   ├── database.py            # SQLite WAL + ALTER-TABLE migrations. Tables: procedures, steps.
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher (shared)
│   ├── step_oracle.py         # Out-of-band visual verification. gemini-2.5-flash (RETRACE_ORACLE_MODEL) compares the learner's frame vs the step's step_N_done.jpg + completion_criteria + error_criteria → {complete, confidence, error_index}. Biased against false positives. Backs POST /session/{id}/verify.
│   ├── prompt_builder.py      # Current-step-only system prompt. Live is conversation-only: an "EVENT SIGNALS" block tells it to act on injected [ADVANCE]/[CORRECTION] turns (from the oracle), not to watch the camera. Live tools: check_step (client-intercepted), get_reference_clip, go_to_step. Tool definitions + Gemini config.
│   ├── metadata_picker.py     # Flash (thinking_budget=0) call that picks each procedure's icon_symbol (Iconify id) + category from a closed enum. Used by all generation flows.
│   ├── ephemeral_token.py     # Mints v1alpha auth_tokens with the full Retrace config locked in
│   ├── procedure_search.py    # Gemini Flash-ranked search over the library by product + symptom
│   ├── manual_ingest.py       # PDF → structured SOP via Gemini Files API. Shared by expert (mode="expert") + troubleshoot (mode="troubleshoot"). Renders source_page → PNG; writes path into step.clip_url.
│   ├── manual_prompts.py      # Mode-specific Gemini prompts for manual_ingest
│   ├── pdf_images.py          # PDF page → PNG rasterization
│   ├── icon_renderer.py       # Iconify SVG → white-on-transparent PNG via cairosvg, disk-cached
│   ├── voice_commands.py      # Loads voice_commands.yaml at import; prompt builders look up phrasing by command id
│   ├── fixtures.py            # Startup loader that ingests bundled manual fixtures on a fresh DB; non-blocking from main.py's lifespan
│   └── troubleshoot/
│       ├── session.py         # In-memory troubleshoot store, phase transitions, diagnostic log
│       ├── prompt.py          # Troubleshoot-specific system prompt
│       ├── handlers.py        # Tool handlers; registered with the shared dispatcher at import time
│       └── web_research.py    # Two-step Gemini Pro pipeline (GoogleSearch grounding + structured-output) → procedure JSON with cited URLs
├── static/
│   ├── index.html             # SOP review dashboard
│   └── voice_previews/        # Pre-generated WAV previews
├── scripts/
│   └── generate_voice_previews.py
├── gemini_live_playground/    # Interactive REPL for prompt/tool tuning against the production config
├── evals/                     # Oracle + extractor eval harness — self-supervised ground truth from clip first/last frames (see evals/README.md)
├── fixtures/                  # Bundled manuals loaded on first start when absent from the DB (nespresso_vertuo_next_descaling.pdf is wired in)
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4   # + on-demand step_{n}.jpg (first-frame poster) and step_{n}_done.jpg (last-frame "done state", oracle reference) cached beside each clip
│   ├── manual_images/{procedure_id}/step_{n}.png   # rendered manual-page PNGs for manual-sourced procedures
│   └── retrace.db             # SQLite (WAL mode)
├── uploads/                   # Source expert videos ({uuid}.{ext})
├── reprocess_clips.py         # Re-run clip extraction without hitting Gemini again (after tweaking ffmpeg flags)
├── reset_data.py              # Wipe user-generated data (DB rows, uploads, clips). Preserves code, .env, voice_previews.
├── backfill_icons.py          # One-shot: render and cache Iconify PNGs for procedures predating the icon system
├── validate_voice_commands.py # Lint voice_commands.yaml (schema, uniqueness, referenced-from-code checks)
├── voice_commands.yaml        # Single source of truth for prompt utterances; designer-editable
├── test_live_session.py       # Ad-hoc integration test that drives a Gemini Live session against a running server
└── probe_*.py                 # Model enumeration + capability bisection + production-config verification + tool-call HIT/MISS measurement + system-prompt honoring canary
```

---

## Gemini Live configuration

Baked into every ephemeral token (the phone can't change them):

- **Model:** `gemini-3.1-flash-live-preview` (released 2026-03-26, public preview) — verified 2026-04-17 via the production ephemeral-token + `BidiGenerateContentConstrained.v1alpha` path. Supports video input, tools, context compression, session resumption, in/out transcription. Override via `RETRACE_LIVE_MODEL`. `turn_coverage` defaults to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO` on 3.1, so silent-period video frames now reach the model.
- **Voice:** server-authoritative list (Puck / Charon / Kore / Fenrir / Aoede / Zephyr / Enceladus / Leda — all 8 verified on 3.1). Default `Puck`.
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. Per-frame token burn scales with media resolution, so the compression cadence moves with it. Compression is transparent to the session — step state re-anchors through the system prompt and injected turns.
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. iOS persists the latest; on reconnect, `POST /session/{id}/token?handle=...` bakes it into `SessionResumptionConfig`.
- **Transcription:** both input (learner) and output (AI reply) audio are transcribed so the iOS activity feed renders text.
- **Media resolution:** `MEDIA_RESOLUTION_MEDIUM` (`services/ephemeral_token.py`) — balances visual detail for the learner's ad-hoc questions against token burn (step completion is decided by the out-of-band oracle on a full-resolution frame, not by Live). Raising/lowering it directly moves the compression cadence.
- **Token uses:** `uses=1`. iOS mints a fresh token per connection attempt (initial + retry + resumption).
- **Expiry format:** ISO8601 with **millisecond** precision. iOS's `ISO8601DateFormatter.withFractionalSeconds` caps at 3 fractional digits — Python's default 6-digit microsecond output silently fails to parse.

---

## Visual verification reliability

Design steps around what VLMs can actually detect. The `completion_criteria` string on each step is what the **verification oracle** (`services/step_oracle.py`) matches the learner's frame against — image-to-image, grounded on the expert clip's last frame (`step_N_done.jpg`). Write **observable single-frame end-states** ("the tray is fully seated in the slot"), not actions.

| Reliability | What works |
|---|---|
| **Reliable** | Object presence/absence, large state changes (lid open/closed, tray in/out) |
| **Marginal** | Small object state, correct orientation |
| **Unreliable** | Fine alignment, sub-millimeter changes, torque/fit |

At extraction, Gemini 2.5 Pro sets each step's `visual_verifiability` (`reliable|marginal|none`) from the spatial delta between the prior step's end-state and this one's. `none` marks steps with no single-frame visual consequence (counting, timing, torque) — the oracle never fires for them and they stay verbal-only.

**Proactive error interjection (Interject).** Each step also carries `error_criteria` — observable *wrong-state* descriptions (empty is fine). The oracle tests the current frame against them on every `/verify` poll and, on a clear match, returns `error_text`; the phone injects a `[CORRECTION]` turn into Live. Interjection stays alive even on verbal-only steps. Same detectability rules — phrase each error as a clearly visible state ("the cap is still on", "the tray is upside down"). `error_criteria` is produced by Pro at extraction; `scripts/backfill_error_criteria.py` seeds legacy rows.

**How we validated the oracle.** The [`evals/`](evals/README.md) harness scores both the oracle and the extractor against **self-supervised ground truth**: the expert clips label themselves — a step clip's last frame is a true "done" state for that step, and the next step's early frames are true "not yet" states. That gave us labeled eval sets at zero annotation cost, so prompt changes were accepted on measured false-positive/recall movement rather than demo vibes. One tightening pass drove completion false-positives from 31% to 0% (at a measured recall cost), and the evals showed the `visual_verifiability` gate is load-bearing — most false positives came from steps that should never have been visually verified in the first place.

Keep the extraction prompt in `gemini_processor.py` phrasing both criteria as observable end-states if you retune it.

---

## Production deployment

Retrace runs in production on **AWS Lightsail** at **https://retracexrcc.duckdns.org**. The iOS client falls back to this URL when Bonjour fails or when the user picks "Cloud" in Server Settings.

### Stack

| Layer | What |
|---|---|
| Host | Lightsail $12/mo — Ubuntu 22.04, 2 GB RAM, 2 vCPU, 60 GB SSD + 2 GB swap |
| Region | `eu-central-1` (Frankfurt) — migrated from Singapore ahead of the Berlin demo for ~10 ms venue latency |
| Public IP | Static IP `<server-ip>` (Lightsail) |
| DNS | DuckDNS: `retracexrcc.duckdns.org` → static IP |
| TLS + reverse proxy | [Caddy](https://caddyserver.com) with auto-provisioned Let's Encrypt certs |
| App | `uvicorn main:app --host 127.0.0.1 --port 8000` under systemd unit `retrace.service` |
| Secrets | `~/main-local-server-test/.env` (`chmod 600`) — `GEMINI_API_KEY` + `RETRACE_ADVERTISE_BONJOUR=0` |

Caddy listens on `:80` and `:443`, proxying to the local uvicorn on `:8000`. `uvicorn` binds `127.0.0.1` only — port 8000 is never exposed. The Lightsail firewall opens only `22 / 80 / 443`.

### CI/CD — auto-deploy on push to `main`

GitHub Actions workflow at `.github/workflows/deploy.yml`:

1. Triggered by `push: branches: [main]` or manual `workflow_dispatch`.
2. SSH into Lightsail as `ubuntu` using the `DEPLOY_SSH_KEY` repo secret.
3. `git pull --ff-only` in `~/main-local-server-test`.
4. `pip install -r requirements.txt` (idempotent — only touches disk if deps changed).
5. `sudo systemctl restart retrace` (passwordless sudo scoped to just this unit via `/etc/sudoers.d/retrace-deploy`).
6. Fail the run if `GET /health` doesn't return 200.

Required GitHub repo secrets:

| Secret | Value |
|---|---|
| `DEPLOY_HOST` | The Lightsail static IP |
| `DEPLOY_USER` | `ubuntu` |
| `DEPLOY_SSH_KEY` | OpenSSH private key whose public half is in `~/.ssh/authorized_keys` on the box |

### Manual operations

```bash
# SSH in
ssh ubuntu@<server-ip>

# Service status / logs
sudo systemctl status retrace --no-pager
journalctl -u retrace -n 100 --no-pager
sudo systemctl status caddy --no-pager
journalctl -u caddy -f

# Force a redeploy without pushing new code
gh workflow run "Deploy to Lightsail" --repo xrcc-retrace/main-local-server-test

# Restart only (no git pull)
sudo systemctl restart retrace
```

### Region migration (done before the Berlin demo)

Singapore → Frankfurt via Lightsail snapshot copy: copy the snapshot to `eu-central-1`, launch a new instance from it, attach a fresh static IP, and point the DuckDNS A record at it. No code changes — `cloudBaseURL` in iOS stays `https://retracexrcc.duckdns.org`; only the DNS target moved.

---

## Data storage

- `uploads/` — source expert videos, named `{uuid}.{ext}` (keep the original extension so ffprobe reads the container correctly).
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips.
- `data/manual_images/{procedure_id}/step_{n}.png` — rendered manual-page PNGs for manual-sourced procedures (written into `step.clip_url`).
- `data/retrace.db` — SQLite (WAL). Two tables:
  - `procedures(id, title, description, total_duration, source_video, created_at, status, error_message, source_type, icon_symbol, icon_emoji, category, sources)` — `source_type ∈ {video, manual, web}`; the last five come from lightweight ALTER-TABLE migrations.
  - `steps(id, procedure_id, step_number, title, description, timestamp_start, timestamp_end, tips, warnings, completion_criteria, error_criteria, visual_verifiability, clip_url, clip_status)` — `error_criteria` is a JSON array of observable error states; `visual_verifiability` is `reliable|marginal|none` (the oracle gate; legacy rows default `reliable`).
  - `tips` and `warnings` are JSON-encoded arrays.
  - `UNIQUE(procedure_id, step_number)` — reorder logic uses a two-pass negative-then-positive trick to avoid unique-constraint violations.
- Sessions are **in-memory only** (a `dict` in `services/session.py`). They die with the process.

---

## Testing

No test framework — manual via curl and the dashboard at `/`.

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

Sample videos live under `test_data/`. `test_live_session.py` drives a full Live session against a running server. The `probe_*.py` scripts are the source of truth for "which models / features work right now on this API key" — keep them around after a model upgrade rather than rewriting from scratch. `gemini_live_playground/` is the fastest way to iterate on prompt + tool behavior without rebuilding the iOS app:

```bash
python -m gemini_live_playground --procedure-id <uuid>
python -m gemini_live_playground --scenario scenarios/auto_advance.yaml
```

Slash commands inside the REPL: `/state`, `/jump <n>`, `/summary`, `/save`, `/quit`.

---

<div align="center">

Built by the XRCC Retrace team — 🥈 2nd place at the Berlin 2026 hackathon.

[**Org profile**](https://github.com/xrcc-retrace) &nbsp;·&nbsp; [**iOS app**](https://github.com/xrcc-retrace/retrace-ios) &nbsp;·&nbsp; [**Live API**](https://retracexrcc.duckdns.org/health)

</div>
