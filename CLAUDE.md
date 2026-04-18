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

The iOS companion app lives in a separate repo (`dat-ios-test`). This server also hosts a static SOP review dashboard at `/` for inspecting and editing procedures without the phone.

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

Two main flows:

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
```

- The assembled system prompt embeds **only the current step** — future-step details are revealed exclusively through `advance_step` tool responses. This prevents the model from narrating ahead or skipping the tool call.
- Tools: `advance_step` (no args), `get_reference_clip` (no args), `go_to_step(step_number)`.
- Sessions are **in-memory only** (a `dict` keyed by session id). They die with the process — no DB table.
- When the phone sees Gemini's `goAway` frame (~60 s warning) or a socket error, it calls `POST /session/{id}/token?handle=<sessionResumptionUpdate.new_handle>` to mint a token that reopens the prior session with compressed context intact. On reconnect it injects the result of `GET /session/{id}/context-summary` as a user turn to re-orient the model on the real current step.

### Bonjour discovery

On startup the server advertises `_retrace._tcp.local.` on port 8000 via zeroconf. The iOS app's `BonjourDiscovery` populates its cached `serverBaseURL` automatically, with a manual-override UI as the fallback. No hardcoded IPs needed when both devices are on the same Wi-Fi.

## Project Layout

```
main-local-server-test/
├── main.py                    # FastAPI app entry, CORS, Bonjour (lifespan), health, static dashboard
├── config.py                  # Paths, GEMINI_API_KEY, GEMINI_MODEL, FFMPEG_PATH (auto-creates data dirs on import)
├── requirements.txt
├── models/
│   ├── procedure.py           # GeminiProcedureOutput (structured-output schema) + Procedure / ProcedureStep (API models, with completion_criteria and clip_status)
│   └── session.py             # Pydantic models for session start/state, StepCompletion (method: "visual" | "verbal"), tool call req/res, ephemeral token response
├── routers/
│   ├── expert.py              # POST /api/expert/upload — async background pipeline, timestamp clamping, per-step status bookkeeping
│   ├── learner.py             # Voices list + session lifecycle (start, tool-call forwarding, state, context summary, token refresh with optional ?handle=)
│   └── procedures.py          # CRUD for procedures / steps + clip serving
├── services/
│   ├── gemini_processor.py    # Uploads video to Gemini Files API, polls, generates structured procedure JSON via Gemini 2.5 Pro. Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step MP4 extraction via ffmpeg subprocess (H.264+AAC, faststart, bt709 colr tags, bf=0 to keep DTS monotonic)
│   ├── video_probe.py         # ffprobe wrapper — authoritative duration read on upload
│   ├── database.py            # SQLite (WAL). Tables: procedures, steps. JSON-serialized tips/warnings. Lightweight ALTER-TABLE migrations for new columns.
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher
│   ├── prompt_builder.py      # Current-step-only system prompt (auto-advance = visual detection; manual = verbal confirmation). Tool definitions + Gemini config.
│   └── ephemeral_token.py     # Mints v1alpha auth_tokens locked to model + system prompt + tools + voice + transcription + compression + resumption handle
├── static/
│   ├── index.html             # SOP review dashboard
│   └── voice_previews/        # Pre-generated WAV previews of Gemini voices
├── scripts/
│   └── generate_voice_previews.py   # Regenerate the WAV previews
├── reprocess_clips.py         # Re-run clip extraction for completed procedures without hitting Gemini again (after tweaking ffmpeg flags)
├── test_live_session.py       # Ad-hoc integration test that drives a Gemini Live session against a running server
├── probe_live_models.py       # Which Live model IDs bidiGenerateContent accepts on this API key, plus per-model probes of proactive_audio and video input
├── probe_retrace_config.py    # End-to-end verification of the production ephemeral-token config against a real Live session
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4
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
| `POST` | `/api/expert/upload` | Multipart video upload. Returns `202 {id, status: "processing"}`; background task runs Gemini analysis + clip extraction. Poll `/api/procedures/{id}` for final status. |

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all (newest first). Includes `status` (`processing` / `completed` / `completed_partial` / `failed`) and `step_count`. |
| `GET` | `/api/procedures/{id}` | One procedure with all steps (includes `completion_criteria`, `clip_status`). |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, and/or reorder steps via `step_order: [old_step_number, ...]`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields (`title`, `description`, `tips`, `warnings`, `timestamp_start`, `timestamp_end`). |
| `DELETE` | `/api/procedures/{id}` | Delete procedure + cascade steps. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices (`{name, description, preview_url}`). iOS reads this instead of hardcoding. |
| `POST` | `/api/learner/session/start` | Body: `{procedure_id, auto_advance?, voice?}`. Returns session id, full procedure, assembled (current-step-only) system prompt, tool definitions, Gemini config, ephemeral token. |
| `POST` | `/api/learner/session/{id}/token[?handle=...]` | Mint a fresh ephemeral token. Pass the latest `sessionResumptionUpdate.new_handle` as `?handle=` to bind the new token to the prior session — the next WebSocket continues with compressed context intact. The regenerated system prompt re-anchors on the session's current step, so a resumed socket doesn't reopen on step 1. |
| `POST` | `/api/learner/session/{id}/tool-call` | Body: `{tool_name, arguments}`. Executes the handler and returns `{result: {...}}`. The phone unwraps `result` and hands it back to Gemini via `toolResponse`. |
| `GET` | `/api/learner/session/{id}/state` | Current step, status, step history (each entry includes `method: "visual" | "verbal"`). |
| `GET` | `/api/learner/session/{id}/context-summary` | Compact text blob the phone injects as a user message after context compression or reconnect to re-orient the model. |
| `DELETE` | `/api/learner/session/{id}` | End and discard the session. |

## Gemini Live config

These settings are locked into every ephemeral token so the phone can't change them:

- **Model:** `gemini-3.1-flash-live-preview` (released 2026-03-26, public preview) — verified 2026-04-17 via the production ephemeral-token + `BidiGenerateContentConstrained.v1alpha` path. Supports video input, tools, context compression, session resumption, in/out transcription. Override via `RETRACE_LIVE_MODEL` env var for fast rollback to 2.5 native-audio. `turn_coverage` default flips to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO` on 3.1, so silent-period video frames now reach the model.
- **Voice:** server-authoritative list (Puck / Charon / Kore / Fenrir / Aoede / Zephyr / Enceladus / Leda). Default `Puck`.
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. At ~0.5 fps video + `MEDIA_RESOLUTION_LOW`, first compression cycle fires around 25 min, then every ~12 min.
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. iOS persists the latest; on reconnect, `POST /session/{id}/token?handle=...` bakes it into `SessionResumptionConfig` so the next WebSocket continues where the prior one left off.
- **Transcription:** both input (learner) and output (AI reply) audio are transcribed so the iOS activity feed renders text.
- **Media resolution:** `MEDIA_RESOLUTION_LOW` keeps video token burn at ~33 tok/s.
- **Token uses:** `uses=1`. Matches documented single-use semantics. The iOS client mints a fresh token for every connection attempt (initial + retry + resumption), so this is sufficient.
- **Expiry format:** ISO8601 with **millisecond** precision (`timespec="milliseconds"`). iOS's `ISO8601DateFormatter.withFractionalSeconds` caps at 3 fractional digits — Python's default 6-digit microsecond output silently fails to parse and forces spurious refreshes.

## Data Storage

- `uploads/` — source expert videos, named `{uuid}.{ext}`. Keep the original extension so ffprobe reads the container correctly.
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips.
- `data/retrace.db` — SQLite (WAL). Two tables:
  - `procedures(id, title, description, total_duration, source_video, created_at, status, error_message)`
  - `steps(id, procedure_id, step_number, title, description, timestamp_start, timestamp_end, tips, warnings, completion_criteria, clip_url, clip_status)`
  - `tips` and `warnings` are JSON-encoded arrays.
  - `UNIQUE(procedure_id, step_number)` — reorder logic uses a two-pass negative-then-positive trick to avoid unique-constraint violations.
- Sessions are **in-memory only** (a `dict` in `services/session.py`). They die with the process.

## Testing

No test framework — manual testing via curl and the dashboard at `/`.

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

Sample videos live under `test_data/`. `test_live_session.py` drives a full Live session against a running server. The two probe scripts (`probe_live_models.py`, `probe_retrace_config.py`) are the source of truth for "which models/features work right now on this API key."

## Critical Architecture Decisions

1. **Voice + video go direct**, phone → Gemini Live. Routing them through this server adds 100-200 ms per exchange and breaks conversational feel. The server is strictly the control plane.
2. **Visual verification runs inline in Gemini Live**, not as a separate server-side pass. The model sees the camera feed alongside audio, watches for each step's `completion_criteria`, and fires `advance_step` via function calling. `auto_advance=True` tags completions as `method="visual"`; manual mode is `"verbal"`. A Gemini 2.5 Pro REST clip-to-clip fallback is deferred unless inline detection proves unreliable.
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

The `completion_criteria` string on each step is what Gemini Live uses to decide when to call `advance_step`. Write criteria that describe an **observable end-state**, not an action. The extraction prompt in `gemini_processor.py` explicitly asks Gemini to phrase completion criteria that way.

## Gotchas

- **`config.py` creates data dirs on import.** Anything that imports `config` (main, routers, services) will touch disk. This is intentional for first-run setup — don't work around it by deferring imports.
- **Gemini Files API uploads must be deleted** or they count against your storage quota. `gemini_processor.process_expert_video` does this in a `finally` block.
- **ffmpeg flags matter for iOS HD playback.** `-bf 0` keeps PTS == DTS (iOS rejects negative DTS); `-x264opts colorprim=bt709:transfer=bt709:colormatrix=bt709` writes the SEI tags VideoToolbox needs or it complains with `verify_image_parameters` errors. Don't remove these.
- **Don't commit `.env`.** Put real API keys there and let `python-dotenv` load them.
- **Session memory leak hazard.** The iOS client fires a best-effort `DELETE /session/{id}` on session end. If that request is lost, the in-memory dict grows until the server restarts. It's a hackathon tradeoff — don't "fix" by persisting sessions unless the Live-session lifecycle also moves to DB.
- **Prompt changes require re-minting tokens.** System prompts are baked into the ephemeral token. Updating `prompt_builder.py` only affects sessions started after the change; in-flight tokens still carry the old prompt.
