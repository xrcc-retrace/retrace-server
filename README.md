# Retrace — Backend Server

FastAPI backend for Retrace, the AI coaching system built for the XRCC Berlin 2026 hackathon.

> **One-liner:** Record an expert once. Coach every learner forever.

This server is the intelligence layer of a three-tier architecture (**glasses / iPhone → phone → server**). The phone opens the Gemini Live WebSocket directly — voice and video never hop through this server — so the backend is strictly the control plane:

- **Expert pipeline** — accepts a video upload, runs it through Gemini 2.5 Pro to extract a structured procedure (with per-step visual completion criteria), and uses ffmpeg to slice a playable clip for each step.
- **Learner pipeline** — starts a coaching session, assembles a current-step-only system prompt, and mints an ephemeral Gemini Live token so the phone can open a direct WebSocket to Gemini Live.
- **Tool-call dispatch** — during a session, Gemini issues `advance_step` / `get_reference_clip` / `go_to_step` function calls; the phone forwards them here and the server returns the result.
- **Session resumption + context recovery** — when Gemini signals `goAway` or the socket drops, the server mints a resumption-bound token and serves a compact context summary so the phone can re-orient the model mid-procedure.
- **Discovery** — advertises itself on the LAN via Bonjour (`_retrace._tcp.local.`) so the iOS app can auto-discover it.

## Prerequisites

- Python 3.11+
- `ffmpeg` (`brew install ffmpeg`) — clip extraction + duration probing
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put your key in a .env file, or export it inline:
GEMINI_API_KEY=your_key_here python main.py
```

Server listens on `http://0.0.0.0:8000`.

- SOP review dashboard: `http://localhost:8000/`
- From an iOS device on the same Wi-Fi: Bonjour discovery finds the server automatically.

## Architecture Overview

### Expert flow

```
iOS app --[POST multipart video]--> /api/expert/upload
       <--[202 + procedure_id]-----
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

Upload returns `202` immediately. Poll `/api/procedures/{id}` until `status` is `completed`, `completed_partial` (some clips failed), or `failed`. Per-step ffmpeg failures don't fail the whole procedure — the step is marked `clip_status="failed"` and the procedure as a whole lands `completed_partial`.

### Learner flow

```
iOS app --[POST procedure_id, auto_advance, voice]--> /api/learner/session/start
       <--[session + current-step prompt + ephemeral token]----
                                   │
                                   ▼
iOS app <=== direct Gemini Live WebSocket (BidiGenerateContentConstrained) ===> Gemini
                                   │
                                   │ (function calls: advance_step / get_reference_clip / go_to_step)
                                   ▼
iOS app --[POST tool_name, arguments]--> /api/learner/session/{id}/tool-call
       <--[tool result]-----------------
                                   │ (phone sends result back to Gemini as toolResponse)
                                   ▼
Gemini continues speaking / listening / watching
```

On `goAway` or socket error the iOS client calls `POST /session/{id}/token?handle=<sessionResumptionUpdate.new_handle>` to mint a token that reopens the prior session with compressed context intact, then injects `GET /session/{id}/context-summary` as a user turn to re-orient the model on the real current step.

### Why voice + video go direct

A server hop would add 100–200 ms per exchange and break conversational feel. Instead, the server mints a short-lived Gemini auth token (via the `v1alpha` `auth_tokens` API), locked to a specific model / system prompt / tool set / voice / transcription / compression / resumption handle, and the phone uses it to open the Live WebSocket directly. The Gemini API key never leaves this server.

## Gemini Live configuration

Baked into every ephemeral token:

- **Model:** `gemini-2.5-flash-native-audio-preview-12-2025` — empirically verified (`probe_live_models.py`, 2026-04-17) as the only Live model accepted by `bidiGenerateContent` on the current API key. Native-audio variant (not half-cascade), but supports video input, tools, context compression, session resumption, proactive audio, and in/out transcription.
- **Voice:** server-authoritative list (Puck / Charon / Kore / Fenrir / Aoede / Zephyr / Enceladus / Leda). iOS reads this from `GET /api/learner/voices`. Default `Puck`.
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. At ~0.5 fps video + `MEDIA_RESOLUTION_LOW`, first compression fires around 25 min, then every ~12 min.
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. Passing that handle on token refresh binds the new token to the prior session.
- **Proactivity:** `proactive_audio=True` — the model may stay silent if the learner isn't addressing it. Requires `v1alpha`.
- **Transcription:** both input (learner) and output (AI reply) audio are transcribed so the iOS activity feed renders text.
- **Media resolution:** `MEDIA_RESOLUTION_LOW`.
- **Token uses:** `uses=1`. iOS mints a fresh token per connection attempt.
- **Expiry format:** ISO8601 with millisecond precision — iOS's `ISO8601DateFormatter.withFractionalSeconds` caps at 3 fractional digits, so 6-digit microsecond output silently fails to parse.

## Project Layout

```
main-local-server-test/
├── main.py                    # FastAPI app, CORS, Bonjour (lifespan), health, static dashboard
├── config.py                  # Paths, GEMINI_API_KEY, GEMINI_MODEL, FFMPEG_PATH (auto-creates data dirs on import)
├── requirements.txt
├── models/
│   ├── procedure.py           # GeminiProcedureOutput (structured-output schema) + Procedure / ProcedureStep (with completion_criteria, clip_status)
│   └── session.py             # Session start/state, StepCompletion (method: visual|verbal), tool call req/res, ephemeral token response
├── routers/
│   ├── expert.py              # POST /api/expert/upload — async pipeline + timestamp clamping
│   ├── learner.py             # Voice list + session lifecycle (start, tool-call, state, context-summary, token refresh with ?handle=)
│   └── procedures.py          # CRUD for procedures / steps / clips
├── services/
│   ├── gemini_processor.py    # Upload video to Gemini Files API, poll, generate structured JSON (Pro). Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step ffmpeg extraction (H.264+AAC, faststart, bf=0, bt709 color tags for iOS VideoToolbox)
│   ├── video_probe.py         # ffprobe authoritative duration
│   ├── database.py            # SQLite WAL + ALTER-TABLE migrations. Tables: procedures, steps.
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher
│   ├── prompt_builder.py      # Current-step-only system prompt (auto-advance = visual; manual = verbal). Tool definitions + Gemini config.
│   └── ephemeral_token.py     # Mints v1alpha auth_tokens with the full Retrace config locked in
├── static/
│   ├── index.html             # SOP review dashboard
│   └── voice_previews/        # Pre-generated WAV previews
├── scripts/
│   └── generate_voice_previews.py
├── reprocess_clips.py         # Re-run clip extraction without hitting Gemini again (after tweaking ffmpeg flags)
├── test_live_session.py       # Ad-hoc integration test against a running server
├── probe_live_models.py       # Which Live model IDs this API key accepts + per-model proactive_audio / video input probes
├── probe_retrace_config.py    # End-to-end verification of the production ephemeral-token config
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4
│   └── retrace.db             # SQLite WAL
└── uploads/                   # Source expert videos ({uuid}.{ext})
```

## API Reference

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

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all procedures (newest first) with `status`, `error_message`, `step_count`. |
| `GET` | `/api/procedures/{id}` | Get one procedure with all steps (includes `completion_criteria`, `clip_status`). |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, or reorder steps via `step_order: [old_step_number, …]`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields (`title`, `description`, `tips`, `warnings`, `timestamp_start`, `timestamp_end`). |
| `DELETE` | `/api/procedures/{id}` | Delete procedure + cascade steps. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices with `preview_url`. |
| `POST` | `/api/learner/session/start` | Start a session. Body: `{procedure_id, auto_advance?, voice?}`. Returns the session id, full procedure, the assembled **current-step-only** system prompt, tool definitions, Gemini config, and an ephemeral token. |
| `POST` | `/api/learner/session/{id}/token[?handle=...]` | Mint a fresh ephemeral token. Pass `?handle=<sessionResumptionUpdate.new_handle>` to bind the new token to the prior session so the next WebSocket continues with compressed context preserved. The regenerated system prompt re-anchors on the session's current step. |
| `POST` | `/api/learner/session/{id}/tool-call` | Forward a Gemini function call (`advance_step`, `get_reference_clip`, `go_to_step`) from the phone; the server executes and returns the result wrapped as `{result: {...}}`. |
| `GET` | `/api/learner/session/{id}/state` | Current step, status, history (each entry carries `method: "visual" | "verbal"`). |
| `GET` | `/api/learner/session/{id}/context-summary` | Compact text blob the phone injects as a user message after context compression or reconnect to re-orient the model. |
| `DELETE` | `/api/learner/session/{id}` | End and discard the session (sessions are held in memory, not DB). |

## Data Storage

- `uploads/` — source expert videos (`{uuid}.{ext}`).
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips (H.264+AAC, faststart).
- `data/retrace.db` — SQLite (WAL mode). Tables: `procedures`, `steps`. `tips` and `warnings` are JSON-encoded arrays; steps carry `completion_criteria` + `clip_status`; procedures carry `status` + `error_message`.
- Sessions are **in-memory only** (a `dict` keyed by session id) — they die with the server. The iOS client fires a best-effort `DELETE` on session end.

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

Sample videos live under `test_data/`. `test_live_session.py` drives a full Live session against a running server. `probe_live_models.py` and `probe_retrace_config.py` are the source of truth for "which models / features work right now on this API key."

## Critical Architecture Decisions

1. **Voice + video go direct** (phone → Gemini Live). A server hop costs 100–200 ms per exchange and kills conversational feel. The server is strictly the control plane.
2. **Visual verification runs inline in Gemini Live** — not as a separate server-side pass. The model receives the camera feed alongside audio, watches for each step's `completion_criteria`, and fires `advance_step` via function calling. `auto_advance=True` tags completions as `method="visual"`; manual mode is `"verbal"`. A Gemini 2.5 Pro REST clip-to-clip fallback is deferred unless inline detection proves unreliable in practice.
3. **Audio + video share one Live session.** Context window compression (100k → 40k) plus session resumption via `sessionResumptionUpdate.new_handle` unlock indefinite duration — the "2-minute audio+video" Gemini limit is a context-fill limit, not a hard timer.
4. **Current-step-only system prompt.** Only the current step is embedded; future-step details reach the model exclusively through `advance_step` tool responses. This prevents narrating ahead or skipping the tool call. On token refresh/resumption the prompt is re-anchored on the session's current step so a reopened socket doesn't act like step 1.
5. **Tool calling is the model↔server channel.** Gemini issues function calls; the phone intercepts, injects `session_id`, and forwards. The server returns `{result: {...}}` and the phone unwraps before replying to Gemini with a `toolResponse`.
6. **Ephemeral tokens** solve API key distribution. Tokens are locked to a specific model / prompt / tool set / voice / transcription / compression / resumption handle / media resolution. A compromised token can't be repurposed.
7. **Bonjour over hardcoded IPs.** Wi-Fi IPs change across hackathon networks; manual override exists in the iOS Settings as a fallback.
8. **Per-step clip failures are tolerated.** If ffmpeg can't cut a step's range, the procedure lands `completed_partial` with that step marked `clip_status="failed"` rather than erroring the whole upload.

## Visual Verification Reliability

Design steps around what VLMs can actually detect. The `completion_criteria` string on each step is what Gemini Live uses to decide when to call `advance_step` — write observable end-states, not actions.

- **Reliable:** Object presence/absence, large state changes (lid open/closed, tray in/out).
- **Marginal:** Small object state, correct orientation.
- **Unreliable:** Fine alignment, sub-millimeter changes, torque/fit.

The extraction prompt in `gemini_processor.py` asks Gemini to phrase completion criteria as observable states; keep it that way if you retune the prompt.
