# Retrace — Backend Server

FastAPI backend for Retrace, the AI coaching system built for the XRCC Berlin 2026 hackathon.

> **One-liner:** Record an expert once. Coach every learner forever.

This server is the intelligence layer of a three-tier architecture (**glasses / iPhone → phone → server**). The phone opens the Gemini Live WebSocket directly — voice and video never hop through this server — so the backend is strictly the control plane:

- **Expert pipeline** — accepts a video upload, runs it through Gemini 2.5 Pro to extract a structured procedure (with per-step visual completion criteria), and uses ffmpeg to slice a playable clip for each step. Also accepts PDF manuals (`/api/expert/manuals/upload`) and renders each step's source page as a PNG so manual-sourced procedures plug into the same learner flow as video-sourced ones.
- **Learner pipeline** — starts a coaching session, assembles a current-step-only system prompt, and mints an ephemeral Gemini Live token so the phone can open a direct WebSocket to Gemini Live.
- **Troubleshoot pipeline** — mode-parallel to learner. No procedure required at start; Gemini Live diagnoses a broken product via `identify_product` / `confirm_identification` / `search_procedures` / `web_search_for_fix` (Gemini Pro + GoogleSearch grounding → structured procedure with cited sources) / `handoff_to_learner` (promotes the chosen procedure into a normal coaching session).
- **Tool-call dispatch** — during a session, Gemini issues function calls (`advance_step` / `get_reference_clip` / `go_to_step` for learner; the troubleshoot tools above); the phone forwards them here and the server returns the result via a shared dispatcher.
- **Session resumption + context recovery** — when Gemini signals `goAway` or the socket drops, the server mints a resumption-bound token and serves a compact context summary so the phone can re-orient the model mid-procedure.
- **Iconify icon serving** — `GET /{prefix}/{name}.png` returns a white-on-transparent PNG (cairosvg-rendered, disk-cached) so the iOS HUD can reference any Iconify icon ID without bundling assets.
- **Discovery** — advertises itself on the LAN via Bonjour (`_retrace._tcp.local.`) so the iOS app can auto-discover it.

## Prerequisites

- Python 3.11+
- `ffmpeg` (`brew install ffmpeg`) — clip extraction + duration probing
- [Gemini API key](https://aistudio.google.com/apikey)

## Local Development

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

### Environment variables

| Name | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. Google Gemini API key. |
| `RETRACE_ADVERTISE_BONJOUR` | `1` | Set to `0` on cloud deploys (VPC doesn't support multicast; suppresses noisy startup). |
| `RETRACE_LIVE_MODEL` | `gemini-3.1-flash-live-preview` | Override the Live model for a fast rollback. |

## Production Deployment

Retrace runs in production on AWS Lightsail at **https://retracexrcc.duckdns.org**. The iOS client falls back to this URL when Bonjour fails or when the user picks "Cloud" in Server Settings.

### Stack

| Layer | What |
|---|---|
| Host | Lightsail $12/mo — Ubuntu 22.04, 2 GB RAM, 2 vCPU, 60 GB SSD + 2 GB swap |
| Region | `ap-southeast-1` (Singapore) — migrating to `eu-central-1` before the Berlin demo |
| Public IP | Static IP `<server-ip>` (Lightsail) |
| DNS | DuckDNS: `retracexrcc.duckdns.org` → static IP (free, manually refreshed on IP change) |
| TLS + reverse proxy | [Caddy](https://caddyserver.com) with auto-provisioned Let's Encrypt certs |
| App | `uvicorn main:app --host 127.0.0.1 --port 8000` under systemd unit `retrace.service` |
| Secrets | `~/main-local-server-test/.env` (`chmod 600`) — `GEMINI_API_KEY` + `RETRACE_ADVERTISE_BONJOUR=0` |

Caddy listens on `:80` and `:443`, proxying to the local uvicorn on `:8000`. `uvicorn` binds `127.0.0.1` only — port 8000 is never exposed to the internet. The Lightsail firewall opens only `22 / 80 / 443`.

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
| `DEPLOY_SSH_KEY` | Full OpenSSH private key (including BEGIN/END banners) whose public half is in `~/.ssh/authorized_keys` on the box |

### Manual operations

```bash
# SSH in (browser console from Lightsail, or local .pem key)
ssh -i ~/.ssh/<deploy-key>.pem ubuntu@<server-ip>

# Service status / logs
sudo systemctl status retrace --no-pager
journalctl -u retrace -n 100 --no-pager
sudo systemctl status caddy --no-pager
journalctl -u caddy -n 100 --no-pager

# Force a redeploy without pushing new code
gh workflow run "Deploy to Lightsail" --repo xrcc-retrace/main-local-server-test

# Restart only (no git pull)
sudo systemctl restart retrace

# Tail the live Caddy request log
sudo journalctl -u caddy -f

# Check disk / memory / swap
df -h /
free -h
```

### Pausing billing

Lightsail bills the bundle whether the instance is running or stopped — true "pause" = **snapshot-and-delete**.

1. Lightsail console → instance → Snapshots → **Create snapshot** (~$0.05/GB/mo while paused; typical ~5 GB snapshot ≈ $0.25/mo).
2. Delete the instance.
3. Networking → Static IPs → **Detach and release** (attached IPs are free; detached ~$3.60/mo).

To resume: create a new instance from the snapshot, allocate a new static IP, update DuckDNS to the new IP. Caddy re-provisions TLS on the first HTTPS request. ~5 min hands-on.

### Region migration (planned before Berlin demo)

Current Singapore deploy → Frankfurt for ~10 ms latency vs the XRCC Berlin hackathon venue:

1. Lightsail console → Snapshots → **Copy snapshot to another region** → `eu-central-1`.
2. Launch a new instance from the copied snapshot.
3. Allocate a new static IP in Frankfurt, attach.
4. Update DuckDNS A record to the new IP.
5. Delete the Singapore instance and release its static IP.

No code changes. The `cloudBaseURL` in the iOS `ServerEndpoint.swift` stays `https://retracexrcc.duckdns.org`; only the DNS target moves.

## Architecture Overview

The server runs **three** flows: expert (video / PDF → SOP), learner (real-time coaching), and troubleshoot (diagnose-then-coach).

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
                                   │            (Gemini Pro + GoogleSearch grounding → structured
                                   │             procedure JSON with cited URLs; no PDF runtime path)
                                   │        → handoff_to_learner
                                   ▼
              POST /api/troubleshoot/session/{id}/handoff
              opens a normal learner session on the chosen procedure
```

`web_search_for_fix` replaced the old `fetch_manual` PDF flow inside the troubleshoot path. PDF ingest still exists as a separate **expert-side** flow (`/api/expert/manuals/*`), where the user explicitly uploads a manual to grow the library.

### Why voice + video go direct

A server hop would add 100–200 ms per exchange and break conversational feel. Instead, the server mints a short-lived Gemini auth token (via the `v1alpha` `auth_tokens` API), locked to a specific model / system prompt / tool set / voice / transcription / compression / resumption handle, and the phone uses it to open the Live WebSocket directly. The Gemini API key never leaves this server.

## Gemini Live configuration

Baked into every ephemeral token:

- **Model:** `gemini-3.1-flash-live-preview` (released 2026-03-26, public preview) — verified 2026-04-17 via the production ephemeral-token + `BidiGenerateContentConstrained.v1alpha` path. Supports video input, tools, context compression, session resumption, in/out transcription. Override via `RETRACE_LIVE_MODEL` env var for fast rollback. `turn_coverage` default flips to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO` on 3.1, so silent-period video frames now reach the model.
- **Voice:** server-authoritative list (Puck / Charon / Kore / Fenrir / Aoede / Zephyr / Enceladus / Leda — all 8 verified on 3.1). iOS reads this from `GET /api/learner/voices`. Default `Puck`.
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. At ~0.5 fps video + `MEDIA_RESOLUTION_LOW`, first compression fires around 25 min, then every ~12 min.
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. Passing that handle on token refresh binds the new token to the prior session.
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
│   ├── procedure.py           # GeminiProcedureOutput (structured-output schema) + Procedure / ProcedureStep (with completion_criteria, clip_status, source_type)
│   ├── session.py             # Session start/state, StepCompletion (method: visual|verbal), tool call req/res, ephemeral token response
│   └── troubleshoot.py        # Troubleshoot session start/state, handoff, manual upload/status, identified product, candidate procedure, diagnostic log entry
├── routers/
│   ├── expert.py              # POST /api/expert/upload — async pipeline + timestamp clamping
│   ├── expert_manuals.py      # POST /api/expert/manuals/upload + GET /api/expert/manuals/{id} + GET /api/manual_images/{procedure_id}/step_{n}.png
│   ├── icons.py               # GET /{prefix}/{name}.png — Iconify icon serving (white-on-transparent PNG, cached)
│   ├── learner.py             # Voice list + session lifecycle (start, tool-call, state, context-summary, token refresh with ?handle=)
│   ├── procedures.py          # CRUD for procedures / steps / clips
│   └── troubleshoot.py        # Mode-parallel to learner; importing this router imports services.troubleshoot which registers handlers with the shared dispatcher
├── services/
│   ├── gemini_processor.py    # Upload video to Gemini Files API, poll, generate structured JSON (Pro). Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step ffmpeg extraction (H.264+AAC, faststart, bf=0, bt709 color tags for iOS VideoToolbox)
│   ├── video_probe.py         # ffprobe authoritative duration
│   ├── database.py            # SQLite WAL + ALTER-TABLE migrations. Tables: procedures, steps.
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher (shared by learner + troubleshoot)
│   ├── prompt_builder.py      # Current-step-only system prompt (auto-advance = visual; manual = verbal). Tool definitions + Gemini config.
│   ├── ephemeral_token.py     # Mints v1alpha auth_tokens with the full Retrace config locked in
│   ├── procedure_search.py    # Gemini Flash-ranked search over the existing library by product + symptom
│   ├── manual_ingest.py       # PDF → structured SOP via Gemini Files API. Shared by expert (mode="expert") + troubleshoot (mode="troubleshoot"). Renders source_page → PNG; writes path to step.clip_url so manual procedures plug into existing clip endpoints unchanged.
│   ├── manual_prompts.py      # Mode-specific Gemini prompts for manual_ingest
│   ├── pdf_images.py          # PDF page → PNG rasterization
│   ├── icon_renderer.py       # Iconify SVG → white-on-transparent PNG via cairosvg, disk-cached by Iconify ID
│   ├── voice_commands.py      # Loads voice_commands.yaml at import; prompt builders look up phrasing by command id
│   ├── fixtures.py            # Startup loader that ingests bundled manual fixtures on a fresh DB; scheduled non-blocking from main.py's lifespan
│   └── troubleshoot/
│       ├── session.py         # In-memory troubleshoot store, phase transitions, diagnostic log
│       ├── prompt.py          # Troubleshoot-specific system prompt
│       ├── handlers.py        # Tool handlers: identify_product, confirm_identification, search_procedures, web_search_for_fix, handoff_to_learner
│       └── web_research.py    # Two-step Gemini Pro pipeline (GoogleSearch grounding + structured-output) → procedure JSON with cited URLs. Backs web_search_for_fix; replaced the old fetch_manual flow.
├── static/
│   ├── index.html             # SOP review dashboard
│   └── voice_previews/        # Pre-generated WAV previews
├── scripts/
│   └── generate_voice_previews.py
├── gemini_live_playground/    # Interactive REPL for prompt/tool tuning against the production config (no FastAPI server required)
├── reprocess_clips.py         # Re-run clip extraction without hitting Gemini again (after tweaking ffmpeg flags)
├── reset_data.py              # Wipe user-generated data (DB rows, uploads, clips). Preserves code, .env, voice_previews.
├── backfill_icons.py          # One-shot: render and cache Iconify PNGs for procedures predating the icon system
├── validate_voice_commands.py # Lint voice_commands.yaml (schema, uniqueness, referenced-from-code checks)
├── voice_commands.yaml        # Single source of truth for prompt utterances
├── test_live_session.py       # Ad-hoc integration test against a running server
├── probe_live_models.py       # Which Live model IDs this API key accepts + per-model proactive_audio / video input probes
├── probe_list_models.py       # Full models.list() across v1alpha/v1beta/v1 (fallback when the curated candidate list returns nothing)
├── probe_all_models.py        # Dump models.list() with capability filters
├── probe_retrace_config.py    # End-to-end verification of the production ephemeral-token config
├── probe_31_config.py         # Bisect which LiveConnectConfig field the current Live model rejects
├── probe_gemini_live_system_instruction.py  # Canary probe: verify Gemini honors the system prompt baked into ephemeral tokens
├── probe_tool_reliability.py  # HIT/MISS rate for advance_step tool calls over many trials
├── probe_diagnostic_reliability.py  # HIT/MISS rate for troubleshoot tool calls (identify_product, confirm_identification, search_procedures, web_search_for_fix, handoff_to_learner)
├── fixtures/                  # Bundled manuals loaded on first start when absent from the DB (e.g. levoit_lv600hh_manual.pdf)
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
| `POST` | `/api/expert/manuals/upload` | Multipart PDF upload. Returns **202** with `{manual_id}`; background task runs `services.manual_ingest.process_pdf(mode="expert")` → new procedure with `source_type="manual"`. Per-step manual-page PNG is written to `step.clip_url`. |
| `GET`  | `/api/expert/manuals/{manual_id}` | Poll ingest status. When ready, returns the resulting `procedure_id`. |
| `GET`  | `/api/manual_images/{procedure_id}/step_{n}.png` | Rendered manual-page PNG used in place of an MP4 clip for manual-sourced procedures. |

### Icons — `/{prefix}/{name}.png`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/{prefix}/{name}.png` | Iconify icon (e.g. `/lucide/refrigerator.png`) rendered to a white-on-transparent PNG via `cairosvg`, disk-cached. |

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

### Troubleshoot — `/api/troubleshoot`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/troubleshoot/session/start` | Body: `{voice?}`. Returns session id, troubleshoot system prompt, tool definitions, and ephemeral token. No procedure required at start. |
| `POST` | `/api/troubleshoot/session/{id}/token[?handle=...]` | Resumption-aware token refresh. |
| `POST` | `/api/troubleshoot/session/{id}/tool-call` | Tools: `identify_product`, `confirm_identification`, `search_procedures`, `web_search_for_fix`, `handoff_to_learner`. |
| `GET`  | `/api/troubleshoot/session/{id}/state` | Current phase, identified product, candidate procedures, diagnostic log. |
| `POST` | `/api/troubleshoot/session/{id}/handoff` | Promote a chosen `procedure_id` into a fresh learner session; returns the full `SessionStartResponse`. |
| `DELETE` | `/api/troubleshoot/session/{id}` | End and discard. |
| `GET`  | `/api/troubleshoot/sessions` | List active troubleshoot sessions. |

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

Sample videos live under `test_data/`. `test_live_session.py` drives a full Live session against a running server. The probe scripts (`probe_live_models.py`, `probe_list_models.py`, `probe_retrace_config.py`, `probe_31_config.py`, `probe_gemini_live_system_instruction.py`, `probe_tool_reliability.py`) are the source of truth for "which models / features work right now on this API key." For tuning the coaching prompt and tool flow without rebuilding the iOS app, use the interactive REPL: `python -m gemini_live_playground --procedure-id <uuid>`.

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
