<div align="center">

<img src="https://github.com/xrcc-retrace/.github/raw/main/profile/assets/retrace-banner.svg" alt="Retrace" width="100%" />

# Retrace — Backend Server

### Record an expert once. Coach every learner forever.

The FastAPI backend for **Retrace**, an AI coaching system built for the **XRCC Berlin 2026** hackathon.
The control plane that turns expert videos into structured procedures, mints ephemeral Gemini Live tokens for the phone, and dispatches tool calls during a coaching session.

[**▶ Watch the demo**](https://www.youtube.com/watch?v=azK-CLpyRQg) &nbsp;·&nbsp; [**🎬 Full walkthrough video**](https://www.youtube.com/watch?v=CqUj5HtP0QE) &nbsp;·&nbsp; [**🌐 Live API**](https://retracexrcc.duckdns.org/health) &nbsp;·&nbsp; [**🏛 Org overview**](https://github.com/xrcc-retrace) &nbsp;·&nbsp; [**📱 iOS repo**](https://github.com/xrcc-retrace/retrace-ios)

</div>

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
- **Current-step-only system prompt** — only the current step is embedded; future steps reach the model exclusively through `advance_step` tool responses. Prevents narration-ahead and forces the tool call. Re-anchors on the session's current step on token refresh / resumption.
- **Shared tool dispatcher** — both Learner and Troubleshoot register handlers with a single dispatcher at import time. The phone forwards tool calls; the server returns `{result: {...}}` for the phone to hand back to Gemini as `toolResponse`.
- **Web-grounded Troubleshoot** — `web_search_for_fix` runs a two-step Gemini Pro pipeline (GoogleSearch grounding + structured-output JSON) that returns a fresh procedure with cited URLs. Replaced an earlier brittle PDF-fetch path.
- **PDF manual ingest** — `/api/expert/manuals/upload` runs the same structured-output extraction over a PDF and rasterizes each step's source page to a PNG, written into `step.clip_url` so manual-sourced procedures plug into the same clip-serving endpoints as video-sourced ones.
- **Iconify icon library** — `GET /{prefix}/{name}.png` serves any Iconify ID as a white-on-transparent PNG via cairosvg, disk-cached. The iOS HUD references icons without bundling assets.
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
                                    │ (function calls: advance_step / get_reference_clip / go_to_step)
                                    ▼
iOS app --[POST tool_name, arguments]--> /api/learner/session/{id}/tool-call
       <--[{result: {...}}]-------------
                                    │ (phone unwraps result, sends back to Gemini as toolResponse)
                                    ▼
                       Gemini continues speaking / listening / watching
```

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
2. **Visual verification runs inline in Gemini Live** — not as a separate server-side pass. The model sees the camera feed alongside audio, watches for each step's `completion_criteria`, and fires `advance_step` via function calling. `auto_advance=True` tags completions as `method="visual"`; manual mode is `"verbal"`. A Gemini 2.5 Pro REST clip-to-clip fallback is deferred unless inline detection proves unreliable.
3. **Audio + video share one Live session.** Context compression + session resumption unlock indefinite duration. The "2-minute audio+video" Gemini limit is a context-fill limit, not a hard timer.
4. **Current-step-only system prompt.** Future-step details are delivered exclusively through `advance_step` tool responses. Prevents the model from narrating ahead. On token refresh / resumption, the prompt is re-anchored on `session["current_step"]` so a reopened socket doesn't act like step 1.
5. **Tool calling is the model↔server channel.** Gemini issues function calls; the phone intercepts, injects `session_id`, and forwards to `/tool-call`. Server handler result is wrapped as `{result: {...}}`; the phone unwraps before handing back to Gemini.
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

### Icons — `/{prefix}/{name}.png`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/{prefix}/{name}.png` | Iconify icon (e.g. `/lucide/refrigerator.png`) rendered to a white-on-transparent PNG via `cairosvg`, disk-cached. |

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all (newest first) with `status`, `error_message`, `step_count`. |
| `GET` | `/api/procedures/{id}` | Get one with all steps (includes `completion_criteria`, `clip_status`). |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, or reorder steps via `step_order: [old_step_number, …]`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields. |
| `DELETE` | `/api/procedures/{id}` | Delete procedure + cascade steps. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices with `preview_url`. |
| `POST` | `/api/learner/session/start` | Body: `{procedure_id, auto_advance?, voice?}`. Returns session id, full procedure, the assembled **current-step-only** system prompt, tool definitions, Gemini config, and an ephemeral token. |
| `POST` | `/api/learner/session/{id}/token[?handle=...]` | Mint a fresh ephemeral token. Pass `?handle=<sessionResumptionUpdate.new_handle>` to bind it to the prior session so the next WebSocket continues with compressed context preserved. |
| `POST` | `/api/learner/session/{id}/tool-call` | Forward a Gemini function call (`advance_step`, `get_reference_clip`, `go_to_step`); server executes and returns `{result: {...}}`. |
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
│   ├── icons.py               # GET /{prefix}/{name}.png — Iconify icon serving (white-on-transparent PNG, cached)
│   ├── learner.py             # Voice list + session lifecycle (start, tool-call, state, context-summary, token refresh with ?handle=)
│   ├── procedures.py          # CRUD for procedures / steps / clips
│   └── troubleshoot.py        # Mode-parallel to learner; importing this router imports services.troubleshoot which registers handlers with the shared dispatcher
├── services/
│   ├── gemini_processor.py    # Uploads video to Gemini Files API, polls, generates structured procedure JSON via Gemini 2.5 Pro. Falls back to gemini-2.5-flash on 503.
│   ├── clip_extractor.py      # Per-step ffmpeg extraction (H.264+AAC, faststart, bf=0, bt709 colr tags for iOS VideoToolbox)
│   ├── video_probe.py         # ffprobe authoritative duration
│   ├── database.py            # SQLite WAL + ALTER-TABLE migrations. Tables: procedures, steps.
│   ├── session.py             # In-memory session store, step advancement with method tagging, context-summary builder, tool-call dispatcher (shared)
│   ├── prompt_builder.py      # Current-step-only system prompt. Tool definitions + Gemini config.
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
├── fixtures/                  # Bundled manuals loaded on first start when absent from the DB (e.g. levoit_lv600hh_manual.pdf)
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4
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
- **Context compression:** `trigger_tokens=100_000 → target_tokens=40_000`. At ~0.5 fps + `MEDIA_RESOLUTION_LOW`, first compression cycle fires around 25 min, then every ~12 min.
- **Session resumption:** Gemini periodically emits `sessionResumptionUpdate.new_handle`. iOS persists the latest; on reconnect, `POST /session/{id}/token?handle=...` bakes it into `SessionResumptionConfig`.
- **Transcription:** both input (learner) and output (AI reply) audio are transcribed so the iOS activity feed renders text.
- **Media resolution:** `MEDIA_RESOLUTION_LOW` keeps video token burn at ~33 tok/s.
- **Token uses:** `uses=1`. iOS mints a fresh token per connection attempt (initial + retry + resumption).
- **Expiry format:** ISO8601 with **millisecond** precision. iOS's `ISO8601DateFormatter.withFractionalSeconds` caps at 3 fractional digits — Python's default 6-digit microsecond output silently fails to parse.

---

## Visual verification reliability

Design steps around what VLMs can actually detect. The `completion_criteria` string on each step is what Gemini Live uses to decide when to call `advance_step` — write **observable end-states**, not actions.

| Reliability | What works |
|---|---|
| **Reliable** | Object presence/absence, large state changes (lid open/closed, tray in/out) |
| **Marginal** | Small object state, correct orientation |
| **Unreliable** | Fine alignment, sub-millimeter changes, torque/fit |

The extraction prompt in `gemini_processor.py` asks Gemini to phrase completion criteria as observable end-states; keep it that way if you retune the prompt.

---

## Production deployment

Retrace runs in production on **AWS Lightsail** at **https://retracexrcc.duckdns.org**. The iOS client falls back to this URL when Bonjour fails or when the user picks "Cloud" in Server Settings.

### Stack

| Layer | What |
|---|---|
| Host | Lightsail $12/mo — Ubuntu 22.04, 2 GB RAM, 2 vCPU, 60 GB SSD + 2 GB swap |
| Region | `ap-southeast-1` (Singapore) — migrating to `eu-central-1` before the Berlin demo |
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
ssh -i ~/.ssh/<deploy-key>.pem ubuntu@<server-ip>

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

### Region migration (planned before Berlin demo)

Singapore → Frankfurt for ~10 ms latency vs the XRCC Berlin venue:

1. Lightsail console → Snapshots → **Copy snapshot to another region** → `eu-central-1`.
2. Launch a new instance from the copied snapshot.
3. Allocate a new static IP in Frankfurt, attach.
4. Update DuckDNS A record to the new IP.
5. Delete the Singapore instance and release its static IP.

No code changes — `cloudBaseURL` in iOS stays `https://retracexrcc.duckdns.org`; only the DNS target moves.

---

## Data storage

- `uploads/` — source expert videos, named `{uuid}.{ext}` (keep the original extension so ffprobe reads the container correctly).
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips.
- `data/retrace.db` — SQLite (WAL). Two tables:
  - `procedures(id, title, description, total_duration, source_video, created_at, status, error_message, source_type, …)`
  - `steps(id, procedure_id, step_number, title, description, timestamp_start, timestamp_end, tips, warnings, completion_criteria, clip_url, clip_status)`
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

Built by the XRCC Retrace team for the Berlin 2026 hackathon.

[**Org profile**](https://github.com/xrcc-retrace) &nbsp;·&nbsp; [**iOS app**](https://github.com/xrcc-retrace/retrace-ios) &nbsp;·&nbsp; [**Live API**](https://retracexrcc.duckdns.org/health)

</div>
