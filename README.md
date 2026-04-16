# Retrace — Backend Server

FastAPI backend for Retrace, the AI coaching system for the XRCC Berlin 2026 hackathon.

> **One-liner:** Record an expert once. Coach every learner forever.

The server handles the intelligence layer of a three-tier architecture (**glasses → phone → server**):

- **Expert pipeline** — Accepts a video upload, runs it through Gemini 2.5 Pro to extract a structured procedure, and uses ffmpeg to slice a playable clip for each step.
- **Learner pipeline** — Spins up a coaching session, assembles the system prompt, and mints an ephemeral Gemini Live token so the phone can open a direct WebSocket to Gemini Live (no server hop on the voice path).
- **Discovery** — Advertises itself on the local network via Bonjour (`_retrace._tcp.local`) so the iOS app can auto-discover the server.

## Prerequisites

- Python 3.11+
- `ffmpeg` (`brew install ffmpeg`)
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
- From an iOS device on the same Wi-Fi: Bonjour discovery will find the server automatically.

## Architecture Overview

```
Expert flow:
  iOS app --[POST multipart video]--> /api/expert/upload
         <--[202 + procedure_id]----
                                    |
                                    v (background task)
                           Gemini 2.5 Pro Files API
                                    |
                                    v
                           Structured procedure JSON
                                    |
                                    v
                           ffmpeg clip extraction
                                    |
                                    v
                           SQLite + clips on disk

Learner flow:
  iOS app --[POST procedure_id]---> /api/learner/session/start
         <--[session + system prompt + ephemeral token]---
                                    |
                                    v
  iOS app <=== direct WebSocket ===> Gemini Live API
                                    |
                                    | (tool calls)
                                    v
  iOS app --[POST tool-call]-----> /api/learner/session/{id}/tool-call
         <--[tool result]----------
```

Voice traffic never hits this server — that hop would add 100–200 ms of latency and break the conversational feel. Instead, the server mints a short-lived Gemini auth token (via the `v1alpha` `auth_tokens` API), locked to a specific model, system prompt, and tool set, and the phone uses it to open the Live WebSocket directly.

## Project Layout

```
main-local-server-test/
├── main.py                  # FastAPI app, CORS, Bonjour advertising, dashboard
├── config.py                # Paths, API key, ffmpeg location (auto-creates dirs)
├── requirements.txt
├── models/
│   ├── procedure.py         # Gemini structured-output schema + API response models
│   └── session.py           # Session start/state, tool calls, ephemeral token
├── routers/
│   ├── expert.py            # POST /api/expert/upload — async pipeline
│   ├── learner.py           # Session lifecycle + voice list
│   └── procedures.py        # CRUD for procedures / steps / clips
├── services/
│   ├── gemini_processor.py  # Upload video to Gemini, poll, generate structured JSON
│   ├── clip_extractor.py    # Per-step MP4 extraction via ffmpeg (H.264+AAC, faststart)
│   ├── database.py          # SQLite (WAL). Tables: procedures, steps
│   ├── session.py           # In-memory session store + tool-call dispatcher
│   ├── prompt_builder.py    # System prompt assembly, tool definitions, Gemini config
│   └── ephemeral_token.py   # Mints short-lived Gemini Live auth tokens
├── static/
│   ├── index.html           # SOP review dashboard
│   └── voice_previews/      # Pre-generated WAV previews of Gemini voices
├── scripts/
│   └── generate_voice_previews.py
├── data/
│   ├── clips/{procedure_id}/step_{n}.mp4
│   └── retrace.db           # SQLite
└── uploads/                 # Source expert videos ({uuid}.{ext})
```

## API Reference

### Health & Static

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `GET` | `/` | SOP review dashboard. |
| `GET` | `/api/uploads/{filename}` | Serve the original source video for review playback. |
| `GET` | `/static/voice_previews/{name}.wav` | Pre-generated voice preview. |

### Expert — `/api/expert`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/expert/upload` | Multipart video upload. Returns **202** immediately with `{id, status}`; Gemini analysis + clip extraction run in a background `asyncio` task. Poll `/api/procedures/{id}` for `status: processing / completed / failed`. |

### Procedures — `/api/procedures`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/procedures` | List all procedures (includes `status`). |
| `GET` | `/api/procedures/{id}` | Get one procedure with all steps. |
| `PUT` | `/api/procedures/{id}` | Update `title`, `description`, or reorder steps via `step_order`. |
| `PUT` | `/api/procedures/{id}/steps/{n}` | Update a single step's fields (`title`, `description`, `tips`, `warnings`, `timestamp_start`, `timestamp_end`). |
| `DELETE` | `/api/procedures/{id}` | Delete procedure, steps, and clips. |
| `GET` | `/api/clips/{id}/step_{n}.mp4` | Per-step clip (H.264+AAC, faststart). |

### Learner — `/api/learner`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/learner/voices` | Server-authoritative list of Gemini Live voices with `preview_url`. |
| `POST` | `/api/learner/session/start` | Start a session. Body: `{procedure_id, auto_advance?, voice?}`. Returns the session id, full procedure, assembled system prompt, tool definitions, Gemini config, and an ephemeral token. |
| `POST` | `/api/learner/session/{id}/token` | Mint a fresh ephemeral token (tokens are short-lived). |
| `POST` | `/api/learner/session/{id}/tool-call` | Forward a Gemini function call (`advance_step`, `get_reference_clip`, `go_to_step`, …) from the phone; the server executes and returns the result for the phone to hand back to Gemini Live. |
| `GET` | `/api/learner/session/{id}/state` | Current step, status, history. |
| `GET` | `/api/learner/session/{id}/context-summary` | Compact text blob the phone injects as a user message after context compression or reconnect to re-orient the model. |
| `DELETE` | `/api/learner/session/{id}` | End and discard the session (sessions are held in memory, not DB). |

## Data Storage

- `uploads/` — source expert videos (`{uuid}.{ext}`).
- `data/clips/{procedure_id}/step_{n}.mp4` — per-step clips.
- `data/retrace.db` — SQLite (WAL mode). Tables: `procedures`, `steps`. `tips` and `warnings` are JSON-encoded arrays.
- Sessions are **in-memory only** (a dict keyed by session id) — they die with the server.

## Testing

No test framework — manual testing via curl and the dashboard at `/`.

```bash
# Health
curl http://localhost:8000/health

# Upload (returns 202 immediately; poll /api/procedures/{id} until status=completed)
curl -X POST -F "video=@test_data/some_video.mp4" http://localhost:8000/api/expert/upload

# List procedures
curl http://localhost:8000/api/procedures

# Start a learner session
curl -X POST -H "Content-Type: application/json" \
  -d '{"procedure_id":"<id>"}' http://localhost:8000/api/learner/session/start
```

Sample videos live under `test_data/`. `test_live_session.py` is an ad-hoc script for driving a Live session against a running server.

## Critical Architecture Decisions

1. **Voice goes direct** (phone → Gemini Live). A server hop costs 100–200 ms per exchange and kills conversational feel.
2. **Audio-only Live sessions** (15 min) beat audio+video (2 min) — the context compressor evicts reference frames first, which would defeat visual verification.
3. **Visual verification uses Gemini 2.5 Pro via REST**, not the Live API. Pro has stronger vision than Flash, no context eviction, and no session-length cap.
4. **Tool calling is the server↔model channel.** Gemini Live issues function calls; the phone intercepts and forwards them here.
5. **Ephemeral tokens** solve key distribution. The server holds the Gemini API key and mints short-lived tokens locked to a specific model / prompt / tool set so the phone never has the raw key.
