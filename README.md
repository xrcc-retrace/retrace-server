<div align="center">

<img src="docs/assets/retrace-banner.svg" alt="Retrace" width="100%" />

# Retrace — Backend

### Record an expert once. Coach every learner forever.

FastAPI control plane for **Retrace**, an AI coaching system for hands-on tasks. 🥈 2nd place at the XRCC Berlin 2026 hackathon; work continued through 2026.
An expert records a task once on Ray-Ban Meta glasses or an iPhone. Gemini turns the video into a structured procedure. Learners are then coached by voice, hands-free, while a visual oracle checks their work.

[**🌐 retracelab.com**](https://retracelab.com) · [**▶ Demo**](https://www.youtube.com/watch?v=azK-CLpyRQg) · [**🎬 Full walkthrough**](https://www.youtube.com/watch?v=CqUj5HtP0QE) · [**📱 iOS client**](https://github.com/xrcc-retrace/retrace-ios)

</div>

---

## System at a glance

```
  Capture            Phone (iOS)                     Backend (this repo)
┌───────────┐      ┌────────────────────┐        ┌──────────────────────────┐
│ Ray-Ban   │      │ Capture + mp4      │  REST  │ Video → procedure (Pro)  │
│ Meta      │ ───► │ Gemini Live client │ ◄────► │ Ephemeral tokens         │
│ glasses   │      │ Token actor        │        │ Visual oracle (Flash)    │
│ or iPhone │      │ Lens HUD           │        │ Interview planner (Flash)│
└───────────┘      └─────────┬──────────┘        │ Tool-call dispatcher     │
                             │ direct WebSocket   │ SQLite + ffmpeg clips    │
                             ▼                    └────────────┬─────────────┘
                   ┌─────────────────────────────────────────────▼───────────┐
                   │                 Gemini API (Google)                     │
                   │   Live 3.1 Flash  ·  2.5 Pro  ·  2.5 Flash              │
                   └─────────────────────────────────────────────────────────┘
```

Two rules shape everything here:

1. **Voice and video go phone → Gemini Live directly.** A server hop adds 100–200 ms and kills conversational feel. The backend never touches the audio path.
2. **Gemini Live is never allowed to decide anything on its own.** It talks and listens. Every decision that matters (is the step done, is there a mistake, is there a gap worth asking about) is made out-of-band by a small Gemini Flash call with a narrow prompt, and the result is injected into Live as a text turn. This is the same pattern in all three flows.

The product has three flows. Each gets its own router, session store, prompt builder, and tool handlers (`routers/<mode>.py` + `services/<mode>/`), all sharing one dispatcher and one token minter.

---

## 1 · Expert flow — one video becomes a procedure

**What the expert sees.** Pick glasses or iPhone, hit record, do the task while narrating. Optionally, an AI interviewer listens and asks the one question a learner would have needed ("what socket size is that?") in a pause. Stop, review, upload. A few minutes later the procedure appears with per-step clips, tips, warnings and machine-checkable completion criteria.

### Video → structured procedure

```
POST /api/expert/upload  (multipart video [+ interview JSON])
  ├─ 202 {id, status: "processing"}        ← returns immediately
  └─ background task
       ├─ ffprobe        authoritative duration (Gemini timestamps are clamped to it)
       ├─ Files API      upload + poll until ACTIVE
       ├─ Gemini 2.5 Pro structured output → GeminiVideoProcedureOutput
       │                 steps · completion_criteria · error_criteria · visual_verifiability
       │                 display_summary · tips · warnings · parts glossary
       ├─ Flash          icon_symbol (Iconify id) + category
       ├─ ffmpeg         step_{n}.mp4 (H.264+AAC, faststart, bf=0, BT.709) + lens variant
       └─ SQLite         status: completed | completed_partial | failed
```

Design points:

- **Completion criteria are written for a camera, not a human.** Pro is prompted to describe each step's end-state as something visible in a single frame ("tray fully seated, handle flush"), and to tag `visual_verifiability` as `reliable | marginal | none`. `none` steps (counting, timing, torque) are never visually checked. This gate turned out to be the single biggest false-positive reducer in evals.
- **Per-step failures don't fail the upload.** A bad timestamp range marks that step `clip_status="failed"` and the procedure `completed_partial`.
- **Re-extraction is a first-class operation.** `EXTRACTION_PIPELINE_VERSION` is stamped on each procedure; on startup, stale video procedures are re-extracted from the kept source video (`services/reprocess.py`). Anything that must survive that (the interview log below) is persisted next to the video, not in the DB row.
- **Manuals work too.** `POST /api/expert/manuals/upload` runs the same structured extraction over a PDF, rasterizes each step's source page to a PNG and writes it into `step.clip_url`, so manual-sourced procedures flow through the same clip endpoints with no client branching.

### The expert interviewer (runs during recording)

Experts under-explain because of the curse of knowledge. The interviewer's job is to capture what is in the expert's head but not on camera, without constantly interrupting. Live is deliberately not trusted to time this; three independent loops are:

| Loop | Where | Decides |
|---|---|---|
| **Planner** | server, Gemini Flash, one call per ~5 s `observe` tick | *Whether* there is a gap. Transcribes the 5 s narration chunk, maintains a gap ledger (`tool_spec`, `safety`, `why`, …), proposes at most one question with a confidence. |
| **Policy** | server, pure Python, unit-tested | *If* it is worth a live interruption: time-sensitive priorities only, ≥60 s between asks, max 6 per session, 0.7 confidence floor, topic + token-Jaccard dedup, "quiet mode" on request. Everything else is deferred to the debrief. |
| **Pause gate** | phone | *When*: ≥1.5 s of silence, nothing playing, no stop countdown, question TTL not expired. Then it injects `[ASK] …` into Live. |

Live only voices the question and hears the answer. The phone opens the mic to Live **only inside an answer window**; otherwise Live would reply to every pause in the narration (3.1 tokens reject `proactivity`). The phone owns the `InterviewLog`, uploads it as the `interview` part, and the server persists it at `data/interviews/{id}.json`. Facts extracted from the Q/A are folded into the extraction prompt (`=== EXPERT INTERVIEW NOTES ===`), and the AI's speech windows are ducked (−22 dB) out of every step clip with an ffmpeg `volume=enable=between(t,…)` filter, so learners never hear the interviewer.

Full architecture and the failure matrix: [`docs/expert-interviewer-design.md`](docs/expert-interviewer-design.md). Policy tests: `python -m unittest tests.test_interview_policy`.

### Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/expert/upload` | multipart `video` + optional `interview` → 202. Interview log is strict (`extra="forbid"`, 2 MiB cap). |
| `POST` | `/api/expert/manuals/upload` · `GET /api/expert/manuals/{id}` | PDF ingest + status poll |
| `POST` | `/api/expert/interview/session/start` | `{voice?, transport?, task_hint?}` → prompt + token (no tools) + policy constants |
| `POST` | `/api/expert/interview/session/{id}/observe` | multipart frame + 5 s PCM/WAV + `video_t` + `phase` → `{status, ask?}`; single in-flight, idle-skip on silent frames |
| `POST` | `/api/expert/interview/session/{id}/answer` · `/debrief` · `/finalize` | record outcome · leftover questions as `[DEBRIEF]` · `InterviewLog` with facts |
| `GET/PUT/DELETE` | `/api/procedures[/{id}[/steps/{n}]]` | CRUD, step reorder |
| `GET` | `/api/clips/{id}/step_{n}.mp4` · `step_{n}.jpg` · `step_{n}_done.jpg` | clip, first-frame poster, last-frame "done state" (the oracle's reference) |

---

## 2 · Learner flow — voice coaching that checks the work

**What the learner sees.** Pick a procedure, put the glasses on, start. A voice walks them through the current step and answers questions. When the work looks done, the coach says so and moves on. If the frame shows a known mistake, the coach interjects. The learner never taps "next".

### Session start

```
POST /api/learner/session/start {procedure_id, voice?, active_monitoring?, auto_advance?}
  → session_id, procedure, system_prompt (CURRENT STEP ONLY), tools, gemini_config,
    ephemeral_token (v1alpha auth_tokens; model + prompt + tools + voice + compression
    + resumption + transcription locked in — the phone never sees the API key)
```

The phone opens `BidiGenerateContentConstrained` with that token and streams mic audio plus ~1 fps JPEG. From here the server is only called for decisions.

### The oracle owns step advancement

```
every ~4 s (Active mode)                          on "done" / "is this right?"
POST /verify  (camera JPEG)                       Live calls check_step(claim)
   │                                                 │ phone intercepts: a frame can't ride
   ▼                                                 │ a tool-call body
Flash oracle: frame vs step_N_done.jpg               ▼
  + completion_criteria + error_criteria          POST /verify?mode=on_demand&claim=…
   │                                                 │
   ├─ advanced   → server moves to step N+1          ├─ confirmed → advance, phone synthesises toolResponse
   ├─ error      → error_text                        ├─ not_confirmed → coach asks to adjust
   └─ pending                                        └─ oracle down → trust the learner (advance verbally)
   ▼
phone injects "[ADVANCE] …" / "[CORRECTION] …" as a text turn → Live speaks it
```

- **Image-to-image, not image-to-text.** The oracle gets the expert clip's last frame as a reference "done" state next to the learner's frame, and is biased against false positives. Dwell: 1 consecutive confirmation for `reliable` steps, 2 for `marginal`.
- **Insist-once.** A second rejected done-claim advances verbally. A skeptical model can never trap a human.
- **Active / Passive.** One toggle. Active polls the oracle (auto-advance + interjection); Passive is on-demand only. `active_monitoring` and `auto_advance` are always sent equal by the client.
- **Current-step-only prompt.** The system prompt embeds only the current step. Future steps arrive exclusively via `[ADVANCE]` turns, and the `check_step` response carries zero next-step content, so Live cannot narrate ahead. On token refresh the prompt is re-anchored on the server's current step.

### Staying connected indefinitely

Gemini Live's "2-minute audio+video limit" is a context-fill limit, not a timer. Tokens bake in `ContextWindowCompression(trigger=100k, target=40k)` and `SessionResumption`. On `goAway` the phone mints `POST /session/{id}/token?handle=<new_handle>`, reconnects, waits for `setupComplete`, and injects `GET /session/{id}/context-summary` as a user turn so the model wakes up on the real current step.

### Validated, not vibed

`evals/` scores the oracle and the extractor against **self-supervised ground truth**: a step clip's last frame is a true "done", the next step's first frames are true "not done". Zero annotation cost. Across the two largest runs, prompt tightening took early-advance false positives from 31% to 10% at roughly unchanged recall (44% → 41%); the small fixtures that hit 0% false positives did so by collapsing recall, which the harness made visible instead of hiding. The evals also showed the `visual_verifiability` gate was load-bearing. See [`evals/README.md`](evals/README.md).

### Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/learner/voices` | 8 Live voices with preview WAVs |
| `POST` | `/api/learner/session/start` | see above |
| `POST` | `/api/learner/session/{id}/verify` | multipart `frame`; `mode=poll\|on_demand`, `claim=done\|question` |
| `POST` | `/api/learner/session/{id}/token[?handle=]` | resumption-aware token refresh |
| `POST` | `/api/learner/session/{id}/tool-call` | `get_reference_clip`, `go_to_step` (and `advance_step` for the HUD swipe) |
| `GET` | `/api/learner/session/{id}/state` · `/context-summary` | step, history (`method: visual\|verbal`), re-orientation text |

---

## 3 · Troubleshooter flow — fix something nobody recorded

**What the user sees.** Point the camera at a broken thing and describe the problem. The AI identifies the product, confirms it, searches the library. If a procedure exists, it coaches it. If not, it researches a fix on the web, builds a cited procedure on the spot, and coaches that.

```
POST /api/troubleshoot/session/start {voice?}          (no procedure required)
   │
   ▼  Gemini Live tools, forwarded by the phone to /tool-call
identify_product ──► confirm_identification ──► search_procedures
                                                   │  Flash ranks the existing library by product + symptom
                                                   │  (no separate index; fine at library scale)
                                                   ├─ match ───────────────────────────────┐
                                                   └─ no match → web_search_for_fix        │
                                                        Gemini Pro + GoogleSearch grounding│
                                                        → structured procedure + cited URLs│
                                                                                            ▼
                                                                                   handoff_to_learner
                                                                     POST /session/{id}/handoff → full learner
                                                                     SessionStartResponse; phone swaps sockets
```

- **Phases are server state**, not prompt state: `identify → diagnose → researching → ready_to_coach`, with a diagnostic log the phone renders.
- **Web research is two-step** (grounded search, then structured-output JSON) because grounding and JSON schema cannot be combined in one call. Sources are kept on the procedure (`source_type="web"`, `sources[]`).
- **A smaller sibling, Ask AI** (`/api/ask_ai/*`), is the same shape with one tool: audio-only voice search of the library from the glasses, then handoff.

| Method | Path |
|---|---|
| `POST` | `/api/troubleshoot/session/start` · `/{id}/token` · `/{id}/tool-call` · `/{id}/handoff` |
| `GET` | `/api/troubleshoot/session/{id}/state` · `/api/troubleshoot/sessions` |

---

## Cross-cutting decisions

| Decision | Why |
|---|---|
| Ephemeral tokens (`v1alpha auth_tokens`, `uses=1`) | The phone connects to Gemini directly but never holds the API key. A leaked token is locked to one model, prompt, tool set and voice. |
| Out-of-band Flash for every judgment | Live hallucinated completions on empty scenes; a narrow-prompt oracle with an image reference does not. Same split reused for the interviewer. |
| Mode-based modularity | Each flow is `routers/<mode>.py` + `services/<mode>/{session,prompt,handlers}.py` registering on one dispatcher. No per-mode branches inside shared code. |
| `voice_commands.yaml` | Single designer-editable source for every spoken phrase the prompts teach the model; linted by `validate_voice_commands.py`. |
| Bonjour (`_retrace._tcp`) | Hackathon Wi-Fi hands out new IPs daily. The phone discovers the server; a manual override exists. |
| SQLite + files on disk | Tens of procedures, one box. WAL mode, ALTER-TABLE migrations, source videos kept for re-extraction. |

### Gemini Live configuration (locked into every token)

| | |
|---|---|
| Model | `gemini-3.1-flash-live-preview` (`RETRACE_LIVE_MODEL` to roll back) |
| Compression | trigger 100k → target 40k tokens |
| Resumption | handle re-baked on every refresh |
| Transcription | input + output, so the phone can render a transcript feed |
| Media | `MEDIA_RESOLUTION_MEDIUM` for Live (the oracle sees full-res frames out-of-band) |
| Oracle / planner model | `gemini-2.5-flash` (`RETRACE_ORACLE_MODEL`, `RETRACE_PLANNER_MODEL`) |

Gotchas worth knowing: `generate_content` rejects HTTP deadlines under 10 s; an empty `function_declarations=[]` is rejected, so tool-less tokens send no tools block; the Constrained endpoint delivers JSON control frames as *binary* WebSocket messages.

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
echo "GEMINI_API_KEY=..." > .env
python main.py          # http://0.0.0.0:8000, dashboard at /
```

| Env var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `RETRACE_ADVERTISE_BONJOUR` | `1` | `0` on cloud hosts |
| `RETRACE_LIVE_MODEL` / `RETRACE_ORACLE_MODEL` / `RETRACE_PLANNER_MODEL` | see above | model overrides |

```bash
curl -F "video=@/path/to/your_recording.mp4" localhost:8000/api/expert/upload   # sample videos are not shipped in the repo
python -m unittest tests.test_interview_policy        # 25 policy/ducking/parse tests
python evals/run_all.py                               # oracle + extractor evals
python -m gemini_live_playground --procedure-id <id>  # REPL against the production Live config
```

The `probe_*.py` scripts are the record of what the Gemini API accepted on a given day (model IDs, config fields, tool-call hit rate). They were kept on purpose: every model upgrade broke something, and the probes found it in minutes.

### Deployment

Production ran on an AWS Lightsail box (Caddy + uvicorn under systemd) with `.github/workflows/deploy.yml` pulling and restarting on push to `main`. The host has since been decommissioned and the deploy + SSH-bootstrap workflows were removed (see git history); `reextract.yml` remains as a dispatch-only reference.

---

## Layout

```
main.py · config.py            app entry (CORS, Bonjour, fixtures), paths + env
models/                        pydantic: procedure, session, troubleshoot, interview
routers/
  expert.py                    upload + background pipeline
  expert_interview.py          /api/expert/interview/*
  expert_manuals.py            PDF ingest
  learner.py                   session start / verify / token / tool-call / state
  troubleshoot.py · ask_ai.py  diagnose-then-coach · audio-only search
  procedures.py · icons.py     CRUD + clips · Iconify PNGs
services/
  gemini_processor.py          Files API + Pro structured extraction (+ interview notes)
  clip_extractor.py            ffmpeg clips, lens variants, speech ducking
  step_oracle.py               visual verification (Flash, image-to-image)
  prompt_builder.py            current-step-only coaching prompt + tool declarations
  ephemeral_token.py           v1alpha auth_tokens with the full Live config
  session.py                   in-memory sessions + shared tool dispatcher
  reprocess.py                 startup re-extraction of stale procedures
  expert_interview/            planner · policy · facts · notes · store · prompt · session
  troubleshoot/ · ask_ai/      session · prompt · handlers (· web_research)
  database.py                  SQLite WAL + migrations
evals/                         self-supervised oracle + extractor evals
tests/                         stdlib unittest (interview policy)
gemini_live_playground/        REPL for prompt/tool tuning
docs/                          expert-interviewer-design.md
```

---

## License

[MIT](LICENSE). Demo recordings, the local database and third-party product manuals are not included; the startup fixture loader downloads the Nespresso descaling guide from nespresso.com on first run.

<div align="center">

Built at XRCC Berlin 2026 by Hyunseok Hwang and Arthur Lee · iOS client: [xrcc-retrace/retrace-ios](https://github.com/xrcc-retrace/retrace-ios)

</div>
