# Expert Interviewer — Architecture & Edge Cases

*A voice agent that listens while the expert records, and asks the question the learner would have needed — at a moment that doesn't break the expert's flow.*

Status (2026-10-15): **implemented for the iPhone transport**, in two worktrees pending merge — `main-local-server-test/.claude/worktrees/expert-interviewer` (branch `worktree-expert-interviewer`) and `retrace-ios/.claude/worktrees/expert-interviewer` (same branch name). File references below are to the current tree.

| Piece | State |
|---|---|
| Server: session / planner / policy / facts / notes / store, `/api/expert/interview/*`, upload `interview` part, extraction notes section, clip ducking, strict `InterviewLog`, voice commands, 25 unit tests | **Done.** Verified end-to-end with curl against Gemini: token mint with zero tools, planner deferring a spec gap and live-asking a safety gap, narration filling a gap, debrief, finalize → 8 facts, upload → notes in the extraction prompt, step clip ducked −31 → −51.5 dB. |
| iOS: `.interview` audio mode (shared session, secondary tap slot, dedicated send converter), `GeminiLiveSessionBase` owner-fed hooks, `ExpertInterviewViewModel` (ticker / pause gate / answer windows / debrief / finalize), frame tee, timeline origin, sidecar log, upload part, phone-side overlay, mode picker | **Done (compiles); not yet exercised on a device** — spike 0(a) in §8 is the first thing to run. |
| Decisions taken | Narration transcript = audio chunk to the planner (§4.3 option B). Debrief on the same socket. Silent fold-in of facts (no attribution). iPhone first. |
| Not done | Glasses transport (HFP volume chime, no-AEC bleed strategy — §4.1); lens-side question card per `DESIGN.md` (phone chrome only today); `record_fact` tool (transcript-derived facts instead); planner evals from recorded sessions. |

---

## 1. Thesis and the one-sentence design

Experts under-explain because of the curse of knowledge: the socket size, the "why", the "how do you know it's tight enough" never make it into the narration. The recording captures what is *visible*; the interviewer captures what is *in the expert's head*.

**Design in one sentence:** the recording pipeline stays exactly as it is and never depends on the interviewer; a server-side Flash *planner* watches a 1 fps frame + the expert's words and decides *whether* there is a gap worth asking about; the phone decides *when* it is safe to speak; Gemini Live only does the speaking and the listening-for-the-answer; everything learned is written to a phone-owned interview log that rides along with the upload and is folded into extraction.

This is the same split that already works for coaching (Live talks; an out-of-band Flash oracle decides), for the same reason: Live cannot be trusted to decide when *not* to speak.

---

## 2. What the code survey changed

| I assumed | What the code actually does | Consequence |
|---|---|---|
| A second `AudioSessionManager` can run next to the recorder | iOS has one shared `AVAudioSession`; every manager instance calls `setCategory`/`setActive` on it, and `stopCapture()` flips it to `.playback` (`AudioSessionManager.swift:1338`). The mic tap has **one primary consumer slot** (`onAudioBuffer`), which the Live base overwrites (`GeminiLiveSessionBase.swift:1053`). | **One** audio manager, shared by recorder and Live via the existing *borrow* pattern. See §4.1. |
| Recording modes can play audio | `.recording` / `.recordingPhoneOnly` have no player node; `playPcm16Audio` returns early (`AudioSessionManager.swift:1494`). | Interview recording must run on a playback-capable mode. The glasses-native path already does this (`GlassesNavigationCoordinator.persistentAudio` is `.coaching`, borrowed by the recorder). |
| Live can share the camera | The base creates its **own** `IPhoneCameraCapture` (`:1865`) or its own DAT stream under `"gemini-live"` at `.high`/2 fps (`:1416`). A second `AVCaptureSession` on the same camera fails; a second DAT stream competes for the BT radio. | New base hook: an *external frame source*. Frames are **tee'd** from the recording pipeline. |
| Live can be told to stay quiet | Tokens on 3.1 can't carry `proactivity`; Live answers every end-of-turn. The in-band `[check]` heartbeat was removed for exactly this (hallucinated replies on empty input). | Live must **not hear the narration**. Mic-to-Live opens only in an *answer window*. See §4.3. |
| Upload can carry context | `POST /api/expert/upload` takes one field, `video`. `process_expert_video(video_path, api_key)` has no context input. | Add `interview` multipart field + prompt section. Manual-ingest (`Form` → `.format()`) is the precedent. |
| Interview facts can be passed transiently | Extraction is **re-run on startup** for every stale video procedure (`reprocess.reextract_stale_procedures`). | Facts must be **persisted next to the video**, not just handed to the first extraction. |
| Server session can be the record of truth | Sessions are in-memory; the server may restart or Bonjour may resolve a different host between Stop and Upload. | The **phone owns the interview log**; the server's session is a planner cache. |
| Tool calls are free | `pendingToolCallIds` closes the mic send gate until the response is posted (`:2207`). | Any Live tool in this mode must be client-intercepted and instant, or not exist. See §4.5. |

---

## 3. Topology

```
                     PHONE                                               SERVER
┌──────────────────────────────────────────────────┐      ┌─────────────────────────────────────┐
│ camera ─┬─▶ ExpertRecordingManager ─▶ .mp4        │      │                                     │
│         └─▶ FrameTee (1 fps JPEG, own queue) ─┬───┼──────┼▶ POST /observe  (frame+audio 5s)    │
│                                               │   │      │   └─ planner.py (Flash)             │
│ mic tap ─┬─▶ writer                           │   │      │       gap ledger ─▶ decision        │
│          ├─▶ 5s PCM ring ─────────────────────┘   │◀─────┼── {none | ask{q,topic,prio,ttl}}    │
│          └─▶ Live (ONLY inside answer window) ────┼──┐   │                                     │
│                                                   │  │   │ POST /session/start  (token)        │
│ InterviewCoordinator                              │  │   │ POST /finalize  (returns server log)│
│   pause gate: silence≥1.5s ∧ no stop-countdown    │  │   │                                     │
│   ∧ budget ok ∧ not reconnecting ∧ ttl not expired│  │   │ POST /api/expert/upload             │
│   ──▶ sendClientTextTurn("[ASK] …") ──────────────┼──┼──▶│   video + interview(json)           │
│   ◀── Live voices question; answer window opens   │  │   │   ├─ data/interviews/{id}.json      │
│   speech windows, Q/A, facts → InterviewLog       │  │   │   ├─ extraction prompt += notes     │
│                                                   │  └───┼─▶ Gemini Live (wss, ephemeral)      │
│ Stop ─▶ Debrief (same socket, camera off) ─▶ Upload│      │   └─ clip_extractor ducks AI windows│
└──────────────────────────────────────────────────┘      └─────────────────────────────────────┘
```

Three independent control loops, each with one job:

| Loop | Owner | Cadence | Decides |
|---|---|---|---|
| **Planner** | server `services/expert_interview/planner.py` | ~5 s, single in-flight | *Is there a concrete, unanswered gap, and is it time-sensitive?* |
| **Pause gate** | phone `InterviewCoordinator` | continuous | *Is this instant safe to speak?* |
| **Voice** | Gemini Live | event-driven | *Say the question naturally; hear the answer; one follow-up max.* |

The planner never speaks. The gate never thinks. Live never decides.

---

## 4. Component design

### 4.1 Audio: one manager, two consumers

**Problem.** The recorder needs the mic; Live needs the mic *and* a speaker; iOS gives us one `AVAudioSession`; `AudioSessionManager` has one primary tap consumer.

**Design.**

- New `AudioSessionManager.Mode.interview` (iPhone) — `.playAndRecord`, `.voiceChat`, `.defaultToSpeaker`, player node attached. This is `.coachingPhoneOnly` plus the recording-side affordances. For glasses, reuse `.coaching` exactly as the glasses-native path already does (`GlassesRecordingHost` passes `externalAudioManager: coordinator.persistentAudio`).
- `IPhoneExpertRecordingViewModel` constructs the manager in `.interview` when the interviewer is enabled; otherwise `.recordingPhoneOnly` as today. **The mode is chosen at `prepare()` and cannot change mid-recording** — that's a product constraint (toggle lives on the pre-record screen).
- `ExpertRecordingManager(borrowedSession:)` already skips `startCaptureAsync`/`stopCapture` — use that constructor so the recorder never flips the shared session.
- `ExpertInterviewViewModel` (subclass of `GeminiLiveSessionBase`) is constructed with `externalAudioManager:` so the base adopts rather than creates.
- **Fan-out.** Keep the writer on `onAudioBuffer`. Give Live the dormant `onAudioBufferSecondary` slot (`installSecondaryAudioConsumer`, currently "no installer in tree"). Change the base to prefer the secondary slot when borrowing. Same tap, same buffer, two closures — zero extra capture cost.
- **Converter contention.** `convertBufferForSend` and `convertBufferToPcmRecording` share one `AVAudioConverter` (`:130`). `AVAudioConverter` is not thread-safe; on glasses both paths convert to the *same* 16 kHz PCM16 mono. Convert **once on the tap thread** and hand the converted buffer to both consumers — fixes contention and halves the work. On iPhone the writer takes the hardware format (AAC via writer), so only the Live path converts.

**Edge cases.**

- **Sample-rate shift.** In `.voiceChat` the hardware rate can differ from what `prepareWriter` baked into the AAC settings (iPhone uses "hardware sample rate" at writer build time, `ExpertRecordingManager.swift:189`). Build the writer **after** `configureSessionOnly()` has settled, not before — or read the rate at first audio sample and re-prepare. Same class of bug as the resolution-change re-prepare already handled for video.
- **AEC changes the mic.** `.voiceChat` applies echo cancellation, AGC and noise suppression. The mp4's audio becomes the processed voice-chat mic. For speech this is fine and *removes most of the AI's own voice from the recording* — a feature. But it is a change from today's raw capture; A/B one recording before committing.
- **Glasses HFP volume.** Downlink rides the separate "call" volume that defaults low (`AudioSessionManager.swift:1558`). First question may be inaudible. Play a short confirmation chime at session start so the expert can adjust volume *before* recording.
- **Glasses radio budget.** HFP SCO is a symmetric synchronous link; the mic uplink already reserves it during recording, so the downlink costs **no additional** BT bandwidth. Video (BT Classic, `.medium`) is unaffected. Lens text sends share the radio — keep the question card small.
- **No VPIO on glasses** (`:981`) — no AEC, so the glasses' open-ear speakers leak slightly into the glasses mic. Ducking (§4.7) covers it.
- **Interruption** (phone call, Siri, another app grabbing the session): `AVAudioSession.interruptionNotification` ends both the recording tap and Live. Today the recorder handles this; the interviewer must treat it as "end answer window, mark unanswered, don't reconnect mid-interruption". Resume follows the recorder's existing recovery.
- **`paramErr -50` at startup** (documented in `recording-pipeline.md` §2) is a race between HFP negotiation and `setCategory`. `.playAndRecord` + HFP has the same exposure. Keep the glasses ordering rule: DAT stream → `awaitHFPRouteReady` → `startCapture`.

### 4.2 Video: tee, don't duplicate

**Design.**

- New base hook `var usesExternalFrameSource: Bool { false }`. When true, `launchCameraLensBranch()` skips both `IPhoneCameraCapture` creation and the DAT `addStream`, and does **not** fall into `bringUpDisplayOnlyLensSession` (which `streamsCameraToGemini = false` triggers today). The owner pushes JPEGs via a public `submitExternalFrame(_ jpeg: Data, videoTime: CMTime)`.
- `FrameTee`: one object owned by the recording VM, fed from the same callback that feeds the writer (`IPhoneExpertRecordingViewModel.swift:95`, `StreamSessionViewModel.swift:1146`). 1 fps throttle + `isEncodingFrame` re-entry guard + `CIContext.jpegRepresentation` **on its own serial queue**, never on the writer's `recordingQueue` and never on the main actor. This is the existing `IPhoneCoachingCameraSource` pattern minus the capture session.
- The tee emits two things: the JPEG to Live (when the gate is open) and the **latest frame** cached for the planner tick.

**Edge cases.**

- **DeviceSessionManager lease.** The recorder holds `"expert"`; the base will still `acquire("gemini-live")` for the Display capability (lens HUD). Refcounted on the same `DeviceSession` — correct and already supported. Do **not** let the base open a second `Stream`.
- **Writer backpressure.** JPEG encode of a 720×1280 frame is ~10–20 ms. If the tee ever shares a queue with the writer it will show up as `dropped` in the recording stats. Own queue, and skip (don't queue) when busy.
- **Glasses `.medium` is 504×896.** Planner and Live see a lower-res frame than coaching's `.high`. Acceptable for gap detection (tools, parts, large states); not for reading printed sizes — which is precisely what we *ask* about rather than read.
- **Frames before `startSession`.** The writer's timeline starts at the first appended sample (`ExpertRecordingManager.swift:539`). Frames the tee sends before that have no video time — stamp `videoTime = nil`, treat as pre-roll (§4.6).

### 4.3 Mic-to-Live: answer windows, not an open mic

**Problem.** If Live hears the expert's continuous narration, it will reply to it — "Great, looks good!" after every pause. There is no API lever to stop that on 3.1 (`proactivity` is rejected; see `prompt_builder.py:101`), and prompting "stay silent" is exactly the failure mode the coaching `[check]` heartbeat had.

**Design.** The Live mic gate (`recomputeSendGate`, inputs today: muted, pending tool call, Gemini ready, clip playing) gets a fifth input: `answerWindowOpen`. It is **closed by default**. It opens when an `[ASK]` turn is delivered and closes when:

- Live's follow-up turn completes (`turnComplete` after the expert's answer), **or**
- 12 s of mic silence after the question finished playing (`lastAudioActivityAt` — exists, `AudioSessionManager.swift:63`), extended while the expert is still speaking, hard cap 30 s, **or**
- the expert says "later" / "skip" (detected from Live's input transcript inside the window — the only time transcripts exist).

Live is also allowed exactly **one follow-up** per question ("Which size exactly?"). The prompt says so; the window closing enforces it.

**Consequence:** Live's input transcription only covers answers. The planner needs the *narration* too, so narration transcription cannot come from Live. Two options:

| | A. On-device `SFSpeechRecognizer` | B. **Audio chunk to the planner** (recommended) |
|---|---|---|
| How | `requiresOnDeviceRecognition`, continuous, phone posts `transcript_delta` text | Phone keeps a 5 s PCM16 16 kHz ring (already the Live send format) and posts it with the frame; Flash transcribes + reasons in one call and returns the transcript delta for the log |
| Quality on technical vocab | Mediocre ("em six" → "M6"? often not) | Gemini-grade, same model that reads the frame |
| Moving parts | Speech permission, SF quirks, locale, 1-minute session limits on some iOS versions | None new on the phone; one bigger payload (~160 KB/tick) |
| Latency | Live-ish | Tick-bound (~5 s) — fine, the planner is not real-time |
| Failure mode | Silent degradation to text-less planner | Same |

Recommend **B**. Overlap chunks by 1 s so words cut at tick boundaries survive. Fallback to A if payload size ever matters.

**Edge cases.**

- **Expert wants to talk *to* the assistant** ("note that this only applies to the 2019 model"). With a gated mic they can't. Give them a deliberate affordance: a hold-to-talk pill on the HUD (the base already has PTT plumbing: `talkPressBegan/Ended`, currently debug-gated) and, on glasses, the planner can open a window when it transcribes an address phrase ("hey Retrace", "note that") in the chunk. Not a wake word engine — the planner already hears everything.
- **Barge-in during the question.** `serverContent.interrupted` flushes playback (base does this). Mark the question `interrupted`; the planner may offer it once more at the next pause, then defer it.
- **Expert ignores the question.** Window times out, question marked `unanswered`, goes to debrief. Never re-ask live more than once.
- **Answer arrives *outside* the window** (expert answers 20 s later, in narration). The planner hears it in the chunk and marks the gap filled. This is why the planner, not Live, owns the ledger.
- **Reconnect** (`goAway` / socket error → `reconnectWithResumption`). No asks while `isGeminiReady == false`; a question in flight when the socket drops is marked `interrupted`. The resumption inject text for this mode is the ledger summary, not a coaching step summary (`contextSummaryForReconnect` override).

### 4.4 The planner (server)

`services/expert_interview/planner.py`, modelled on `step_oracle.py`: one Flash call, structured output, hard timeout, returns `None` on any failure (= stay silent), `thinking_budget=0`, `asyncio.to_thread`.

**Input per tick** (`POST /api/expert/interview/{id}/observe`, multipart): `frame` (JPEG, optional), `audio` (5 s PCM16 WAV, optional), `video_t` (seconds, may be null in pre-roll), `phase` (`preroll|recording|debrief`), `client_state` (`speaking: bool`, `asks_delivered: int`).

**Server-held ledger** (in the session dict, like `diagnostic_log`):

```python
{
  "briefing": {"task": str|None, "tools_mentioned": [str]},      # from pre-roll
  "transcript": [{"t": float|None, "text": str}],                 # planner-transcribed narration + Live answer transcripts
  "gaps": {topic_key: {"status": "open|asked|answered|deferred|dropped|unanswered",
                       "question": str, "priority": "safety|spec|part|judgment|tip",
                       "first_seen_t": float, "asked_t": float|None, "answer": str|None,
                       "deliver_by_t": float|None}},
  "qa": [{"t_ask": float|None, "question": str, "answer": str|None, "topic_key": str, "outcome": str}],
  "ai_speech_windows": [[t0, t1]],   # mirrored from phone at finalize; phone copy is canonical
  "budget": {"live_asks": int, "last_ask_t": float|None}
}
```

**Output schema:**

```python
class PlannerDecision(BaseModel):
    transcript_delta: str = ""                 # what the expert said in this chunk
    filled: list[str] = []                     # topic_keys the narration just answered
    new_gaps: list[Gap] = []                   # concrete gaps noticed, each with priority
    action: Literal["none", "ask", "defer"] = "none"
    ask: Ask | None = None                     # question text, topic_key, ttl_s, time_sensitive: bool
    confidence: float
```

**Policy is pure Python, not prompt.** The model proposes; `policy.py` decides, deterministically and unit-testable without Gemini:

| Rule | Value | Why |
|---|---|---|
| Ask *live* only if `time_sensitive` | the object is in frame now, the step is about to become irreversible, or `priority == safety` | Everything else is just as good in the debrief and costs no flow |
| Min interval between live asks | 60 s | Measured against the *answer window close*, not the ask |
| Max live asks per session | 6 (configurable) | Beyond that, the expert is being interviewed, not recording |
| Max asks per topic | 1 live + 1 debrief | Never nag |
| Priority order | safety > spec (sizes, torque, amounts, settings) > part identity > judgment ("how do you know it's done") > tip | What a learner is most likely to get wrong without it |
| Confidence floor | 0.7 | Same shape as the oracle's `_ORACLE_CONFIDENCE_FLOOR` |
| Dedup | normalized `topic_key` + fuzzy match on question text against `asked/answered` | Flash *will* rephrase the same gap |
| Quiet mode | expert said "stop asking" → all live asks off, debrief still on | Expert control is non-negotiable |

**Edge cases.**

- **Latency vs scene.** Planner round trip is ~2–4 s; the pause gate may hold the question another 10 s. The socket may be gone by then. Hence `ttl_s` (default 20 s) enforced **on the phone** at delivery time, and `deliver_by_t` for "irreversible step" asks. The Live prompt also says: *"If the thing you're asking about is clearly no longer in view, ask it anyway but as a past-tense question."* Cheaper than trying to make Live stay silent.
- **Empty-scene hallucination.** The same risk the oracle has. Prompt bias: *"A gap must name a specific object, quantity or decision that appears in the frame or the transcript. If you cannot name it, return none. Silence is the default; asking is the exception."* Plus the confidence floor. Plus unit tests on policy with adversarial decisions.
- **Asking about the visible.** The planner must not ask what extraction will capture ("what colour is the lid?"). Explicit *ask-worthy categories* in the prompt: quantities & specs; the reason behind a choice; what "done" feels/sounds/looks like when not visible; what goes wrong and how to recover; alternatives if a tool is missing; safety. Explicit *never-ask*: anything answerable by watching.
- **Duplicate ticks.** Single in-flight call with a re-entry guard; if a tick's result arrives after the next tick started, it still updates the ledger but its `ask` is discarded (stale).
- **Idle scene, no speech.** Skip the model call when the frame's perceptual hash is within threshold of the last one *and* the audio chunk's RMS is below the speech floor. Saves ~half the calls on a real recording.
- **Server restart mid-session.** Session gone → `/observe` 404 → phone stops ticking, logs it, keeps recording, keeps its own `InterviewLog`. Debrief degrades to "no planner" (the phone can still run a fixed 3-question debrief).

### 4.5 Facts: transcript-derived, tools optional

Two ways to get structured facts out of an answer:

1. **Live tool `record_fact(kind, value, part?)`** — structured, immediate; but depends on Live remembering to call it (tool reliability was uncertain enough to warrant `probe_tool_reliability.py`), and every call closes the mic gate until answered (`pendingToolCallIds`).
2. **Transcript-derived** — the planner already has Q (from `[ASK]`) and A (Live's input transcript, forwarded by the phone with the next tick). At `/finalize`, one Flash call turns the Q/A list + transcript into `facts[]` with `{kind, value, part, t, source: "live|debrief|narration"}`.

**Recommend 2 as the source of truth, 1 as a later optimization** — and if 1 is added, make it **client-intercepted** (`interceptToolCall` returns `{"ok": true}` instantly, phone batches to the server), so the mic gate never waits on a network hop.

### 4.6 Phases: pre-roll, recording, debrief — one socket

```
[Pre-record screen]  toggle Interviewer ON ─▶ connect Live + warm audio (parallel with camera prewarm)
     │
     ├─ PRE-ROLL (≤ 20 s, optional, skippable): Live asks ONE briefing question
     │    "What are you about to show, and what tools or parts will you need?"
     │    → ledger.briefing (tools_mentioned seeds the planner: it now knows to ask sizes)
     │
[Record] ──▶ RECORDING: planner ticks, ≤ 6 live asks, answer windows
     │
[Stop]  ──▶ camera off, writer finalizing ─▶ DEBRIEF (same socket, ≤ 90 s)
     │       Live asks deferred + unanswered gaps, highest priority first, in one conversational pass
     │       Expert can say "done" to end
     │
[Review screen] ──▶ upload video + InterviewLog JSON ──▶ extraction
```

**Edge cases.**

- **Recording must never wait for Live.** If Live isn't connected when the user presses Record, record anyway; the interviewer joins when ready (planner only; pre-roll skipped). Connection timeout 8 s (the base's `awaitAudioReady`), then "Interviewer unavailable" badge, no retry loop.
- **Debrief without a timeline.** Debrief facts have `t = null` and `phase = "debrief"`; extraction is told they are procedure-level unless the expert names the step.
- **Stop-confirm countdown** (`ExpertStopOverlayHandler`, 3 s) — the gate must treat the countdown as "not safe to speak"; nothing worse than a question landing as the expert decides whether to stop.
- **Discard after Stop.** The review screen's "Discard" must also discard the interview log and `DELETE` the server session. No orphaned facts.
- **"Keep working in the background"** uploads later from `ExpertTabView`; the interview JSON must be persisted on disk next to the mp4 (`<recording>.interview.json`), not held in a VM that gets torn down.
- **Video picker path** (uploading an existing file, `RecordTabView`): no interview → field omitted → server path identical to today. Backwards compatible by construction.

### 4.7 Keeping the AI's voice out of the learner's clips

The AI's questions are in the mp4's audio track (iPhone speaker → mic; glasses open-ear → mic to a lesser degree). AEC on iPhone removes most of it; on glasses there's no AEC.

**Design.** The phone records `ai_speech_windows: [[t0, t1]]` in **video time** — `t = hostTime(PCM chunk start) − writer.timelineOrigin`. Playback start/stop is observable (`observeAudioIsAISpeaking` exists in the base). These windows go in the `InterviewLog`. Server-side:

- `clip_extractor.extract_clips` gets a new `duck_windows: list[tuple[float, float]]` argument and emits per-clip `-af "volume=enable='between(t,a,b)':volume=0.08"` for the windows that intersect the clip (times rebased to clip start). The command today has no `-af`; this is an additive filter.
- The `_lens` variant gets the same filter.
- The **extraction** Gemini call still sees the raw audio. Its prompt gains: *"An AI assistant asked the expert questions at these times: [...]. Its voice is not the expert's. The expert's answers are provided below as text; prefer the text."*

**Edge cases.**

- `ExpertRecordingManager.firstPTS` is local to `appendVideoFrame`; expose `timelineOrigin: CMTime?` and `videoTime(forHostTime:) -> Double?`.
- Windows before the origin (pre-roll) are dropped; windows after Stop (debrief) are irrelevant to clips.
- Fade: `volume` with `enable` is a hard step. Acceptable for a hackathon; `afade` pairs if the click is audible.
- Don't mute (`volume=0`): a window edge off by 100 ms would cut the first syllable of the expert's answer. 0.08 keeps it recoverable.

### 4.8 Persistence and extraction

**Upload contract** (`routers/expert.py`): `video: UploadFile` (unchanged) + `interview: UploadFile | None = File(None)` (JSON). Store at `data/interviews/{procedure_id}.json`. Add `has_interview INTEGER DEFAULT 0` to `procedures` via the existing swallow-OperationalError ALTER pattern (cheap to filter on; the JSON stays a file, like clips).

**Extraction** (`gemini_processor.process_expert_video`): new keyword `interview: InterviewLog | None`. The prompt gains an `=== EXPERT INTERVIEW NOTES ===` section:

```
Timestamped facts the expert stated when asked (t = seconds into the video; null = said after recording):
- [t=42.1] spec: lug nut socket is 19 mm  (part: lug nut socket)
- [t=118.0] judgment: "tight is when the wrench stops moving under body weight; don't jump on it"
- [t=null] tip: on a 2019+ model the cover pops with a trim tool, not a screwdriver
AI speech windows (ignore this voice; it is not the expert): [[41.0, 44.2], ...]
Rules: fold specs into the step description and the matching parts-glossary `appearance`; fold judgment/
tips/warnings into `tips`/`warnings` of the step whose time range contains t (procedure-level if null);
prefer stated values over anything you infer from the image; never invent a value the expert did not state.
```

Schema unchanged. No step-table migration. `reprocess.process_video_into_procedure` and `reextract_stale_procedures` both load the JSON if present — so **re-extraction keeps the facts**.

**Edge cases.**

- `EXTRACTION_PIPELINE_VERSION` should bump when the prompt section lands, so existing procedures don't silently differ from new ones (they have no interview, so output is the same, but the version string is the contract).
- `evals/eval_extractor.py` calls `process_expert_video` positionally — keep the new argument keyword-only.
- Attribution ("Expert note:" prefix in tips) is a product choice. Start **silent fold-in**; add a `source` tag later if the review UI wants to show provenance.

### 4.9 HUD / UX

Follow `Views/RayBanHUD/DESIGN.md`. Additions, all small:

- **Interviewer pill** on the pre-record screen: `On · Quiet · Off`. Quiet = debrief only.
- **Question card** on the recording page: the question text, auto-dismiss when the window closes. On native lens, text only (bandwidth); rate-limited by the same 60 s budget so the renderer's "re-send on change" is cheap.
- **Answer-window indicator**: a subtle mic-listening state so the expert knows the assistant is listening *now* and not otherwise. This is a trust feature — the expert should never wonder whether it heard something private.
- **Hold-to-talk** pill to open a window voluntarily.
- **Debrief page**: list of open questions with ✓ as they're answered; "Done" ends it.

---

## 5. Failure matrix

| Failure | Recording | Interviewer behaviour |
|---|---|---|
| Live never connects | unaffected | badge "Interviewer unavailable"; planner still ticks; debrief runs as a fixed 3-question text card |
| Live drops mid-recording | unaffected | question in flight → `interrupted`; `reconnectWithResumption`; no asks until `.connected` |
| Planner 5xx / timeout | unaffected | tick skipped; after 3 consecutive failures back off to 15 s |
| Server restarts | unaffected | `/observe` 404 → stop ticking; phone log intact; upload still carries it |
| Audio session `-50` on `.interview` | **fall back to `.recordingPhoneOnly`**, interviewer off | the recording must win |
| Writer sample-rate mismatch | writer re-prepares at first sample | — |
| AVAudioSession interruption | existing recorder recovery | close window, `unanswered`, no reconnect until interruption ends |
| Expert says "stop asking" | — | quiet mode for the session; debrief still offered |
| Expert discards | file deleted | log deleted, server session `DELETE`d |
| Phone app backgrounded | existing behaviour | same as interruption |

---

## 6. Server layout (mirrors `services/troubleshoot/`)

```
routers/expert_interview.py          POST /api/expert/interview/session/start   {voice?, transport}
                                     POST /api/expert/interview/session/{id}/token[?handle=]
                                     POST /api/expert/interview/session/{id}/observe   (multipart)
                                     POST /api/expert/interview/session/{id}/answer    {question_id, transcript}
                                     POST /api/expert/interview/session/{id}/finalize  → InterviewLog
                                     DELETE /api/expert/interview/session/{id}
services/expert_interview/__init__.py   imports handlers (registration side effect)
services/expert_interview/session.py    create_interview_session(), ledger ops, finalize()
services/expert_interview/prompt.py     build_interviewer_prompt(), INTERVIEW_TOOL_DEFINITIONS (may be [])
services/expert_interview/planner.py    observe() → PlannerDecision | None   (Flash, like step_oracle)
services/expert_interview/policy.py     decide(ledger, decision, now) → Ask | None   (pure, tested)
services/expert_interview/facts.py      extract_facts(ledger) → list[Fact]   (Flash, at finalize)
models/interview.py                     InterviewLog, Fact, Ask, PlannerDecision, ...
voice_commands.yaml                     + mode: expert_interview  (later, skip, quiet, done)
```

Changes to existing files: `routers/expert.py` (+`interview` field), `services/reprocess.py` (load JSON), `services/gemini_processor.py` (+prompt section, keyword arg), `services/clip_extractor.py` (+`duck_windows`), `services/database.py` (+`has_interview`), `services/voice_commands.py` (`_MODE_INCLUSIONS` + mode), `config.py` (`INTERVIEWS_DIR`, `RETRACE_PLANNER_MODEL`).

## 7. iOS layout

```
ViewModels/ExpertInterviewViewModel.swift      : GeminiLiveSessionBase  (mintSession, toolCallURLSuffix,
                                                 usesExternalFrameSource=true, audioSessionMode=.interview,
                                                 introSeedText=nil, contextSummaryForReconnect → ledger summary)
ViewModels/InterviewCoordinator.swift          pause gate, budget mirror, answer-window state machine,
                                                 planner ticker, InterviewLog owner, speech-window recorder
Services/InterviewAPIService.swift             start/observe/answer/finalize
Services/FrameTee.swift                        1 fps JPEG tee + latest-frame cache (own queue)
Models/InterviewModels.swift                   Codable mirrors; InterviewLog persisted as <recording>.interview.json
Utils/AudioSessionManager.swift                + .interview mode; secondary-consumer preference when borrowed;
                                                 convert-once fan-out on glasses
Utils/ExpertRecordingManager.swift             + timelineOrigin, videoTime(forHostTime:)
Utils/UploadService.swift                      + optional interview JSON part
ViewModels/GeminiLiveSessionBase.swift         + usesExternalFrameSource hook, answerWindowOpen gate input,
                                                 submitExternalFrame
Views/Expert/HUD/…                             question card, interviewer pill, debrief page (per DESIGN.md)
```

Remember the `pbxproj` registration step for every new Swift file.

## 8. Build order (each phase independently shippable)

0. **Spikes (half a day, before any feature code).** (a) iPhone: recorder borrowing a `.coaching`-mode manager while Live plays audio — does the mp4 record, is the sample rate stable, how much AI voice survives AEC? (b) Glasses: same on `persistentAudio` — call-volume audibility, frame drops with HFP downlink active. These two answers decide whether §4.1 is a mode tweak or a bigger audio refactor.
1. **Server core.** `models/interview.py`, session, `policy.py` **with unit tests** (the only part of this system that is cheap to test exhaustively), planner, observe/finalize, upload field, extraction prompt section, ducking. Testable end-to-end with `curl` and a static JPEG + WAV before any iOS work.
2. **iOS debrief-only** (= option C from the first discussion). Live connects at Stop, asks the planner's deferred gaps, uploads the log. No pause gate, no audio coexistence with the writer. This already demonstrates the thesis.
3. **iOS live asks.** Audio fan-out, frame tee, answer windows, pause gate, question card.
4. **Tuning.** Planner evals from recorded sessions (replay `observe` payloads offline, like `evals/eval_extractor.py`); `gemini_live_playground` scenario for the interviewer prompt; threshold sweep on silence/TTL/min-interval.

## 9. Decisions needed

1. **Narration transcript source** — audio chunk to planner (recommended) vs on-device Speech. Affects payload and phone permissions.
2. **Debrief on the same socket** (recommended; one token, no reconnect) vs a fresh session at Stop. Same-socket means the Live prompt must handle a phase switch via an injected `[DEBRIEF]` turn.
3. **Attribution in the generated SOP** — silent fold-in (recommended for v1) vs "Expert note:" provenance.
4. **Demo transport** — if the hackathon demo is glasses-first, run spike 0(b) before 0(a) and accept no-AEC + ducking as the bleed strategy.
