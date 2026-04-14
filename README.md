# Retrace — SOP Generation Server

Backend server for Retrace. Receives expert-recorded videos, processes them through Gemini 2.5 Pro to generate structured Standard Operating Procedures (SOPs), and extracts video clips for each step.

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- [Gemini API key](https://aistudio.google.com/apikey)

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
GEMINI_API_KEY=your_key_here python main.py
```

Server starts on `http://0.0.0.0:8000`. Connect from your phone using your Mac's local IP (e.g., `http://192.168.x.x:8000`).

## API Endpoints

### Health Check

```
GET /health
```

### Upload Expert Video

```
POST /api/expert/upload
Content-Type: multipart/form-data
Field: video (MP4 file)
```

Uploads an expert recording, processes it through Gemini 2.5 Pro, clips each step, and returns a structured procedure. Takes 30-120 seconds depending on video length.

**Response:**

```json
{
  "id": "uuid",
  "title": "Clearing a Grinder Blockage",
  "description": "Step-by-step procedure for...",
  "steps": [
    {
      "step_number": 1,
      "title": "Power off the machine",
      "description": "Locate the main power switch on the right side...",
      "timestamp_start": 0.0,
      "timestamp_end": 8.5,
      "tips": ["Use the main switch, not standby"],
      "warnings": ["Never open panels while powered on"],
      "clip_url": "/api/clips/uuid/step_1.mp4"
    }
  ],
  "total_duration": 180.0,
  "source_video": "uploads/uuid.mp4",
  "created_at": "2026-04-14T12:00:00Z"
}
```

### List Procedures

```
GET /api/procedures
```

### Get Procedure

```
GET /api/procedures/{procedure_id}
```

### Get Step Clip

```
GET /api/clips/{procedure_id}/step_{n}.mp4
```

Returns the video clip of the expert performing step N. Clips are H.264+AAC MP4 with `faststart` for streaming playback.

## Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Upload a video
curl -X POST -F "video=@my_recording.mp4" http://localhost:8000/api/expert/upload

# List all procedures
curl http://localhost:8000/api/procedures

# Get a specific procedure
curl http://localhost:8000/api/procedures/{id}

# Download a step clip
curl http://localhost:8000/api/clips/{id}/step_1.mp4 -o step_1.mp4
```

## Project Structure

```
server/
├── main.py                     # FastAPI app entry point
├── config.py                   # Paths, API keys, ffmpeg location
├── requirements.txt
├── models/
│   └── procedure.py            # Pydantic models + Gemini output schema
├── routers/
│   ├── expert.py               # POST /api/expert/upload
│   └── procedures.py           # GET endpoints for procedures + clips
├── services/
│   ├── gemini_processor.py     # Gemini 2.5 Pro video → structured JSON
│   └── clip_extractor.py       # ffmpeg step clip extraction
├── data/
│   ├── procedures/             # Generated procedure JSON files
│   └── clips/                  # Extracted step video clips
└── uploads/                    # Uploaded expert videos
```

## How It Works

1. Expert records a task wearing Meta Ray-Ban glasses (video + narration)
2. iOS app uploads the MP4 to this server
3. Server uploads the video to Gemini's Files API
4. Gemini 2.5 Pro analyzes both visual and audio content, returns a structured procedure with timestamps
5. Server clips each step from the source video using ffmpeg (H.264+AAC, faststart-enabled)
6. Procedure JSON + clips are stored on disk and returned to the client

Each step clip can be played back on the learner's glasses display when they need to see how the expert performed that specific step.
