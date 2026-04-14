# Retrace — SOP Generation Server

Backend server for Retrace. Receives expert-recorded videos, processes them through Gemini 2.5 Pro to generate structured Standard Operating Procedures (SOPs), and extracts reference keyframes for each step.

## Quick Start

### Prerequisites

- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- [Gemini API key](https://aistudio.google.com/apikey)

### Setup

```bash
cd server
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

Uploads an expert recording, processes it through Gemini 2.5 Pro, extracts keyframes, and returns a structured procedure. Takes 30-120 seconds depending on video length.

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
      "keyframe_url": "/api/keyframes/uuid/step_1.jpg"
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

### Get Keyframe

```
GET /api/keyframes/{procedure_id}/step_{n}.jpg
```

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

# Download a keyframe
curl http://localhost:8000/api/keyframes/{id}/step_1.jpg -o keyframe.jpg
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
│   └── procedures.py           # GET endpoints for procedures + keyframes
├── services/
│   ├── gemini_processor.py     # Gemini 2.5 Pro video → structured JSON
│   └── keyframe_extractor.py   # ffmpeg keyframe extraction
├── data/
│   ├── procedures/             # Generated procedure JSON files
│   └── keyframes/              # Extracted JPEG keyframes
└── uploads/                    # Uploaded expert videos
```

## How It Works

1. Expert records a task wearing Meta Ray-Ban glasses (video + narration)
2. iOS app uploads the MP4 to this server
3. Server uploads the video to Gemini's Files API
4. Gemini 2.5 Pro analyzes both visual and audio content, returns a structured procedure with timestamps
5. Server extracts a JPEG keyframe at the midpoint of each step using ffmpeg
6. Procedure JSON + keyframes are stored on disk and returned to the client
