import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import UPLOAD_DIR
from routers import expert, procedures
from services.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Retrace SOP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(expert.router)
app.include_router(procedures.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded source videos for review playback."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Upload not found")
    media_type = "video/quicktime" if filename.endswith(".mov") else "video/mp4"
    return FileResponse(str(file_path), media_type=media_type)


@app.get("/")
async def serve_dashboard():
    """Serve the SOP review dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
