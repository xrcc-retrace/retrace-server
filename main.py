import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from config import UPLOAD_DIR
from routers import expert, learner, procedures
from services.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_retrace._tcp.local."
SERVICE_NAME = "Retrace Server._retrace._tcp.local."
SERVICE_PORT = 8000


def _get_local_ip() -> str:
    """Get the machine's LAN IP via UDP connect trick (no traffic sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()

    local_ip = _get_local_ip()
    azc = AsyncZeroconf()
    info = ServiceInfo(
        SERVICE_TYPE,
        SERVICE_NAME,
        addresses=[socket.inet_aton(local_ip)],
        port=SERVICE_PORT,
        properties={"host": local_ip, "port": str(SERVICE_PORT), "version": "0.1.0"},
    )
    await azc.async_register_service(info)
    logger.info("Bonjour: advertising %s at %s:%d", SERVICE_NAME, local_ip, SERVICE_PORT)

    yield

    # Shutdown
    await azc.async_unregister_service(info)
    await azc.async_close()
    logger.info("Bonjour: service deregistered")


app = FastAPI(title="Retrace SOP API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(expert.router)
app.include_router(learner.router)
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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_dashboard():
    """Serve the SOP review dashboard."""
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
