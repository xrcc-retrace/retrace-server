import logging
import os
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
from routers import expert, expert_manuals, icons, learner, procedures, troubleshoot
from services.database import init_db
from services.fixtures import schedule_fixtures_load

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


def _get_mdns_hostname() -> str | None:
    """Return the machine's `.local` mDNS hostname (no trailing dot), or None.

    iOS resolves `.local` hostnames through mDNS on every request, so baking
    this into the TXT record lets the phone cache a URL that survives DHCP
    rotation. Returns None for degenerate hostnames so callers fall back to IP.
    """
    raw = socket.gethostname().rstrip(".").strip()
    if not raw or raw.lower() == "localhost":
        return None
    return raw if raw.endswith(".local") else f"{raw}.local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()

    # Background fixture loader — ingests the bundled Nespresso descaling
    # manual the first time the server runs against a fresh DB. Non-blocking
    # so /health and API endpoints come up immediately.
    fixtures_task = schedule_fixtures_load()

    # Bonjour only makes sense on the LAN. In cloud deploys (AWS VPC doesn't
    # support multicast anyway) the iOS app reaches us via a known hostname,
    # so skip the advertisement and the startup/shutdown noise it brings.
    advertise_bonjour = os.getenv("RETRACE_ADVERTISE_BONJOUR", "1") == "1"
    azc: AsyncZeroconf | None = None
    info: ServiceInfo | None = None

    if advertise_bonjour:
        local_ip = _get_local_ip()
        mdns_hostname = _get_mdns_hostname()

        properties: dict[str, str] = {
            "host": local_ip,
            "port": str(SERVICE_PORT),
            "version": "0.1.0",
        }
        if mdns_hostname:
            properties["hostname"] = mdns_hostname

        server_arg = f"{mdns_hostname}." if mdns_hostname else None

        azc = AsyncZeroconf()
        info = ServiceInfo(
            SERVICE_TYPE,
            SERVICE_NAME,
            addresses=[socket.inet_aton(local_ip)],
            port=SERVICE_PORT,
            properties=properties,
            server=server_arg,
        )
        await azc.async_register_service(info)
        logger.info(
            "Bonjour: advertising %s at %s:%d (hostname=%s)",
            SERVICE_NAME,
            local_ip,
            SERVICE_PORT,
            mdns_hostname or "<unavailable>",
        )
    else:
        logger.info("Bonjour: advertisement disabled (RETRACE_ADVERTISE_BONJOUR=0)")

    yield

    # Shutdown
    if not fixtures_task.done():
        fixtures_task.cancel()
    if azc is not None and info is not None:
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

# Order matters: more-specific prefixes first so /api/clips/ etc. match before
# the broader /api/procedures rule.
CACHE_RULES: list[tuple[str, str]] = [
    ("/api/clips/", "public, max-age=86400"),
    ("/api/uploads/", "public, max-age=86400"),
    ("/api/icons/", "public, max-age=86400"),
    ("/api/manual_images/", "public, max-age=86400"),
    ("/static/voice_previews/", "public, max-age=86400, immutable"),
    ("/api/learner/voices", "public, max-age=300"),
    ("/api/procedures", "public, max-age=30"),
]


@app.middleware("http")
async def add_cache_control(request, call_next):
    response = await call_next(request)
    if request.method == "GET" and 200 <= response.status_code < 300:
        path = request.url.path
        for prefix, value in CACHE_RULES:
            if path.startswith(prefix):
                response.headers["Cache-Control"] = value
                break
    return response


app.include_router(expert.router)
app.include_router(expert_manuals.router)
app.include_router(icons.router)
app.include_router(learner.router)
app.include_router(procedures.router)
app.include_router(troubleshoot.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2026-04-28-deploy-test"}


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
