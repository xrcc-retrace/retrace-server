"""Shared utilities for the oracle/extractor eval harness.

Import this FIRST from every eval script — its top-level code inserts the
backend root on sys.path and loads .env, so the subsequent `from services...`
imports resolve whether the script is run as `python evals/foo.py` (sys.path[0]
is evals/) or from the repo root.

Nothing here talks to Gemini or Live; it only reads the local library, cuts
frames with ffmpeg, and formats gate output. The 80/50 PASS/MARGINAL/FAIL
convention matches the existing probe scripts.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --- path + .env bootstrap (must run before importing services / config) ----
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_ENV_PATH = _BACKEND_ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# config.py also calls load_dotenv on import, but the manual pass above matches
# the probe convention and guarantees the key is present before services load.
from config import CLIPS_DIR, FFMPEG_PATH, UPLOAD_DIR  # noqa: E402
from services.clip_extractor import extract_done_frame  # noqa: E402
from services.database import get_procedure, init_db, list_procedures  # noqa: E402
from services.video_probe import FFPROBE_PATH  # noqa: E402

# Apply pending schema migrations (idempotent — identical to what the server
# runs on startup). Ensures the visual_verifiability column exists and legacy
# rows are backfilled to 'reliable' before any eval reads them, so the harness
# works against a fresh checkout whose live DB predates the oracle migration.
init_db()

# Eagerly initialize the oracle's module-level genai client, single-threaded,
# before any concurrent verify_step calls. Its _get_client() lazy-init is
# unsynchronized; racing first-calls from asyncio.to_thread workers create
# throwaway Client objects that get finalized (closing httpx) mid-use, surfacing
# as "Cannot send a request, as the client has been closed". One warm call here
# pins _client so every worker reuses it.
from services import step_oracle as _step_oracle  # noqa: E402

try:
    _step_oracle._get_client()
except Exception:  # missing key etc. — evals that need it will surface it
    pass

EVALS_DIR = _BACKEND_ROOT / "evals"
FRAMES_DIR = EVALS_DIR / "fixtures" / "frames"
STAGED_DIR = EVALS_DIR / "fixtures" / "staged"
REPORTS_DIR = EVALS_DIR / "reports"

# Frame labels harvested per step. `done` is the true last frame (byte-identical
# to what the /verify endpoint uses as its reference); the others are sampled at
# fractional positions so a positive test frame is never the exact reference JPEG.
FRAME_FRACTIONS = {"start": 0.0, "mid": 0.5, "near_done": 0.9}


# --- video / frame helpers ---------------------------------------------------

def clip_duration(clip_path: Path) -> float | None:
    """Duration in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def _extract_frame_at(clip_path: Path, t_seconds: float, dst: Path) -> bool:
    """Cut a single JPEG at t_seconds (input-seek, fast). True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-ss", f"{max(0.0, t_seconds):.3f}", "-i", str(clip_path),
             "-frames:v", "1", "-q:v", "2", str(dst)],
            check=True, capture_output=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return dst.exists()


def harvest_step_frames(procedure_id: str, step_number: int, clip_path: Path) -> dict[str, Path]:
    """Extract + cache the labeled frames for one step's clip.

    Returns a label->Path map for whichever frames extracted successfully.
    `done` reuses clip_extractor.extract_done_frame (the oracle's own reference)
    and copies its bytes into the fixtures dir so all frames live in one place
    and the done frame is byte-identical to production. Idempotent.
    """
    out_dir = FRAMES_DIR / procedure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, Path] = {}

    duration = clip_duration(clip_path)
    for label, frac in FRAME_FRACTIONS.items():
        dst = out_dir / f"step_{step_number}_{label}.jpg"
        if dst.exists():
            frames[label] = dst
            continue
        # near_done at 0.9 of duration; fall back to a fixed 0.5s if no duration.
        t = (duration * frac) if duration else (0.5 if frac > 0 else 0.0)
        if _extract_frame_at(clip_path, t, dst):
            frames[label] = dst

    # done: reuse the production last-frame extractor, mirror bytes into fixtures.
    done_dst = out_dir / f"step_{step_number}_done.jpg"
    if done_dst.exists():
        frames["done"] = done_dst
    else:
        src = extract_done_frame(clip_path)
        if src is not None and src.exists():
            done_dst.write_bytes(src.read_bytes())
            frames["done"] = done_dst
    return frames


def staged_step_frames(procedure_id: str, step_number: int) -> dict[str, list[Path]]:
    """Ingest optional human-staged clips/images for higher-fidelity labels.

    Looks in evals/fixtures/staged/{procedure_id}/ for files named
    step_{n}_{label}[_*].{jpg,png,mp4,mov} where label ∈ {done, notdone, errorK}.
    Video files are reduced to their last frame. Returns label->list[Path].
    """
    base = STAGED_DIR / procedure_id
    if not base.exists():
        return {}
    out: dict[str, list[Path]] = {}
    prefix = f"step_{step_number}_"
    for f in sorted(base.iterdir()):
        if not f.name.startswith(prefix):
            continue
        label = f.name[len(prefix):].split("_")[0].split(".")[0]
        if f.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            frame = extract_done_frame(f)
            if frame is None:
                continue
            path = frame
        elif f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            path = f
        else:
            continue
        out.setdefault(label, []).append(path)
    return out


# --- library helpers ---------------------------------------------------------

def clip_path_for(procedure_id: str, step_number: int) -> Path:
    return CLIPS_DIR / procedure_id / f"step_{step_number}.mp4"


def procedures_with_clips() -> list[dict]:
    """Full get_procedure dicts for every procedure that has clips on disk."""
    result = []
    for row in list_procedures():
        proc = get_procedure(row["id"])
        if not proc:
            continue
        has_clip = any(
            clip_path_for(proc["id"], s["step_number"]).exists()
            for s in proc.get("steps", [])
        )
        if has_clip:
            result.append(proc)
    return result


# --- reporting ---------------------------------------------------------------

# thresholds match the existing probes' 80/50 gate
GATE_PASS = 0.80
GATE_MARGINAL = 0.50

_C = {"pass": "\033[92m", "marginal": "\033[93m", "fail": "\033[91m",
      "dim": "\033[90m", "bold": "\033[1m", "reset": "\033[0m"}


def gate_label(rate: float) -> str:
    if rate >= GATE_PASS:
        return "PASS"
    if rate >= GATE_MARGINAL:
        return "MARGINAL"
    return "FAIL"


def print_gate(name: str, rate: float, detail: str = "") -> str:
    label = gate_label(rate)
    color = {"PASS": _C["pass"], "MARGINAL": _C["marginal"], "FAIL": _C["fail"]}[label]
    tail = f"  {_C['dim']}{detail}{_C['reset']}" if detail else ""
    print(f"  {color}{label:8}{_C['reset']} {name}: {_C['bold']}{rate*100:5.1f}%{_C['reset']}{tail}")
    return label


def header(text: str) -> None:
    print(f"\n{_C['bold']}{'='*68}\n{text}\n{'='*68}{_C['reset']}")


class EventLog:
    """Append-only JSONL sink, mirroring the probe scripts' logging."""

    def __init__(self, name: str):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = REPORTS_DIR / f"{name}_{int(time.time())}.jsonl"
        self._fh = self.path.open("w")

    def write(self, **event) -> None:
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
