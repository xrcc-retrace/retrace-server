"""On-disk store for uploaded InterviewLogs: data/interviews/{procedure_id}.json.

The log is persisted *next to the video* rather than passed transiently into
the first extraction because extraction is re-run in place on startup for
every stale video procedure (services/reprocess.py) — facts handed only to
the first run would silently vanish on the next pipeline-version bump.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import INTERVIEWS_DIR
from models.interview import InterviewLog

logger = logging.getLogger(__name__)


def path_for(procedure_id: str) -> Path:
    return INTERVIEWS_DIR / f"{procedure_id}.json"


def save_log(procedure_id: str, log: InterviewLog) -> Path:
    path = path_for(procedure_id)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(log.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("interview log saved: %s (%d facts, %d qa)", path.name, len(log.facts), len(log.qa))
    return path


def load_log(procedure_id: str) -> InterviewLog | None:
    path = path_for(procedure_id)
    if not path.exists():
        return None
    try:
        return InterviewLog.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("interview log unreadable for %s: %s", procedure_id, exc)
        return None


def load_log_for_video(video_path: Path) -> InterviewLog | None:
    """Uploads are named {procedure_id}.{ext}; the lens-variant refresher only
    has the video path."""
    stem = video_path.stem
    if stem.endswith("_lens"):
        stem = stem[: -len("_lens")]
    return load_log(stem)


def delete_log(procedure_id: str) -> None:
    path_for(procedure_id).unlink(missing_ok=True)


def duck_windows(log: InterviewLog | None) -> list[tuple[float, float]]:
    if log is None:
        return []
    return [(float(a), float(b)) for a, b in log.ai_speech_windows if len((a, b)) == 2 and b > a]


def parse_upload(raw: bytes) -> InterviewLog:
    """Validate a phone-uploaded log. Raises ValueError on bad JSON/shape."""
    try:
        return InterviewLog.model_validate_json(raw)
    except Exception as exc:  # pydantic ValidationError or JSON error
        raise ValueError(str(exc)) from exc


def merge_server_view(phone: InterviewLog, server: InterviewLog | None) -> InterviewLog:
    """Phone copy wins on everything it carries; the server view only
    contributes facts/transcript the phone didn't have (e.g. finalize ran
    after the phone snapshotted)."""
    if server is None:
        return phone
    if not phone.facts and server.facts:
        phone.facts = server.facts
    if len(server.transcript) > len(phone.transcript):
        phone.transcript = server.transcript
    if not phone.qa and server.qa:
        phone.qa = server.qa
    return phone


def _json_safe(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
