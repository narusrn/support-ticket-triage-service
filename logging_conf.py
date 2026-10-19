import json
import logging
from pathlib import Path

from config import settings


class JsonFormatter(logging.Formatter):
    """Writes one JSON object per log line. If the log call passes a dict as the
    message (e.g. logger.info({"event": "request_received", ...})), its fields are
    merged in directly -- that's how the structured api/agent/tools logs in PLAN.md
    get their shape without hand-building JSON strings at every call site."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
        }
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload)


def configure_logging() -> None:
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / settings.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    # avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. once by a test, once by the app itself in the same process)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
