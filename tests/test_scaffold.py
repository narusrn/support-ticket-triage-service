import json
import logging
from pathlib import Path

from config import settings
from logging_conf import configure_logging


def test_settings_load_with_defaults():
    assert settings.OPENAI_MODEL == "gpt-4o-mini"
    assert settings.MAX_STEPS == 12
    assert 0 <= settings.AUTO_RESPOND_CONFIDENCE_THRESHOLD <= 1


def test_app_boots_and_logs_one_line(tmp_path, monkeypatch):
    # point LOG_DIR at a throwaway tmp dir so this test never touches the real logs/ folder
    monkeypatch.setattr(settings, "LOG_DIR", str(tmp_path))
    configure_logging()

    logger = logging.getLogger("scaffold_smoke_test")
    logger.info({"event": "app_boot", "note": "scaffold smoke test"})

    log_file = Path(tmp_path) / settings.LOG_FILE
    assert log_file.exists()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["event"] == "app_boot"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed
