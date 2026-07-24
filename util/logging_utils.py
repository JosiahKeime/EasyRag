import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path


SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "token",
    "authorization",
    "secret",
    "password",
    "access_token",
    "refresh_token",
    "client_secret",
}

API_KEY_PATTERN = re.compile(
    r"(?i)\b("
    r"sk-[A-Za-z0-9]{10,}"
    r"|AIza[0-9A-Za-z\-_]{20,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|Bearer\s+[A-Za-z0-9._\-]+"
    r")\b"
)


def _is_sensitive_key(key):
    if not isinstance(key, str):
        return False
    lowered = key.lower().replace("-", "_")
    return any(name in lowered for name in SENSITIVE_KEY_NAMES)


def redact_sensitive_data(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_key(key) else redact_sensitive_data(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]

    if isinstance(value, str):
        return API_KEY_PATTERN.sub("[REDACTED]", value)

    return value


class RedactingJsonHandler(logging.Handler):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(exist_ok=True)

    def emit(self, record):
        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }

            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)

            with self.file_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(redact_sensitive_data(payload), ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


def configure_logging():
    logger = logging.getLogger("easy_rag")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = Path(__file__).resolve().parent.parent / "log"
    log_dir.mkdir(exist_ok=True)

    text_log_path = log_dir / "easy_rag.log"
    json_log_path = log_dir / "easy_rag.jsonl"

    file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    json_handler = RedactingJsonHandler(json_log_path)
    json_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(json_handler)

    logger.info("Logging initialized | text_log=%s | json_log=%s", text_log_path, json_log_path)
    return logger


logger = configure_logging()


def _serialize_messages(messages_to_log):
    serialized = []
    for message in messages_to_log:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        serialized.append(
            {
                "type": getattr(message, "type", None),
                "content": content,
            }
        )
    return serialized
