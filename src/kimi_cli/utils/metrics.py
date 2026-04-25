from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from kimi_cli.share import get_share_dir
from kimi_cli.utils.logging import logger


def emit_metric(event: str, **fields: Any) -> None:
    try:
        share = get_share_dir()
        log_dir = share / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(UTC).strftime("%Y%m%d")
        path = log_dir / f"metrics-{day}.jsonl"
        row: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        logger.opt(exception=True).debug("metrics emit failed for {event}", event=event)
