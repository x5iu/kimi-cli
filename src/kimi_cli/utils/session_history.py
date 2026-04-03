from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from kimi_cli.eventbus.serde import deserialize_bus_message
from kimi_cli.eventbus.types import is_request
from kimi_cli.utils.turns import is_real_user_turn_start_record


def read_bus_lines(event_log: Path) -> list[str]:
    """Read and parse ``events.jsonl`` into JSON-RPC event strings."""
    result: list[str] = []
    with open(event_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                record = cast(dict[str, Any], record)
                record_type = record.get("type")
                if isinstance(record_type, str) and record_type == "metadata":
                    continue
                message_raw = record.get("message")
                if not isinstance(message_raw, dict):
                    continue
                message_raw = cast(dict[str, Any], message_raw)
                message = deserialize_bus_message(message_raw)
                is_req = is_request(message)
                event_msg: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "method": "request" if is_req else "event",
                    "params": message_raw,
                }
                if is_req and (request_id := getattr(message, "id", None)) is not None:
                    event_msg["id"] = request_id
                result.append(json.dumps(event_msg, ensure_ascii=False))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    return result


def truncate_context_at_turn(context_path: Path, turn_index: int) -> list[str]:
    """Return context lines up to and including ``turn_index``."""
    if not context_path.exists():
        return []

    lines: list[str] = []
    current_turn = -1

    with open(context_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record: dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if is_real_user_turn_start_record(record):
                current_turn += 1
                if current_turn > turn_index:
                    break

            if current_turn <= turn_index:
                lines.append(stripped)

    return lines
