from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from kosong.message import Message

from kimi_cli.wire.types import TextPart

CHECKPOINT_USER_PATTERN = re.compile(r"^<system>CHECKPOINT \d+</system>$")
_INTERNAL_USER_PREFIXES = ("<system>", "<system-reminder>")


def _first_text_part_message(message: Message) -> str | None:
    for part in message.content:
        if isinstance(part, TextPart):
            return part.text
    return None


def _first_text_part_record(record: Mapping[str, Any]) -> str | None:
    content = record.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str):
            return text
    return None


def is_checkpoint_user_text(text: str) -> bool:
    return CHECKPOINT_USER_PATTERN.fullmatch(text.strip()) is not None


def is_internal_user_text(text: str) -> bool:
    return text.strip().startswith(_INTERNAL_USER_PREFIXES)


def is_internal_user_message(message: Message) -> bool:
    if message.role != "user":
        return False
    first_text = _first_text_part_message(message)
    return first_text is not None and is_internal_user_text(first_text)


def is_internal_user_record(record: Mapping[str, Any]) -> bool:
    if record.get("role") != "user":
        return False
    first_text = _first_text_part_record(record)
    return first_text is not None and is_internal_user_text(first_text)


def is_real_user_turn_start_message(message: Message) -> bool:
    return message.role == "user" and not is_internal_user_message(message)


def is_real_user_turn_start_record(record: Mapping[str, Any]) -> bool:
    return record.get("role") == "user" and not is_internal_user_record(record)
