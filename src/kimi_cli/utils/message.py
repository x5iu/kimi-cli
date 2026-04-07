from __future__ import annotations

import re
from collections.abc import Sequence

from kimi_cli.eventbus.types import AudioURLPart, ContentPart, ImageURLPart, TextPart, VideoURLPart
from llmkit.message import Message

_MEDIA_OPEN_TAG_RE = re.compile(r"<(image|video|audio)(?:\s+[^>]*)?>")
_MEDIA_CLOSE_TAG_RE = re.compile(r"</(image|video|audio)>")


def _stringify_content_part(part: ContentPart) -> str:
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, ImageURLPart):
        return "[image]"
    if isinstance(part, AudioURLPart):
        suffix = f":{part.audio_url.id}" if part.audio_url.id else ""
        return f"[audio{suffix}]"
    if isinstance(part, VideoURLPart):
        return "[video]"
    return f"[{part.type}]"


def _match_media_wrapper(parts: Sequence[ContentPart], index: int) -> tuple[str, int] | None:
    if index + 2 >= len(parts):
        return None
    open_part = parts[index]
    media_part = parts[index + 1]
    close_part = parts[index + 2]
    if not isinstance(open_part, TextPart) or not isinstance(close_part, TextPart):
        return None
    open_match = _MEDIA_OPEN_TAG_RE.fullmatch(open_part.text.strip())
    close_match = _MEDIA_CLOSE_TAG_RE.fullmatch(close_part.text.strip())
    if open_match is None or close_match is None:
        return None
    tag = open_match.group(1)
    if close_match.group(1) != tag:
        return None
    if tag == "image" and isinstance(media_part, ImageURLPart):
        return (_stringify_content_part(media_part), index + 3)
    if tag == "audio" and isinstance(media_part, AudioURLPart):
        return (_stringify_content_part(media_part), index + 3)
    if tag == "video" and isinstance(media_part, VideoURLPart):
        return (_stringify_content_part(media_part), index + 3)
    return None


def content_parts_stringify(parts: Sequence[ContentPart]) -> str:
    """Get a compact string representation of content parts for TUI display."""
    rendered: list[str] = []
    index = 0
    while index < len(parts):
        wrapped_media = _match_media_wrapper(parts, index)
        if wrapped_media is not None:
            text, index = wrapped_media
            rendered.append(text)
            continue
        rendered.append(_stringify_content_part(parts[index]))
        index += 1
    return "".join(rendered)


def message_stringify(message: Message) -> str:
    """Get a compact string representation of a message."""
    if isinstance(message.content, str):
        return message.content
    return content_parts_stringify(message.content)
