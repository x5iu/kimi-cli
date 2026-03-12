from __future__ import annotations

import re
from collections.abc import Sequence

from kosong.message import Message

from kimi_cli.wire.types import AudioURLPart, ContentPart, ImageURLPart, TextPart, VideoURLPart

_MEDIA_WRAPPER_TAG_RE = re.compile(r"</?(?:image|video|audio)(?:\s+[^>]*)?>")


def content_parts_stringify(parts: Sequence[ContentPart]) -> str:
    """Get a compact string representation of content parts for TUI display."""
    rendered: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            if _MEDIA_WRAPPER_TAG_RE.fullmatch(part.text.strip()):
                continue
            rendered.append(part.text)
        elif isinstance(part, ImageURLPart):
            rendered.append("[image]")
        elif isinstance(part, AudioURLPart):
            suffix = f":{part.audio_url.id}" if part.audio_url.id else ""
            rendered.append(f"[audio{suffix}]")
        elif isinstance(part, VideoURLPart):
            rendered.append("[video]")
        else:
            rendered.append(f"[{part.type}]")
    return "".join(rendered)


def message_stringify(message: Message) -> str:
    """Get a compact string representation of a message."""
    if isinstance(message.content, str):
        return message.content
    return content_parts_stringify(message.content)
