from __future__ import annotations

from typing import TYPE_CHECKING

from kosong.message import Message

from kimi_cli.soul import LLMNotSet, LLMNotSupported
from kimi_cli.soul.message import check_message
from kimi_cli.wire.types import ContentPart

if TYPE_CHECKING:
    from kimi_cli.llm import LLM


def validate_live_user_input(llm: LLM | None, content: str | list[ContentPart]) -> None:
    """Validate live user input before it is injected into an active turn."""
    if llm is None:
        raise LLMNotSet()

    reminder_message = Message(role="user", content=content)
    if missing_caps := check_message(reminder_message, llm.capabilities):
        raise LLMNotSupported(llm, list(missing_caps))
