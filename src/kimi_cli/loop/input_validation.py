from __future__ import annotations

from typing import TYPE_CHECKING

from kimi_cli.eventbus.types import ContentPart
from kimi_cli.loop import LLMNotSet, LLMNotSupported
from kimi_cli.loop.message import check_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.llm import LLM


def validate_live_user_input(llm: LLM | None, content: str | list[ContentPart]) -> None:
    """Validate live user input before it is injected into an active turn."""
    if llm is None:
        raise LLMNotSet()

    reminder_message = Message(role="user", content=content)
    if missing_caps := check_message(reminder_message, llm.capabilities):
        raise LLMNotSupported(llm, list(missing_caps))
