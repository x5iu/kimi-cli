from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kimi_cli.notifications import is_notification_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


@dataclass(frozen=True, slots=True)
class Attachment:
    """A dynamic prompt content to be injected before an LLM step."""

    type: str  # identifier for the attachment type
    content: str  # text content (will be wrapped in <system-reminder> or <system-hint> tags)
    is_hint: bool = False  # True = <system-hint>, False = <system-reminder>


class AttachmentProvider(ABC):
    """Base class for attachment providers.

    Called before each LLM step. Implementations handle their own throttling.
    Providers can access all runtime state via the ``agent_loop`` parameter
    (context_usage, runtime, config, etc.).
    """

    @abstractmethod
    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]: ...


def normalize_history(history: Sequence[Message]) -> list[Message]:
    """Merge adjacent user messages to produce a clean API input sequence.

    Attachments are stored as standalone user messages in history;
    normalization merges them into the adjacent user message.

    Only ``user`` role messages are merged. Assistant and tool messages
    are never merged because their ``tool_calls`` / ``tool_call_id``
    fields form linked pairs that must stay intact.
    """
    if not history:
        return []

    result: list[Message] = []
    for msg in history:
        if (
            result
            and result[-1].role == msg.role
            and msg.role == "user"
            and not is_notification_message(result[-1])
            and not is_notification_message(msg)
        ):
            merged_content = list(result[-1].content) + list(msg.content)
            result[-1] = Message(role="user", content=merged_content)
        else:
            result.append(msg)
    return result


class IncrementalHistoryNormalizer:
    """Incrementally normalizes history by tracking the last merge point."""

    def __init__(self) -> None:
        self._last_input_len: int = 0
        self._last_tail_id: int | None = None  # id() of last message
        self._cached_result: list[Message] = []

    def normalize(self, history: Sequence[Message]) -> list[Message]:
        input_len = len(history)
        tail_id = id(history[-1]) if history else None

        if input_len == self._last_input_len and tail_id == self._last_tail_id:
            return self._cached_result

        if (
            input_len > self._last_input_len
            and self._cached_result
            and self._last_tail_id is not None
            # Verify the previous tail is still in the expected position
            and input_len > 0
            and self._last_input_len > 0
            and id(history[self._last_input_len - 1]) == self._last_tail_id
        ):
            # Incremental: only process new messages
            result = list(self._cached_result)
            for msg in history[self._last_input_len :]:
                if (
                    result
                    and result[-1].role == msg.role
                    and msg.role == "user"
                    and not is_notification_message(result[-1])
                    and not is_notification_message(msg)
                ):
                    merged_content = list(result[-1].content) + list(msg.content)
                    result[-1] = Message(role="user", content=merged_content)
                else:
                    result.append(msg)
        else:
            # Full reprocess (history was reset/compacted/replaced)
            result = normalize_history(history)

        self._last_input_len = input_len
        self._last_tail_id = id(result[-1]) if result else None
        self._cached_result = result
        return result
