from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.attachment import Attachment, AttachmentProvider

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


_COMPACTION_MARKER = "Previous context has been compacted"
_CONTINUITY_TYPE = "post_compaction_continuity"
_CONTINUITY_CONTENT = (
    "Context was just compacted. "
    "Review the compaction summary and your todo list above carefully. "
    "Continue working on the current task without asking the user to repeat instructions. "
    "If you need specific details from before compaction, use the RecallCompactedContext tool "
    "with targeted keywords."
)


class PostCompactionContinuityAttachmentProvider(AttachmentProvider):
    """Injects a continuity reminder once after each compaction event.

    Detects compaction via the ``_compaction_generation`` counter on the agent
    loop — each compaction increments the counter, so the provider fires exactly
    once per generation change when the compaction marker is present.
    """
    def __init__(self) -> None:
        self._last_seen_generation: int = 0

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        # Check if history[0] is a compaction summary
        first_msg = history[0]
        if not _is_compaction_summary(first_msg):
            return []

        # Fire exactly once per compaction event.
        gen = agent_loop._compaction_generation  # noqa: SLF001
        if gen == self._last_seen_generation:
            return []

        self._last_seen_generation = gen
        return [
            Attachment(
                type=_CONTINUITY_TYPE,
                content=_CONTINUITY_CONTENT,
                is_hint=False,  # system-reminder (authoritative)
            )
        ]

def _is_compaction_summary(message: Message) -> bool:
    """Check if a message is a compaction summary."""
    for part in message.content:
        if isinstance(part, TextPart) and _COMPACTION_MARKER in part.text:
            return True
    return False
