from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Default: start injecting after this many assistant messages in the current turn
_DEFAULT_ACTIVATE_AFTER = 10
# Default: inject every N assistant messages after activation
_DEFAULT_INJECT_EVERY = 10
# Max characters to include from the original user request
_MAX_GOAL_LENGTH = 300


class GoalTrackingAttachmentProvider(AttachmentProvider):
    """Periodically reminds the agent of the original user request during long turns.

    Activates after ``activate_after`` assistant messages in the current turn,
    then injects a goal-tracking reminder every ``inject_every`` assistant messages.
    """

    def __init__(
        self,
        *,
        activate_after: int = _DEFAULT_ACTIVATE_AFTER,
        inject_every: int = _DEFAULT_INJECT_EVERY,
    ) -> None:
        self._activate_after = activate_after
        self._inject_every = inject_every

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        # Find the last real user turn-start message (the current turn's request)
        goal_text: str | None = None
        goal_index: int = -1
        for i in range(len(history) - 1, -1, -1):
            if is_real_user_turn_start_message(history[i]):
                goal_text = _extract_text(history[i])
                goal_index = i
                break

        if goal_text is None or goal_index < 0:
            return []

        # Count assistant messages after the turn-start message
        assistant_count = sum(1 for msg in history[goal_index + 1 :] if msg.role == "assistant")

        if assistant_count < self._activate_after:
            return []

        # Only inject at multiples of inject_every (after activate_after)
        steps_since_activation = assistant_count - self._activate_after
        if steps_since_activation % self._inject_every != 0:
            return []

        # Truncate long goals
        truncated = goal_text
        if len(truncated) > _MAX_GOAL_LENGTH:
            truncated = truncated[:_MAX_GOAL_LENGTH] + "..."

        return [
            Attachment(
                type="goal_tracking",
                content=(
                    f"You are now {assistant_count} steps into this turn. "
                    f"Stay focused on the original request:\n\n{truncated}"
                ),
                is_hint=False,  # system-reminder (authoritative)
            )
        ]


def _extract_text(message: Message) -> str | None:
    """Extract the concatenated text content from a message."""
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextPart):
            parts.append(part.text)
    return " ".join(parts).strip() if parts else None
