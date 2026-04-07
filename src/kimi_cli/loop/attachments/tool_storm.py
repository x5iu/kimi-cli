from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Fire after this many consecutive same-tool calls in a turn
_STORM_THRESHOLD = 8

# After injection, suppress for this many more same-tool calls
_STORM_COOLDOWN = 8

_TOOL_STORM_TYPE = "tool_storm_breaker"


class ToolStormBreakerAttachmentProvider(AttachmentProvider):
    """Interrupts long runs of consecutive calls to the same tool.

    When the agent makes ``threshold`` or more consecutive calls to the
    same tool within a turn, injects a ``<system-hint>`` suggesting it
    pause to synthesize findings or batch operations.

    Each individual ``tool_call`` within an assistant message is counted
    separately.  An assistant message with parallel tool calls counts as
    same-tool only when **all** calls target the same tool; mixed-tool
    parallel calls break the streak.

    Self-throttles: after injection, suppresses for another ``cooldown``
    same-tool calls before re-injecting.  Resets fully on turn change
    or compaction (compaction removes the tool-call sequence from history,
    so the streak is effectively broken).
    """

    def __init__(
        self,
        *,
        threshold: int = _STORM_THRESHOLD,
        cooldown: int = _STORM_COOLDOWN,
    ) -> None:
        self._threshold = threshold
        self._cooldown = cooldown
        self._last_turn_start_index: int = -1
        self._last_streak_start_index: int = -1
        self._last_fired_at_count: int = 0
        self._last_seen_generation: int = 0

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        # Find the current turn boundary (walk backward)
        turn_start: int = 0
        for i in range(len(history) - 1, -1, -1):
            if is_real_user_turn_start_message(history[i]):
                turn_start = i
                break

        # Compaction replaces history (removing the tool-call sequence),
        # and turn change means a new user request.  Both require a full
        # reset of streak tracking.
        gen = agent_loop._compaction_generation  # noqa: SLF001
        if gen != self._last_seen_generation or turn_start != self._last_turn_start_index:
            self._last_seen_generation = gen
            self._last_turn_start_index = turn_start
            self._last_streak_start_index = -1
            self._last_fired_at_count = 0

        # Walk backward to measure the current consecutive same-tool streak
        streak_tool: str | None = None
        streak_count: int = 0
        streak_first_index: int = len(history)

        for i in range(len(history) - 1, turn_start - 1, -1):
            msg = history[i]
            if msg.role != "assistant":
                continue  # skip tool results, user/system messages
            if not msg.tool_calls:
                break  # text-only assistant response breaks the streak

            tool_names = {tc.function.name for tc in msg.tool_calls}
            if len(tool_names) != 1:
                break  # mixed parallel tools break the streak

            tool_name = next(iter(tool_names))
            if streak_tool is None:
                streak_tool = tool_name
            elif tool_name != streak_tool:
                break  # different tool breaks the streak

            streak_count += len(msg.tool_calls)
            streak_first_index = i

        if streak_tool is None or streak_count < self._threshold:
            return []

        # Detect new streak vs continuation of the same one
        if streak_first_index != self._last_streak_start_index:
            self._last_streak_start_index = streak_first_index
            self._last_fired_at_count = 0

        # Cooldown: first fire at threshold; then every cooldown calls after
        if (
            self._last_fired_at_count > 0
            and streak_count < self._last_fired_at_count + self._cooldown
        ):
            return []

        self._last_fired_at_count = streak_count
        return [
            Attachment(
                type=_TOOL_STORM_TYPE,
                content=(
                    f"You've made {streak_count} consecutive {streak_tool} calls. Consider:\n"
                    "- Synthesize what you've learned so far before continuing\n"
                    "- Batch multiple operations into a single call "
                    "(e.g., chain Shell commands with &&)\n"
                    "- If searching, narrow your query instead of broadening it"
                ),
                is_hint=True,  # non-binding suggestion
            )
        ]
