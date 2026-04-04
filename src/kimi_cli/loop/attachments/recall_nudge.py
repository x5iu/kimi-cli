from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.loop.attachment import Attachment, AttachmentProvider

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop

_RECALL_NUDGE_TYPE = "recall_nudge_after_compaction"

_COMPACTION_PREFIX = (
    "<system>Previous context has been compacted"
)

# Agent must have done at least this many steps since
# compaction before we fire the first nudge.
_MIN_STEPS_AFTER_COMPACTION = 10

# After firing, wait this many assistant messages before
# firing again.
_COOLDOWN_MESSAGES = 15

# Maximum number of nudges per compaction event.
_MAX_FIRES_PER_COMPACTION = 2

# Tool names that indicate "exploration mode"
_EXPLORATION_TOOLS = frozenset(
    {"Shell", "ReadFile", "Grep"}
)


class RecallNudgeAfterCompactionProvider(AttachmentProvider):
    """Remind the agent to use RecallCompactedContext.

    After compaction, periodically nudges the agent when it
    appears to be searching/exploring and has not used
    ``RecallCompactedContext``.

    Detection criteria (all must be true):
    - History starts with a compaction summary.
    - Agent has made 10+ assistant steps since compaction
      without calling RecallCompactedContext.
    - Recent messages include exploration tools
      (Shell/ReadFile/Grep).

    Cooldown: once per 15 assistant messages, max 2 times
    total per compaction event.

    Uses ``_compaction_generation`` to detect new compaction
    events and reset counters.
    """

    def __init__(
        self,
        *,
        min_steps: int = _MIN_STEPS_AFTER_COMPACTION,
        cooldown: int = _COOLDOWN_MESSAGES,
        max_fires: int = _MAX_FIRES_PER_COMPACTION,
    ) -> None:
        self._min_steps = min_steps
        self._cooldown = cooldown
        self._max_fires = max_fires

        self._last_seen_generation: int = 0
        self._fire_count: int = 0
        self._last_fired_at_step: int = 0

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if not history:
            return []

        gen = agent_loop._compaction_generation  # noqa: SLF001

        # New compaction event → reset counters
        if gen != self._last_seen_generation:
            self._last_seen_generation = gen
            self._fire_count = 0
            self._last_fired_at_step = 0

        # Must be in a compacted state (gen > 0) and
        # first message must be a compaction summary.
        if gen == 0:
            return []
        if not _is_compaction_summary(history[0]):
            return []

        # Max fires per compaction event
        if self._fire_count >= self._max_fires:
            return []

        # Count assistant steps since compaction start and
        # check for RecallCompactedContext usage.
        # NOTE: Only check RecallCompactedContext in messages AFTER
        # the compaction summary to avoid preserved pre-compaction
        # calls suppressing nudges for the new generation.
        assistant_step_count = 0
        has_recall = False
        has_exploration = False
        past_summary = False

        for msg in history:
            # Skip the compaction summary itself
            if not past_summary:
                if _is_compaction_summary(msg):
                    past_summary = True
                    continue
                continue
            if msg.role != "assistant":
                continue
            assistant_step_count += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    if name == "RecallCompactedContext":
                        has_recall = True
                    if name in _EXPLORATION_TOOLS:
                        has_exploration = True

        # Suppress if RecallCompactedContext was already used
        if has_recall:
            return []

        # Need enough steps
        if assistant_step_count < self._min_steps:
            return []

        # Must be in exploration mode
        if not has_exploration:
            return []

        # Cooldown: first fire at min_steps, then every
        # cooldown messages after that.
        if (
            self._last_fired_at_step > 0
            and assistant_step_count
            < self._last_fired_at_step + self._cooldown
        ):
            return []

        self._fire_count += 1
        self._last_fired_at_step = assistant_step_count
        return [
            Attachment(
                type=_RECALL_NUDGE_TYPE,
                content=(
                    "You have compacted archives available. "
                    "If you're looking for information "
                    "discussed earlier in this session, use "
                    "RecallCompactedContext with targeted "
                    "keywords instead of re-searching."
                ),
                is_hint=True,
            )
        ]


def _is_compaction_summary(msg: Message) -> bool:
    """Check if a message is a compaction summary."""
    if msg.role not in ("user", "assistant"):
        return False
    text = msg.extract_text(" ").strip()
    return text.startswith(_COMPACTION_PREFIX)
