from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Thresholds for injection
_HINT_THRESHOLD = 0.60  # inject a <system-hint> at 60%
_REMINDER_THRESHOLD = 0.80  # escalate to <system-reminder> at 80%

# Don't inject more than once every N assistant messages
_COOLDOWN_ASSISTANT_MESSAGES = 5

_CONTEXT_BUDGET_TYPE = "context_budget"


class ContextBudgetAttachmentProvider(AttachmentProvider):
    """Warns the agent when context budget usage is high.

    - At 60% usage: injects a ``<system-hint>`` suggesting conciseness.
    - At 80% usage: escalates to ``<system-reminder>`` with stronger guidance.
    Self-throttles by checking history for a recent context_budget attachment.
    """

    def __init__(
        self,
        *,
        hint_threshold: float = _HINT_THRESHOLD,
        reminder_threshold: float = _REMINDER_THRESHOLD,
        cooldown: int = _COOLDOWN_ASSISTANT_MESSAGES,
    ) -> None:
        self._hint_threshold = hint_threshold
        self._reminder_threshold = reminder_threshold
        self._cooldown = cooldown
        self._last_inject_index: int = -1  # history length at last injection
        self._last_seen_generation: int = 0

    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        usage = agent_loop._context_usage  # noqa: SLF001
        if usage < self._hint_threshold:
            return []

        # Cooldown: count assistant messages since last injection.
        # Reset if compaction happened (generation changed).
        gen = agent_loop._compaction_generation  # noqa: SLF001
        if gen != self._last_seen_generation:
            self._last_seen_generation = gen
            self._last_inject_index = -1
        if self._last_inject_index >= 0:
            assistant_since = sum(
                1 for msg in history[self._last_inject_index :] if msg.role == "assistant"
            )
            if assistant_since < self._cooldown:
                return []
        pct = int(usage * 100)
        is_critical = usage >= self._reminder_threshold

        if is_critical:
            content = (
                f"Context budget is at {pct}% — nearing the limit. "
                "Minimize all tool output. "
                "Finish the current task or summarize progress."
            )
        else:
            content = (
                f"Context budget is at {pct}%. "
                "Prefer smaller, targeted tool calls — use line ranges, "
                "filters, and output limits to keep results compact."
            )

        self._last_inject_index = len(history)
        return [
            Attachment(
                type=_CONTEXT_BUDGET_TYPE,
                content=content,
                is_hint=not is_critical,  # hint at 60%, reminder at 80%
            )
        ]
