from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit.message import Message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


# Fire after this many TaskOutput polls to the same task_id in a turn
_POLL_THRESHOLD = 3

_TASK_POLL_TYPE = "task_poll_escalation"


class TaskPollEscalationAttachmentProvider(AttachmentProvider):
    """Interrupts repeated TaskOutput polling of the same background task.

    When the agent polls the same task_id 3+ times in the current turn,
    injects a ``<system-reminder>`` telling it to stop polling and rely
    on the automatic completion notification instead.

    Fires at most once per task_id per turn.  Resets when a new turn starts.
    """

    def __init__(self, *, threshold: int = _POLL_THRESHOLD) -> None:
        self._threshold = threshold
        self._fired_task_ids: set[str] = set()
        self._last_turn_id: int | None = None
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

        # Use _active_turn_id (monotonic, compaction-immune) for turn
        # detection.  Only preserve _fired_task_ids on same-turn compaction.
        turn_id = agent_loop._active_turn_id  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        gen = agent_loop._compaction_generation  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if turn_id != self._last_turn_id:
            # New turn — always reset (covers between-turn /compact too)
            self._last_turn_id = turn_id
            self._last_seen_generation = gen
            self._fired_task_ids = set()
        elif gen != self._last_seen_generation:
            # Same turn, auto-compaction — preserve fired set
            self._last_seen_generation = gen

        # Count TaskOutput calls per task_id in the current turn
        poll_counts: dict[str, int] = {}
        for msg in history[turn_start:]:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function.name != "TaskOutput":
                    continue
                task_id = _extract_task_id(tc.function.arguments)
                if task_id is not None:
                    poll_counts[task_id] = poll_counts.get(task_id, 0) + 1

        # Find task_ids that crossed threshold and haven't been fired yet
        attachments: list[Attachment] = []
        for task_id, count in poll_counts.items():
            if count >= self._threshold and task_id not in self._fired_task_ids:
                # Skip escalation for interactive tasks — polling is the
                # expected workflow since they never reach terminal status
                # and automatic completion notifications do not apply.
                view = agent_loop._runtime.background_tasks.get_task(task_id)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                if view is not None and view.spec.interactive:
                    continue
                self._fired_task_ids.add(task_id)
                attachments.append(
                    Attachment(
                        type=_TASK_POLL_TYPE,
                        content=(
                            f"You've polled task {task_id} {count} times and it's still running. "
                            "Stop polling — rely on the automatic completion notification instead. "
                            "Move on to other work or inform the user the task is long-running."
                        ),
                        is_hint=False,  # system-reminder (authoritative)
                    )
                )
        return attachments


def _extract_task_id(arguments: str | None) -> str | None:
    """Extract task_id from a TaskOutput tool call's arguments JSON."""
    if not arguments:
        return None
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(args, dict):
        d = cast(dict[str, Any], args)
        task_id: str | None = d.get("task_id")
        if isinstance(task_id, str):
            return task_id
    return None
