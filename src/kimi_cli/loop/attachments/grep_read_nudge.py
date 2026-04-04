from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from llmkit.message import Message

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.utils.turns import is_real_user_turn_start_message

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop

_GREP_READ_NUDGE_TYPE = "grep_read_nudge"

# How many recent assistant messages (from the end) to scan
_SCAN_WINDOW = 3


class GrepThenTargetedReadNudgeProvider(AttachmentProvider):
    """Nudge the agent to use targeted ReadFile after grep/rg.

    After detecting Shell calls containing ``rg `` or ``grep ``,
    or Grep tool calls within the last few assistant messages,
    injects a ``<system-hint>`` suggesting the agent read only
    the relevant sections instead of whole files.

    Fires at most once per turn.  Resets on turn change
    (stateless re-derive means compaction is safe without
    extra tracking).
    """

    def __init__(
        self, *, scan_window: int = _SCAN_WINDOW
    ) -> None:
        self._scan_window = scan_window
        self._fired_this_turn: bool = False
        self._last_turn_id: int | None = None

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

        # Reset on turn change
        turn_id = agent_loop._active_turn_id  # noqa: SLF001
        if turn_id != self._last_turn_id:
            self._last_turn_id = turn_id
            self._fired_this_turn = False

        # Cooldown: once per turn
        if self._fired_this_turn:
            return []

        # Collect last N assistant messages in this turn
        assistant_msgs: list[Message] = []
        for msg in reversed(history[turn_start:]):
            if msg.role == "assistant":
                assistant_msgs.append(msg)
                if len(assistant_msgs) >= self._scan_window:
                    break

        if not assistant_msgs:
            return []

        # Check for grep/rg in those messages
        if not _has_grep_calls(assistant_msgs):
            return []

        self._fired_this_turn = True
        return [
            Attachment(
                type=_GREP_READ_NUDGE_TYPE,
                content=(
                    "Your recent search returned line-numbered "
                    "results. When reading those files, use "
                    "ReadFile with line_offset and n_lines to "
                    "read only the relevant sections "
                    "(\u00b120 lines around matches) instead "
                    "of the whole file."
                ),
                is_hint=True,
            )
        ]


def _has_grep_calls(
    messages: Sequence[Message],
) -> bool:
    """Return True if any message has a grep/rg tool call."""
    for msg in messages:
        if not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            name = tc.function.name
            if name == "Grep":
                return True
            if name == "Shell" and _shell_args_contain_grep(
                tc.function.arguments,
            ):
                return True
    return False


def _shell_args_contain_grep(
    arguments: str | None,
) -> bool:
    """Check if Shell arguments contain rg or grep."""
    if not arguments:
        return False
    try:
        args = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(args, dict):
        cmd = args.get("command", "")
        if not isinstance(cmd, str):
            return False
        return "rg " in cmd or "grep " in cmd
    return False
