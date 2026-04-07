from __future__ import annotations

import json
from unittest.mock import MagicMock

from kimi_cli.loop.attachments.task_poll import TaskPollEscalationAttachmentProvider
from llmkit.message import Message, TextPart, ToolCall


def _make_agent_loop_mock(
    *,
    compaction_generation: int = 0,
    active_turn_id: int = 1,
) -> MagicMock:
    mock = MagicMock()
    mock._compaction_generation = compaction_generation
    mock._active_turn_id = active_turn_id
    return mock


def _user_msg(text: str = "hello") -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _task_output_call(task_id: str, *, call_id: str = "call_1") -> Message:
    """Create an assistant message with a TaskOutput tool call."""
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(
                    name="TaskOutput",
                    arguments=json.dumps({"task_id": task_id, "block": True}),
                ),
            )
        ],
    )


def _tool_result(call_id: str = "call_1", text: str = "timeout") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id=call_id)


def _internal_user_msg(text: str) -> Message:
    """Create an internal user message (not a real turn start)."""
    return Message(role="user", content=[TextPart(text=text)], name="_kimi_internal")


class TestTaskPollEscalationAttachmentProvider:
    async def test_no_injection_below_threshold(self) -> None:
        provider = TaskPollEscalationAttachmentProvider()
        history = [
            _user_msg(),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert result == []

    async def test_injection_at_threshold(self) -> None:
        provider = TaskPollEscalationAttachmentProvider()
        history = [
            _user_msg(),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].is_hint is False
        assert "bash-abc" in result[0].content
        assert "3 times" in result[0].content

    async def test_injection_includes_task_id(self) -> None:
        provider = TaskPollEscalationAttachmentProvider()
        history = [
            _user_msg(),
            _task_output_call("bash-xyz789", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-xyz789", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-xyz789", call_id="c3"),
            _tool_result("c3"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert "bash-xyz789" in result[0].content

    async def test_different_task_ids_tracked_independently(self) -> None:
        provider = TaskPollEscalationAttachmentProvider()
        history = [
            _user_msg(),
            _task_output_call("bash-aaa", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-bbb", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-aaa", call_id="c3"),
            _tool_result("c3"),
            _task_output_call("bash-bbb", call_id="c4"),
            _tool_result("c4"),
            _task_output_call("bash-aaa", call_id="c5"),
            _tool_result("c5"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        # Only bash-aaa has 3 polls; bash-bbb only has 2
        assert len(result) == 1
        assert "bash-aaa" in result[0].content

    async def test_is_not_hint(self) -> None:
        """Attachment must be a system-reminder (is_hint=False), not a hint."""
        provider = TaskPollEscalationAttachmentProvider()
        history = [
            _user_msg(),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].is_hint is False

    async def test_cooldown_same_task_same_turn(self) -> None:
        """Once fired for a task_id, don't fire again for same task_id in same turn."""
        provider = TaskPollEscalationAttachmentProvider()
        mock = _make_agent_loop_mock()
        history = [
            _user_msg(),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]

        # First call triggers
        result1 = await provider.get_attachments(history, mock)
        assert len(result1) == 1

        # Agent keeps polling (4th time)
        history.append(_task_output_call("bash-abc", call_id="c4"))
        history.append(_tool_result("c4"))

        # Second call should NOT trigger again for same task_id
        result2 = await provider.get_attachments(history, mock)
        assert result2 == []

    async def test_turn_boundary_resets_counts(self) -> None:
        """A new turn resets tracking so the same task_id can fire again."""
        provider = TaskPollEscalationAttachmentProvider()
        mock = _make_agent_loop_mock(active_turn_id=1)
        history: list[Message] = [
            _user_msg("first request"),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]

        # Triggers in first turn
        result1 = await provider.get_attachments(history, mock)
        assert len(result1) == 1

        # New turn starts — new user message, new turn_id
        mock._active_turn_id = 2
        history.append(_assistant_msg("Done."))
        history.append(_user_msg("second request"))
        # Only 2 polls in new turn — below threshold
        history.append(_task_output_call("bash-abc", call_id="c4"))
        history.append(_tool_result("c4"))
        history.append(_task_output_call("bash-abc", call_id="c5"))
        history.append(_tool_result("c5"))

        result2 = await provider.get_attachments(history, mock)
        assert result2 == []

        # 3rd poll in new turn — fires again
        history.append(_task_output_call("bash-abc", call_id="c6"))
        history.append(_tool_result("c6"))

        result3 = await provider.get_attachments(history, mock)
        assert len(result3) == 1
        assert "bash-abc" in result3[0].content

    async def test_compaction_does_not_reset_fired_set(self) -> None:
        """Same-turn compaction must not re-fire for already-handled task_ids."""
        provider = TaskPollEscalationAttachmentProvider()

        # --- Pre-compaction: 3 polls → fires ---
        history = [
            _user_msg("do something"),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]
        result1 = await provider.get_attachments(
            history,
            _make_agent_loop_mock(
                compaction_generation=0,
                active_turn_id=1,
            ),
        )
        assert len(result1) == 1

        # --- Post-compaction (same turn): history replaced ---
        compacted_history: list[Message] = [
            _internal_user_msg("[compacted summary]"),
            _user_msg("do something"),
            _task_output_call("bash-abc", call_id="c4"),
            _tool_result("c4"),
            _task_output_call("bash-abc", call_id="c5"),
            _tool_result("c5"),
            _task_output_call("bash-abc", call_id="c6"),
            _tool_result("c6"),
        ]
        result2 = await provider.get_attachments(
            compacted_history,
            _make_agent_loop_mock(
                compaction_generation=1,
                active_turn_id=1,  # same turn
            ),
        )
        # bash-abc was already fired — same-turn compaction preserves
        assert result2 == []

    async def test_between_turn_compaction_resets_fired_set(self) -> None:
        """Between-turn /compact must reset fired set for the new turn."""
        provider = TaskPollEscalationAttachmentProvider()

        # Turn 1: poll bash-abc 3x → fires
        history = [
            _user_msg("turn 1"),
            _task_output_call("bash-abc", call_id="c1"),
            _tool_result("c1"),
            _task_output_call("bash-abc", call_id="c2"),
            _tool_result("c2"),
            _task_output_call("bash-abc", call_id="c3"),
            _tool_result("c3"),
        ]
        result1 = await provider.get_attachments(
            history,
            _make_agent_loop_mock(
                compaction_generation=0,
                active_turn_id=1,
            ),
        )
        assert len(result1) == 1

        # Between-turn /compact: gen bumps, AND new turn starts
        compacted_history: list[Message] = [
            _internal_user_msg("[compacted summary]"),
            _user_msg("turn 2"),
            _task_output_call("bash-abc", call_id="c4"),
            _tool_result("c4"),
            _task_output_call("bash-abc", call_id="c5"),
            _tool_result("c5"),
            _task_output_call("bash-abc", call_id="c6"),
            _tool_result("c6"),
        ]
        result2 = await provider.get_attachments(
            compacted_history,
            _make_agent_loop_mock(
                compaction_generation=1,
                active_turn_id=2,  # new turn!
            ),
        )
        # New turn — must fire again even though gen also changed
        assert len(result2) == 1
        assert "bash-abc" in result2[0].content
