from __future__ import annotations

from unittest.mock import MagicMock

from llmkit.message import Message, TextPart, ToolCall

from kimi_cli.loop.attachments.tool_storm import ToolStormBreakerAttachmentProvider


def _make_agent_loop_mock(*, compaction_generation: int = 0) -> MagicMock:
    mock = MagicMock()
    mock._compaction_generation = compaction_generation
    return mock


def _user_msg(text: str = "hello") -> Message:
    return Message(role="user", content=[TextPart(text=text)])


def _assistant_msg(text: str = "ok") -> Message:
    return Message(role="assistant", content=[TextPart(text=text)])


def _tool_call_msg(tool_name: str, *, call_id: str = "c1") -> Message:
    """Create an assistant message with a single tool call."""
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(name=tool_name, arguments="{}"),
            )
        ],
    )


def _parallel_tool_call_msg(tool_names: list[str], *, call_id_prefix: str = "p") -> Message:
    """Create an assistant message with multiple parallel tool calls."""
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id=f"{call_id_prefix}{i}",
                function=ToolCall.FunctionBody(name=name, arguments="{}"),
            )
            for i, name in enumerate(tool_names)
        ],
    )


def _tool_result(call_id: str = "c1", text: str = "ok") -> Message:
    return Message(role="tool", content=[TextPart(text=text)], tool_call_id=call_id)


def _shell_streak(count: int, *, start: int = 1) -> list[Message]:
    """Build a streak of N single Shell calls with interleaved tool results."""
    msgs: list[Message] = []
    for i in range(start, start + count):
        msgs.append(_tool_call_msg("Shell", call_id=f"c{i}"))
        msgs.append(_tool_result(f"c{i}"))
    return msgs


class TestToolStormBreakerAttachmentProvider:
    async def test_no_injection_below_threshold(self) -> None:
        provider = ToolStormBreakerAttachmentProvider(threshold=8)
        history: list[Message] = [_user_msg()]
        history.extend(_shell_streak(7))

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert result == []

    async def test_injection_at_threshold(self) -> None:
        provider = ToolStormBreakerAttachmentProvider(threshold=8)
        history: list[Message] = [_user_msg()]
        history.extend(_shell_streak(8))

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].is_hint is True
        assert "8 consecutive Shell calls" in result[0].content
        assert "Synthesize" in result[0].content

    async def test_is_hint_true(self) -> None:
        """Attachment must be a system-hint (is_hint=True), not a reminder."""
        provider = ToolStormBreakerAttachmentProvider(threshold=3)
        history: list[Message] = [_user_msg()]
        history.extend(_shell_streak(3))

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        assert len(result) == 1
        assert result[0].is_hint is True

    async def test_cooldown_suppresses_until_next_interval(self) -> None:
        """After firing at threshold, don't fire again until cooldown more calls."""
        provider = ToolStormBreakerAttachmentProvider(threshold=4, cooldown=4)
        mock = _make_agent_loop_mock()
        history: list[Message] = [_user_msg()]
        history.extend(_shell_streak(4))

        # First call fires at 4
        result1 = await provider.get_attachments(history, mock)
        assert len(result1) == 1
        assert "4 consecutive" in result1[0].content

        # Grow streak to 6 — within cooldown window
        history.extend(_shell_streak(2, start=5))

        result2 = await provider.get_attachments(history, mock)
        assert result2 == []

        # Grow streak to 8 — fires again (4 + 4)
        history.extend(_shell_streak(2, start=7))

        result3 = await provider.get_attachments(history, mock)
        assert len(result3) == 1
        assert "8 consecutive" in result3[0].content

    async def test_turn_boundary_resets(self) -> None:
        """A new turn resets all tracking."""
        provider = ToolStormBreakerAttachmentProvider(threshold=4, cooldown=4)
        mock = _make_agent_loop_mock()
        history: list[Message] = [_user_msg("first")]
        history.extend(_shell_streak(4))

        # Fires in first turn
        result1 = await provider.get_attachments(history, mock)
        assert len(result1) == 1

        # New turn
        history.append(_assistant_msg("Done."))
        history.append(_user_msg("second"))
        history.extend(_shell_streak(4, start=20))

        # Fires again — turn reset cleared cooldown
        result2 = await provider.get_attachments(history, mock)
        assert len(result2) == 1

    async def test_mixed_tools_break_streak(self) -> None:
        """A different tool in the middle resets the consecutive count."""
        provider = ToolStormBreakerAttachmentProvider(threshold=4)
        history: list[Message] = [
            _user_msg(),
            # 3 Shell
            _tool_call_msg("Shell", call_id="c1"),
            _tool_result("c1"),
            _tool_call_msg("Shell", call_id="c2"),
            _tool_result("c2"),
            _tool_call_msg("Shell", call_id="c3"),
            _tool_result("c3"),
            # 1 ReadFile breaks the streak
            _tool_call_msg("ReadFile", call_id="c4"),
            _tool_result("c4"),
            # 3 Shell — new streak, only 3 long
            _tool_call_msg("Shell", call_id="c5"),
            _tool_result("c5"),
            _tool_call_msg("Shell", call_id="c6"),
            _tool_result("c6"),
            _tool_call_msg("Shell", call_id="c7"),
            _tool_result("c7"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        # Streak is 3 (after ReadFile), below threshold of 4
        assert result == []

    async def test_parallel_same_tool_counts_each_call(self) -> None:
        """Parallel calls to the same tool each count toward the streak."""
        provider = ToolStormBreakerAttachmentProvider(threshold=5)
        history: list[Message] = [
            _user_msg(),
            # 3 parallel Shell calls in one message
            _parallel_tool_call_msg(["Shell", "Shell", "Shell"], call_id_prefix="p"),
            _tool_result("p0"),
            _tool_result("p1"),
            _tool_result("p2"),
            # 2 more single Shell calls
            _tool_call_msg("Shell", call_id="c1"),
            _tool_result("c1"),
            _tool_call_msg("Shell", call_id="c2"),
            _tool_result("c2"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        # 3 + 2 = 5 consecutive Shell calls, hits threshold
        assert len(result) == 1
        assert "5 consecutive Shell" in result[0].content

    async def test_parallel_mixed_tools_break_streak(self) -> None:
        """An assistant message with mixed parallel tools breaks the streak."""
        provider = ToolStormBreakerAttachmentProvider(threshold=4)
        history: list[Message] = [
            _user_msg(),
            # 3 single Shell
            _tool_call_msg("Shell", call_id="c1"),
            _tool_result("c1"),
            _tool_call_msg("Shell", call_id="c2"),
            _tool_result("c2"),
            _tool_call_msg("Shell", call_id="c3"),
            _tool_result("c3"),
            # Mixed parallel: Shell + ReadFile — breaks streak
            _parallel_tool_call_msg(["Shell", "ReadFile"], call_id_prefix="m"),
            _tool_result("m0"),
            _tool_result("m1"),
            # 2 Shell after the break
            _tool_call_msg("Shell", call_id="c4"),
            _tool_result("c4"),
            _tool_call_msg("Shell", call_id="c5"),
            _tool_result("c5"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        # Streak is 2 (after the mixed break), below threshold of 4
        assert result == []

    async def test_text_assistant_message_breaks_streak(self) -> None:
        """An assistant text response (no tool_calls) breaks the streak."""
        provider = ToolStormBreakerAttachmentProvider(threshold=4)
        history: list[Message] = [
            _user_msg(),
            _tool_call_msg("Shell", call_id="c1"),
            _tool_result("c1"),
            _tool_call_msg("Shell", call_id="c2"),
            _tool_result("c2"),
            _tool_call_msg("Shell", call_id="c3"),
            _tool_result("c3"),
            # Text response breaks streak
            _assistant_msg("Let me think about this..."),
            # 2 Shell after the break
            _tool_call_msg("Shell", call_id="c4"),
            _tool_result("c4"),
            _tool_call_msg("Shell", call_id="c5"),
            _tool_result("c5"),
        ]

        result = await provider.get_attachments(history, _make_agent_loop_mock())

        # Streak is 2 (after text break), below threshold of 4
        assert result == []
