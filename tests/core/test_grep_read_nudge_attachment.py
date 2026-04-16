from __future__ import annotations

import json
from unittest.mock import MagicMock

from kimi_cli.loop.attachments.grep_read_nudge import (
    GrepThenTargetedReadNudgeProvider,
)
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


def _tool_call_msg(
    tool_name: str,
    *,
    call_id: str = "c1",
    arguments: str = "{}",
) -> Message:
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(name=tool_name, arguments=arguments),
            )
        ],
    )


def _shell_call(command: str, *, call_id: str = "c1") -> Message:
    """Shell tool call with a specific command string."""
    return _tool_call_msg(
        "Shell",
        call_id=call_id,
        arguments=json.dumps({"command": command}),
    )



def _tool_result(call_id: str = "c1", text: str = "ok") -> Message:
    return Message(
        role="tool",
        content=[TextPart(text=text)],
        tool_call_id=call_id,
    )


class TestGrepThenTargetedReadNudgeProvider:
    async def test_no_injection_without_grep(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _tool_call_msg("ReadFile", call_id="c1"),
            _tool_result("c1"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_injection_after_shell_grep(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _shell_call("grep -rn TODO .", call_id="c1"),
            _tool_result("c1", "file.py:10:# TODO"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert "ReadFile" in result[0].content
        assert "line_offset" in result[0].content

    async def test_injection_after_shell_rg(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _shell_call("rg TODO .", call_id="c1"),
            _tool_result("c1", "file.py:10:# TODO"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert "ReadFile" in result[0].content

    async def test_injection_after_shell_rg_variant(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _shell_call("/opt/homebrew/bin/rg pattern .", call_id="c1"),
            _tool_result("c1", "file.py:10:match"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert "ReadFile" in result[0].content

    async def test_is_hint_true(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _shell_call("rg TODO .", call_id="c1"),
            _tool_result("c1"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert result[0].is_hint is True

    async def test_cooldown_once_per_turn(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        mock = _make_agent_loop_mock()
        history: list[Message] = [
            _user_msg(),
            _shell_call("rg TODO .", call_id="c1"),
            _tool_result("c1"),
        ]
        # First call fires
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # Another rg in same turn — suppressed
        history.append(_shell_call("rg FIXME .", call_id="c2"))
        history.append(_tool_result("c2"))
        r2 = await provider.get_attachments(history, mock)
        assert r2 == []

    async def test_turn_reset(self) -> None:
        provider = GrepThenTargetedReadNudgeProvider()
        mock = _make_agent_loop_mock()
        history: list[Message] = [
            _user_msg("first"),
            _shell_call("rg TODO .", call_id="c1"),
            _tool_result("c1"),
        ]
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # New turn
        mock._active_turn_id = 2
        history.append(_assistant_msg("Done."))
        history.append(_user_msg("second"))
        history.append(_shell_call("rg FIXME .", call_id="c2"))
        history.append(_tool_result("c2"))

        r2 = await provider.get_attachments(history, mock)
        assert len(r2) == 1

    async def test_shell_rg_triggers(self) -> None:
        """Shell with 'rg ' in command triggers the nudge."""
        provider = GrepThenTargetedReadNudgeProvider()
        history = [
            _user_msg(),
            _shell_call("rg --type py import", call_id="c1"),
            _tool_result("c1"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1

