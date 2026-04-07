from __future__ import annotations

import json
from unittest.mock import MagicMock

from kimi_cli.loop.attachments.edit_verify import (
    EditVerificationReminderProvider,
)
from llmkit.message import Message, TextPart, ToolCall


def _make_agent_loop_mock(*, compaction_generation: int = 0) -> MagicMock:
    mock = MagicMock()
    mock._compaction_generation = compaction_generation
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


def _edit_call(file_path: str, *, call_id: str = "c1") -> Message:
    return _tool_call_msg(
        "Edit",
        call_id=call_id,
        arguments=json.dumps(
            {
                "path": file_path,
                "old_string": "x",
                "new_string": "y",
            }
        ),
    )


def _write_call(file_path: str, *, call_id: str = "c1") -> Message:
    return _tool_call_msg(
        "WriteFile",
        call_id=call_id,
        arguments=json.dumps({"path": file_path, "content": "x"}),
    )


def _shell_call(command: str, *, call_id: str = "c1") -> Message:
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


class TestEditVerificationReminderProvider:
    async def test_no_injection_with_1_edit(self) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/foo.py", call_id="c1"),
            _tool_result("c1"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_injection_with_2_edits(self) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/foo.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/bar.py", call_id="c2"),
            _tool_result("c2"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert "2 source files" in result[0].content

    async def test_no_injection_if_verified_pytest(
        self,
    ) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/foo.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/bar.py", call_id="c2"),
            _tool_result("c2"),
            _shell_call("pytest tests/", call_id="c3"),
            _tool_result("c3"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_no_injection_if_verified_ruff(
        self,
    ) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
            _shell_call("ruff check .", call_id="c3"),
            _tool_result("c3"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_no_injection_if_verified_make_test(
        self,
    ) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
            _shell_call("make test", call_id="c3"),
            _tool_result("c3"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_no_injection_if_verified_uv_run_pytest(
        self,
    ) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
            _shell_call("uv run pytest tests/", call_id="c3"),
            _tool_result("c3"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_excludes_non_source_files(
        self,
    ) -> None:
        """md, txt, json, yaml etc. don't count."""
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("README.md", call_id="c1"),
            _tool_result("c1"),
            _edit_call("config.json", call_id="c2"),
            _tool_result("c2"),
            _edit_call("notes.txt", call_id="c3"),
            _tool_result("c3"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_excludes_tmp_files(self) -> None:
        """/tmp/* files don't count."""
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("/tmp/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("/tmp/b.py", call_id="c2"),
            _tool_result("c2"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert result == []

    async def test_is_hint_true(self) -> None:
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert result[0].is_hint is True

    async def test_cooldown_once_per_turn(self) -> None:
        provider = EditVerificationReminderProvider()
        mock = _make_agent_loop_mock()
        history: list[Message] = [
            _user_msg(),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
        ]
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # More edits in same turn — suppressed
        history.append(_edit_call("src/c.py", call_id="c3"))
        history.append(_tool_result("c3"))
        r2 = await provider.get_attachments(history, mock)
        assert r2 == []

    async def test_turn_reset(self) -> None:
        provider = EditVerificationReminderProvider()
        mock = _make_agent_loop_mock()
        history: list[Message] = [
            _user_msg("first"),
            _edit_call("src/a.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/b.py", call_id="c2"),
            _tool_result("c2"),
        ]
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # New turn
        history.append(_assistant_msg("Done."))
        history.append(_user_msg("second"))
        history.append(_edit_call("src/x.py", call_id="c3"))
        history.append(_tool_result("c3"))
        history.append(_edit_call("src/y.py", call_id="c4"))
        history.append(_tool_result("c4"))

        r2 = await provider.get_attachments(history, mock)
        assert len(r2) == 1

    async def test_write_file_counts(self) -> None:
        """WriteFile calls count the same as Edit."""
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _write_call("src/new.py", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/old.py", call_id="c2"),
            _tool_result("c2"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
        assert "2 source files" in result[0].content

    async def test_various_source_extensions(
        self,
    ) -> None:
        """Multiple source extensions are recognized."""
        provider = EditVerificationReminderProvider()
        history = [
            _user_msg(),
            _edit_call("src/main.go", call_id="c1"),
            _tool_result("c1"),
            _edit_call("src/lib.rs", call_id="c2"),
            _tool_result("c2"),
        ]
        result = await provider.get_attachments(history, _make_agent_loop_mock())
        assert len(result) == 1
