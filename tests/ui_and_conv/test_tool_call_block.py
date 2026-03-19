from __future__ import annotations

from io import StringIO

from kosong.tooling import BriefDisplayBlock, ToolError, ToolReturnValue
from rich.console import Console

from kimi_cli.ui.shell.visualize import MAX_TOOL_ERROR_OUTPUT_LINES, _ToolCallBlock
from kimi_cli.wire.types import (
    DiffDisplayBlock,
    TodoDisplayBlock,
    TodoDisplayItem,
    ToolCall,
    ToolResult,
)


def _render_to_str(block: _ToolCallBlock) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(block.compose())
    return buf.getvalue()


class TestExtractFullUrl:
    """Tests for _ToolCallBlock._extract_full_url static method."""

    def test_fetchurl_normal_url(self):
        url = _ToolCallBlock._extract_full_url(
            '{"url": "https://example.com/very/long/path"}', "FetchURL"
        )
        assert url == "https://example.com/very/long/path"

    def test_fetchurl_short_url(self):
        url = _ToolCallBlock._extract_full_url('{"url": "https://x.co"}', "FetchURL")
        assert url == "https://x.co"

    def test_non_fetchurl_tool(self):
        url = _ToolCallBlock._extract_full_url('{"url": "https://example.com"}', "ReadFile")
        assert url is None

    def test_arguments_none(self):
        url = _ToolCallBlock._extract_full_url(None, "FetchURL")
        assert url is None

    def test_invalid_json(self):
        url = _ToolCallBlock._extract_full_url("not json", "FetchURL")
        assert url is None

    def test_missing_url_field(self):
        url = _ToolCallBlock._extract_full_url('{"query": "hello"}', "FetchURL")
        assert url is None

    def test_empty_string(self):
        url = _ToolCallBlock._extract_full_url("", "FetchURL")
        assert url is None


class TestHeadlineRendering:
    def test_renders_full_shell_command_without_truncation(self):
        long_command = "echo " + "x" * 80
        block = _ToolCallBlock(
            ToolCall(
                id="call_1",
                function=ToolCall.FunctionBody(
                    name="Shell",
                    arguments=f'{{"command": "{long_command}"}}',
                ),
            )
        )

        rendered = _render_to_str(block)

        assert long_command in rendered
        assert "..." not in rendered

    def test_append_args_part_returns_false_when_headline_is_unchanged(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_2",
                function=ToolCall.FunctionBody(name="Shell", arguments=None),
            )
        )

        changed = block.append_args_part('{"co')

        assert changed is False

    def test_append_args_part_returns_true_when_key_argument_becomes_available(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_3",
                function=ToolCall.FunctionBody(name="Shell", arguments=None),
            )
        )

        changed = block.append_args_part('{"command":"echo ok"}')

        assert changed is True

    def test_renders_set_todo_list_headline_with_ready_summary(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_4",
                function=ToolCall.FunctionBody(
                    name="SetTodoList",
                    arguments=(
                        '{"todos":[{"title":"Inspect parser","status":"pending",'
                        '"executor":"task","subagent_name":"coder"},'
                        '{"title":"Share findings","status":"pending"}]}'
                    ),
                ),
            )
        )

        rendered = _render_to_str(block)

        assert "Updating Todo List (2 todos; ready: Inspect parser @coder)" in rendered

    def test_set_todo_list_status_text_prioritizes_ready_todo(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_4_status",
                function=ToolCall.FunctionBody(
                    name="SetTodoList",
                    arguments=(
                        '{"todos":[{"title":"Inspect parser","status":"pending",'
                        '"executor":"task","subagent_name":"coder"},'
                        '{"title":"Share findings","status":"pending"}]}'
                    ),
                ),
            )
        )

        assert block.status_text == "Updating Todo List (ready: Inspect parser @coder)"

    def test_renders_execute_todo_headline_with_title(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_5",
                function=ToolCall.FunctionBody(
                    name="ExecuteTodo",
                    arguments='{"title":"Inspect parser","subagent_name":"coder"}',
                ),
            )
        )

        rendered = _render_to_str(block)

        assert "Executing Todo (Inspect parser @coder)" in rendered


class TestErrorRendering:
    def test_renders_error_message_and_output(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_1",
                function=ToolCall.FunctionBody(
                    name="Shell",
                    arguments='{"command": "ls /missing"}',
                ),
            )
        )
        block.finish(
            ToolError(
                message="Command failed with exit code: 1.",
                brief="Failed with exit code: 1",
                output="ls: /missing: No such file or directory\n",
            )
        )

        rendered = _render_to_str(block)

        assert "Command failed with exit code: 1." in rendered
        assert "ls: /missing: No such file or directory" in rendered

    def test_renders_detailed_error_message_instead_of_generic_brief(self):
        block = _ToolCallBlock(
            ToolCall(
                id="call_1",
                function=ToolCall.FunctionBody(
                    name="SearchWeb",
                    arguments='{"query": "kimi cli"}',
                ),
            )
        )
        block.finish(
            ToolError(
                message=(
                    "Failed to search. Status: 503. "
                    "This may indicates that the search service is currently unavailable."
                ),
                brief="Failed to search",
            )
        )

        rendered = _render_to_str(block)

        assert "Status: 503" in rendered
        assert "service is currently unavailable" in rendered

    def test_truncates_long_error_output_preview(self):
        output = "\n".join(f"line {idx}" for idx in range(MAX_TOOL_ERROR_OUTPUT_LINES + 5))
        block = _ToolCallBlock(
            ToolCall(
                id="call_1",
                function=ToolCall.FunctionBody(name="Shell", arguments='{"command": "bad"}'),
            )
        )
        block.finish(ToolError(message="boom", brief="boom", output=output))

        rendered = _render_to_str(block)

        assert "line 0" in rendered
        assert f"line {MAX_TOOL_ERROR_OUTPUT_LINES - 1}" in rendered
        assert f"line {MAX_TOOL_ERROR_OUTPUT_LINES}" not in rendered
        assert "[...truncated]" in rendered


def test_renders_diff_display_for_file_edit_result() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_write",
            function=ToolCall.FunctionBody(
                name="WriteFile",
                arguments='{"path": "src/example.py", "content": "after"}',
            ),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=False,
            output="",
            message="File successfully overwritten.",
            display=[
                DiffDisplayBlock(
                    path="src/example.py",
                    old_text="before",
                    new_text="after",
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "File successfully overwritten." in rendered
    assert "src/example.py" in rendered
    assert "@@ -1 +1 @@" in rendered
    assert "-before" in rendered
    assert "+after" in rendered


def test_renders_diff_display_for_subagent_file_edit_result() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="task_1",
            function=ToolCall.FunctionBody(
                name="Task",
                arguments='{"description": "edit file", "subagent_name": "coder", "prompt": "..."}',
            ),
        )
    )
    sub_call = ToolCall(
        id="call_edit",
        function=ToolCall.FunctionBody(
            name="Edit",
            arguments='{"path": "src/example.py", "edit": {"kind": "replace", "old": "before", "new": "after"}}',
        ),
    )
    block.append_sub_tool_call(sub_call)
    block.finish_sub_tool_call(
        ToolResult(
            tool_call_id="call_edit",
            return_value=ToolReturnValue(
                is_error=False,
                output="",
                message="File successfully edited.",
                display=[
                    DiffDisplayBlock(
                        path="src/example.py",
                        old_text="before",
                        new_text="after",
                        old_start_line=42,
                        new_start_line=42,
                    )
                ],
            ),
        )
    )
    block.finish(ToolReturnValue(is_error=False, output="", message="done", display=[]))

    rendered = _render_to_str(block)

    assert "Used Edit (src/example.py)" in rendered
    assert "File successfully edited." in rendered
    assert "src/example.py" in rendered
    assert "@@ -42 +42 @@" in rendered
    assert "42    │ -before" in rendered
    assert "42 │ +after" in rendered


def test_renders_line_numbers_for_top_level_edit_diff_display() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_edit",
            function=ToolCall.FunctionBody(
                name="Edit",
                arguments='{"path": "src/example.py", "edit": {"kind": "replace", "old": "before", "new": "after"}}',
            ),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=False,
            output="",
            message="File successfully edited.",
            display=[
                DiffDisplayBlock(
                    path="src/example.py",
                    old_text="before",
                    new_text="after",
                    old_start_line=42,
                    new_start_line=42,
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "@@ -42 +42 @@" in rendered
    assert "42    │ -before" in rendered
    assert "42 │ +after" in rendered


def test_renders_ready_to_execute_todo_hint() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_todo",
            function=ToolCall.FunctionBody(name="SetTodoList", arguments='{"todos": []}'),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=False,
            output="",
            message="Todo list updated",
            display=[
                TodoDisplayBlock(
                    items=[
                        TodoDisplayItem(
                            title="Inspect parser",
                            status="pending",
                            executor="task",
                            subagent_name="coder",
                        ),
                        TodoDisplayItem(title="Share findings", status="pending", executor="main"),
                    ]
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "Ready to ExecuteTodo: Inspect parser @coder" in rendered


def test_omits_ready_to_execute_todo_hint_when_multiple_candidates_exist() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_todo_many",
            function=ToolCall.FunctionBody(name="SetTodoList", arguments='{"todos": []}'),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=False,
            output="",
            message="Todo list updated",
            display=[
                TodoDisplayBlock(
                    items=[
                        TodoDisplayItem(
                            title="Inspect parser",
                            status="pending",
                            executor="task",
                            subagent_name="coder",
                        ),
                        TodoDisplayItem(
                            title="Inspect lexer",
                            status="pending",
                            executor="task",
                            subagent_name="coder",
                        ),
                    ]
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "Ready to ExecuteTodo:" not in rendered


def test_renders_completed_todo_brief_for_execute_todo_result() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_exec_todo",
            function=ToolCall.FunctionBody(
                name="ExecuteTodo",
                arguments='{"title":"Inspect parser","subagent_name":"coder"}',
            ),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=False,
            output="",
            message='Todo "Inspect parser" completed via Task.',
            display=[
                BriefDisplayBlock(text="Completed Todo: Inspect parser"),
                TodoDisplayBlock(
                    items=[
                        TodoDisplayItem(
                            title="Inspect parser",
                            status="done",
                            executor="task",
                            subagent_name="coder",
                        )
                    ]
                ),
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "Executed Todo (Inspect parser @coder)" in rendered
    assert "Completed Todo: Inspect parser" in rendered


def test_renders_blocked_todo_error_for_execute_todo_result() -> None:
    block = _ToolCallBlock(
        ToolCall(
            id="call_exec_todo_fail",
            function=ToolCall.FunctionBody(
                name="ExecuteTodo",
                arguments='{"title":"Fix build","subagent_name":"coder"}',
            ),
        )
    )
    block.finish(
        ToolReturnValue(
            is_error=True,
            output=(
                "1 todos, blocked=1\n"
                "Next state: blocked\n"
                "Suggested next step: inspect [Task output], revise the prompt, or update the todo state before retrying.\n\n"
                "[Task output]\n"
                "Task failed\n\n"
                "Subagent failed"
            ),
            message="Blocked Todo: Fix build. Delegated Task failed.",
            display=[
                TodoDisplayBlock(
                    items=[
                        TodoDisplayItem(
                            title="Fix build",
                            status="blocked",
                            executor="task",
                            subagent_name="coder",
                        )
                    ]
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "Executed Todo (Fix build @coder)" in rendered
    assert "Blocked Todo: Fix build. Delegated Task failed." in rendered
    assert "Next state: blocked" in rendered
    assert "Fix build" in rendered
    assert "(blocked)" in rendered
