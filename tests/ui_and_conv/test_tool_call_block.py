from __future__ import annotations

from io import StringIO

from rich.console import Console

from kimi_cli.eventbus.types import (
    DiffDisplayBlock,
    TodoDisplayBlock,
    TodoDisplayItem,
    ToolCall,
)
from kimi_cli.ui.shell.blocks import MAX_TOOL_ERROR_OUTPUT_LINES, ToolCallBlock
from llmkit.tooling import ToolError, ToolReturnValue


def _render_to_str(block: ToolCallBlock) -> str:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(block.compose())
    return buf.getvalue()


class TestExtractFullUrl:
    """Tests for ToolCallBlock._extract_full_url static method."""

    def test_fetchurl_normal_url(self):
        url = ToolCallBlock._extract_full_url(
            '{"url": "https://example.com/very/long/path"}', "FetchURL"
        )
        assert url == "https://example.com/very/long/path"

    def test_fetchurl_short_url(self):
        url = ToolCallBlock._extract_full_url('{"url": "https://x.co"}', "FetchURL")
        assert url == "https://x.co"

    def test_non_fetchurl_tool(self):
        url = ToolCallBlock._extract_full_url('{"url": "https://example.com"}', "ReadFile")
        assert url is None

    def test_arguments_none(self):
        url = ToolCallBlock._extract_full_url(None, "FetchURL")
        assert url is None

    def test_invalid_json(self):
        url = ToolCallBlock._extract_full_url("not json", "FetchURL")
        assert url is None

    def test_missing_url_field(self):
        url = ToolCallBlock._extract_full_url('{"query": "hello"}', "FetchURL")
        assert url is None

    def test_empty_string(self):
        url = ToolCallBlock._extract_full_url("", "FetchURL")
        assert url is None


class TestHeadlineRendering:
    def test_renders_full_shell_command_without_truncation(self):
        long_command = "echo " + "x" * 80
        block = ToolCallBlock(
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
        block = ToolCallBlock(
            ToolCall(
                id="call_2",
                function=ToolCall.FunctionBody(name="Shell", arguments=None),
            )
        )

        changed = block.append_args_part('{"co')

        assert changed is False

    def test_append_args_part_returns_true_when_key_argument_becomes_available(self):
        block = ToolCallBlock(
            ToolCall(
                id="call_3",
                function=ToolCall.FunctionBody(name="Shell", arguments=None),
            )
        )

        changed = block.append_args_part('{"command":"echo ok"}')

        assert changed is True

    def test_renders_set_todo_list_headline_with_summary(self):
        block = ToolCallBlock(
            ToolCall(
                id="call_4",
                function=ToolCall.FunctionBody(
                    name="SetTodoList",
                    arguments=(
                        '{"todos":[{"title":"Inspect parser","status":"pending"},'
                        '{"title":"Share findings","status":"pending"}]}'
                    ),
                ),
            )
        )

        rendered = _render_to_str(block)

        assert "Updating Todo List (2 todos)" in rendered


class TestErrorRendering:
    def test_renders_error_message_and_output(self):
        block = ToolCallBlock(
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

    def test_keeps_shell_error_summary_when_output_tail_exists(self):
        block = ToolCallBlock(
            ToolCall(
                id="call_shell_fail",
                function=ToolCall.FunctionBody(
                    name="Shell",
                    arguments='{"command": "ls /missing"}',
                ),
            )
        )
        block.append_output("ls: /missing: No such file or directory\n", stream="stderr")
        block.finish(
            ToolError(
                message="Command failed with exit code: 1.",
                brief="Failed with exit code: 1",
                output="ls: /missing: No such file or directory\n",
            )
        )

        rendered = _render_to_str(block)

        assert "Output tail" in rendered
        assert rendered.count("ls: /missing: No such file or directory") == 1
        assert "Command failed with exit code: 1." in rendered

    def test_keeps_shell_error_output_when_tail_does_not_cover_everything(self):
        block = ToolCallBlock(
            ToolCall(
                id="call_shell_partial",
                function=ToolCall.FunctionBody(
                    name="Shell",
                    arguments='{"command": "echo hi; false"}',
                ),
            )
        )
        block.append_output("line 2\n", stream="stderr")
        block.finish(
            ToolError(
                message="Command failed with exit code: 1.",
                brief="Failed with exit code: 1",
                output="line 1\nline 2\n",
            )
        )

        rendered = _render_to_str(block)

        assert "Command failed with exit code: 1." in rendered
        assert "line 1" in rendered
        assert rendered.count("line 2") == 2

    def test_renders_detailed_error_message_instead_of_generic_brief(self):
        block = ToolCallBlock(
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
        block = ToolCallBlock(
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


def test_renders_line_numbers_for_writefile_diff_display() -> None:
    block = ToolCallBlock(
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
                    old_start_line=42,
                    new_start_line=42,
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "File successfully overwritten." in rendered
    assert "src/example.py" in rendered
    assert "@@ -42 +42 @@" in rendered
    assert "42    │ -before" in rendered
    assert "42 │ +after" in rendered


def test_renders_line_numbers_for_top_level_edit_diff_display() -> None:
    block = ToolCallBlock(
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


def test_renders_todo_display_block_with_items() -> None:
    block = ToolCallBlock(
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
                            executor="main",
                        ),
                        TodoDisplayItem(title="Share findings", status="pending", executor="main"),
                    ]
                )
            ],
        )
    )

    rendered = _render_to_str(block)

    assert "Inspect parser" in rendered
    assert "Share findings" in rendered
