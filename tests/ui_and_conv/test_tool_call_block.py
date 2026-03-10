from __future__ import annotations

from io import StringIO

from kosong.tooling import ToolError
from rich.console import Console

from kimi_cli.ui.shell.visualize import (
    MAX_TOOL_ERROR_OUTPUT_LINES,
    _ToolCallBlock,
)
from kimi_cli.wire.types import ToolCall


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
