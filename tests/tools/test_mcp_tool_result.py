"""Tests for convert_mcp_tool_result: truncation + unsupported content handling."""
from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import mcp.types

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.toolset import MCP_MAX_OUTPUT_CHARS, convert_mcp_tool_result
from llmkit.tooling import ToolError, ToolOk


def _make_result(content: Sequence[mcp.types.ContentBlock], *, is_error: bool = False) -> MagicMock:
    r = MagicMock()
    r.content = content
    r.is_error = is_error
    return r


def _text(part: object) -> str:
    assert isinstance(part, TextPart)
    return part.text


class TestMCPTruncation:
    def test_small_text_passes_through(self):
        result = _make_result([mcp.types.TextContent(type="text", text="hello")])
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolOk)
        assert len(out.output) == 1
        assert _text(out.output[0]) == "hello"

    def test_text_truncated_at_budget(self):
        big_text = "x" * (MCP_MAX_OUTPUT_CHARS + 5000)
        result = _make_result([mcp.types.TextContent(type="text", text=big_text)])
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolOk)
        assert len(out.output) == 2
        assert len(_text(out.output[0])) == MCP_MAX_OUTPUT_CHARS
        assert "truncated" in _text(out.output[1]).lower()

    def test_budget_exhausted_skips_remaining_text(self):
        full = mcp.types.TextContent(type="text", text="x" * MCP_MAX_OUTPUT_CHARS)
        extra = mcp.types.TextContent(type="text", text="should be dropped")
        result = _make_result([full, extra])
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolOk)
        texts = [p for p in out.output if isinstance(p, TextPart)]
        assert len(texts) == 2
        assert "truncated" in texts[-1].text.lower()

    def test_error_result_preserves_truncation(self):
        big_text = "e" * (MCP_MAX_OUTPUT_CHARS + 1000)
        result = _make_result([mcp.types.TextContent(type="text", text=big_text)], is_error=True)
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolError)
        assert len(out.output) == 2
        assert "truncated" in out.output[1].text.lower()


class TestMCPUnsupportedContent:
    def test_unsupported_content_type_becomes_text_error(self):
        unknown = MagicMock(spec=[])
        result = _make_result([unknown])
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolOk)
        assert len(out.output) == 1
        assert "unsupported" in _text(out.output[0]).lower()

    def test_mixed_valid_and_invalid_parts(self):
        good = mcp.types.TextContent(type="text", text="valid")
        bad = MagicMock(spec=[])
        result = _make_result([good, bad])
        out = convert_mcp_tool_result(result)
        assert isinstance(out, ToolOk)
        assert len(out.output) == 2
        assert _text(out.output[0]) == "valid"
        assert "unsupported" in _text(out.output[1]).lower()
