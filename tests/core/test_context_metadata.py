"""Tests for Context.append_message with message_metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.context import Context
from llmkit.message import Message, Role


def _msg(role: Role, text: str) -> Message:
    return Message(role=role, content=[TextPart(text=text)])


@pytest.fixture
def context_file(tmp_path: Path) -> Path:
    return tmp_path / "context.jsonl"


async def test_is_error_written_to_jsonl(context_file: Path):
    """message_metadata should merge extra fields (like is_error) into JSONL lines."""
    ctx = Context(context_file)

    msgs = [
        _msg("tool", "success output"),
        _msg("tool", "failure output"),
    ]
    metadata: list[dict[str, object]] = [
        {"is_error": False},
        {"is_error": True},
    ]
    await ctx.append_message(msgs, message_metadata=metadata)

    lines = [
        json.loads(line) for line in context_file.read_text(encoding="utf-8").strip().splitlines()
    ]

    assert len(lines) == 2
    assert lines[0]["is_error"] is False
    assert lines[0]["role"] == "tool"
    assert lines[1]["is_error"] is True
    assert lines[1]["role"] == "tool"


async def test_append_without_metadata_has_no_extra_fields(context_file: Path):
    """Without message_metadata, JSONL lines should not contain is_error."""
    ctx = Context(context_file)

    await ctx.append_message(_msg("assistant", "hello"))

    lines = [
        json.loads(line) for line in context_file.read_text(encoding="utf-8").strip().splitlines()
    ]

    assert len(lines) == 1
    assert "is_error" not in lines[0]
    assert lines[0]["role"] == "assistant"


async def test_metadata_restored_messages_ignore_unknown_fields(context_file: Path):
    """Messages with extra metadata fields should still be restorable via
    Message.model_validate (unknown fields are silently ignored)."""
    ctx = Context(context_file)

    msgs = [_msg("tool", "output")]
    metadata = [{"is_error": True, "custom_flag": 42}]
    await ctx.append_message(msgs, message_metadata=metadata)

    # Verify that the JSONL has the extra fields
    raw = json.loads(context_file.read_text(encoding="utf-8").strip())
    assert raw["is_error"] is True
    assert raw["custom_flag"] == 42

    # Verify restoration works (model_validate ignores unknown fields)
    restored = Message.model_validate(raw)
    assert restored.role == "tool"
