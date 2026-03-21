from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from kosong.message import Message, TextPart

import kimi_cli.soul.attachments.prefer_shell_rg as prefer_shell_rg_module
from kimi_cli.soul.attachments.prefer_shell_rg import (
    _REMINDER_PREFIX,
    PreferShellRgAttachmentProvider,
)


def _make_soul_mock(*, shell_name: str = "bash", has_shell_tool: bool = True) -> MagicMock:
    soul = MagicMock()
    soul.runtime.environment.shell_name = shell_name
    soul.agent.toolset.find.return_value = object() if has_shell_tool else None
    return soul


class TestPreferShellRgAttachmentProvider:
    async def test_injects_reminder_when_rg_is_available(self, monkeypatch) -> None:
        provider = PreferShellRgAttachmentProvider()
        monkeypatch.setattr(
            prefer_shell_rg_module,
            "find_existing_rg",
            lambda: Path("/tmp/tools/rg"),
        )

        result = await provider.get_attachments([], _make_soul_mock())

        assert len(result) == 1
        assert result[0].type == "prefer_shell_rg"
        assert _REMINDER_PREFIX in result[0].content
        assert "`/tmp/tools/rg`" in result[0].content
        assert "instead of the Grep tool" in result[0].content

    async def test_returns_empty_when_rg_is_unavailable(self, monkeypatch) -> None:
        provider = PreferShellRgAttachmentProvider()
        monkeypatch.setattr(prefer_shell_rg_module, "find_existing_rg", lambda: None)

        result = await provider.get_attachments([], _make_soul_mock())

        assert result == []

    async def test_returns_empty_when_reminder_already_exists(self, monkeypatch) -> None:
        provider = PreferShellRgAttachmentProvider()
        monkeypatch.setattr(
            prefer_shell_rg_module,
            "find_existing_rg",
            lambda: Path("/tmp/tools/rg"),
        )
        history = [Message(role="user", content=[TextPart(text=_REMINDER_PREFIX)])]

        result = await provider.get_attachments(history, _make_soul_mock())

        assert result == []

    async def test_returns_empty_when_shell_tool_is_missing(self, monkeypatch) -> None:
        provider = PreferShellRgAttachmentProvider()
        monkeypatch.setattr(
            prefer_shell_rg_module,
            "find_existing_rg",
            lambda: Path("/tmp/tools/rg"),
        )

        result = await provider.get_attachments([], _make_soul_mock(has_shell_tool=False))

        assert result == []
