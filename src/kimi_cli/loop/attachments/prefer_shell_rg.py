from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from kimi_cli.loop.attachment import Attachment, AttachmentProvider
from kimi_cli.tools.file.rg_path import find_existing_rg, format_rg_command
from llmkit.message import Message, TextPart

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop


_REMINDER_PREFIX = "Prefer Shell with `rg` for file-content search."


class PreferShellRgAttachmentProvider(AttachmentProvider):
    async def get_attachments(
        self,
        history: Sequence[Message],
        agent_loop: KimiAgentLoop,
    ) -> list[Attachment]:
        if _has_prefer_shell_rg_reminder(history) or not _has_shell_tool(agent_loop):
            return []

        rg_path = find_existing_rg()
        if rg_path is None:
            return []

        is_powershell = agent_loop.runtime.environment.shell_name == "Windows PowerShell"
        return [
            Attachment(
                type="prefer_shell_rg",
                content=_reminder_text(rg_path=rg_path, is_powershell=is_powershell),
                is_hint=True,
            )
        ]


def _has_shell_tool(agent_loop: KimiAgentLoop) -> bool:
    toolset = agent_loop.agent.toolset
    find = getattr(toolset, "find", None)
    if callable(find):
        return find("Shell") is not None
    return True


def _has_prefer_shell_rg_reminder(history: Sequence[Message]) -> bool:
    for msg in history:
        for part in msg.content:
            if isinstance(part, TextPart) and _REMINDER_PREFIX in part.text:
                return True
    return False


def _reminder_text(*, rg_path: Path, is_powershell: bool) -> str:
    command = format_rg_command(rg_path, is_powershell=is_powershell)
    return "\n".join(
        [
            _REMINDER_PREFIX,
            f"`rg` path: `{rg_path}`.",
            (f"When searching file contents, prefer the Shell tool with `{command}`."),
        ]
    )
