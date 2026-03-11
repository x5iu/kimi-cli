from __future__ import annotations

from pathlib import Path

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul


@pytest.mark.asyncio
async def test_consume_pending_steer_appends_user_reminder(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    soul.steer("also do this")

    consumed = await soul._consume_pending_steers()

    assert consumed is True
    assert [message.role for message in soul.context.history] == ["user"]

    reminder_text = soul.context.history[-1].extract_text(" ")
    assert "<system-reminder>" in reminder_text
    assert "latest user instruction" in reminder_text
    assert "also do this" in reminder_text
