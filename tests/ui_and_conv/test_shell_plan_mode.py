from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell import Shell


@pytest.mark.asyncio
async def test_shell_prompt_plan_mode_toggle_callback_uses_manual_toggle(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    captured: dict[str, object] = {}

    soul = KimiSoul(
        Agent(
            name="Shell Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )
    shell = Shell(soul)

    class FakePromptSession:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        async def prompt(self):
            raise EOFError

    async def fake_replay_recent_history(*args, **kwargs) -> None:
        return None

    def fake_start_background_task(coro):
        coro.close()
        return None

    monkeypatch.setattr(shell_module, "CustomPromptSession", FakePromptSession)
    monkeypatch.setattr(shell_module, "replay_recent_history", fake_replay_recent_history)
    monkeypatch.setattr(shell_module, "ensure_tty_sane", lambda: None)
    monkeypatch.setattr(shell_module, "ensure_new_line", lambda: None)
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell, "_start_background_task", fake_start_background_task)

    assert await shell.run() is True
    toggle_callback = captured.get("plan_mode_toggle_callback")
    assert toggle_callback is not None

    toggle = cast(Callable[[], Awaitable[bool]], toggle_callback)
    assert await toggle() is True
    assert soul.plan_mode is True
    assert runtime.session.state.plan_mode is True

    assert await toggle() is False
    assert soul.plan_mode is False
    assert runtime.session.state.plan_mode is False
