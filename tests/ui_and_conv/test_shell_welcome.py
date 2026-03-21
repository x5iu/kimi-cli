from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.console import Console

from kimi_cli.app import KimiCLI
from kimi_cli.soul import Soul
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell import Shell


@pytest.mark.asyncio
async def test_run_shell_welcome_info_excludes_promotional_tips(
    runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    captured: dict[str, object] = {}

    class FakeShell:
        def __init__(self, soul, welcome_info=None):
            captured["soul"] = soul
            captured["welcome_info"] = list(welcome_info or [])

        async def run(self, command):
            captured["command"] = command
            return True

    monkeypatch.setattr(shell_module, "Shell", FakeShell)

    cli = KimiCLI(
        cast(KimiSoul, SimpleNamespace(name="Test", model_name="gpt-4")),
        runtime,
        {},
    )

    @asynccontextmanager
    async def fake_env():
        yield

    monkeypatch.setattr(cli, "_env", fake_env)

    assert await cli.run_shell() is True

    welcome_info = cast(list[Any], captured["welcome_info"])
    values = [item.value for item in welcome_info]

    assert all("latest kimi-k2.5 model" not in value for value in values)
    assert all("Kimi Code Web UI" not in value for value in values)


@pytest.mark.asyncio
async def test_shell_run_does_not_start_background_update(monkeypatch: pytest.MonkeyPatch) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    created_tasks: list[object] = []

    class FakePromptSession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        async def prompt(self):
            raise EOFError
        def execute_deferred_erase(self):
            pass

    def fake_create_task(coro):
        created_tasks.append(coro)
        raise AssertionError("unexpected background task")

    monkeypatch.setattr(shell_module, "CustomPromptSession", FakePromptSession)
    monkeypatch.setattr(shell_module, "ensure_tty_sane", lambda: None)
    monkeypatch.setattr(shell_module, "ensure_new_line", lambda: None)
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell_module.asyncio, "create_task", fake_create_task)

    shell = Shell(
        cast(
            Soul,
            SimpleNamespace(
                name="Test",
                available_slash_commands=[],
                status=SimpleNamespace(),
                model_capabilities=set(),
                model_name="kimi-code",
                thinking=False,
            ),
        )
    )

    assert await shell.run() is True
    assert created_tasks == []


def test_print_welcome_info_does_not_show_update_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    render_console = Console(record=True, width=120, highlight=False)
    latest_version_file = tmp_path / "latest_version.txt"
    latest_version_file.write_text("999.0.0", encoding="utf-8")

    monkeypatch.setattr(shell_module, "console", render_console)
    monkeypatch.setattr(shell_module, "LATEST_VERSION_FILE", latest_version_file, raising=False)

    shell_module._print_welcome_info(
        "Test",
        [shell_module.WelcomeInfoItem(name="Directory", value=".")],
    )

    rendered = render_console.export_text()

    assert "New version available" not in rendered
