from __future__ import annotations

from types import SimpleNamespace

from kimi_cli.app import KimiCLI


def test_shutdown_background_tasks_skips_kill_when_keep_alive_enabled() -> None:
    calls: list[str] = []
    cli = KimiCLI.__new__(KimiCLI)
    cli._runtime = SimpleNamespace(
        config=SimpleNamespace(background=SimpleNamespace(keep_alive_on_exit=True)),
        background_tasks=SimpleNamespace(kill_all_active=lambda reason: calls.append(reason) or []),
    )

    cli.shutdown_background_tasks()

    assert calls == []


def test_shutdown_background_tasks_kills_when_keep_alive_disabled() -> None:
    calls: list[str] = []
    cli = KimiCLI.__new__(KimiCLI)
    cli._runtime = SimpleNamespace(
        config=SimpleNamespace(background=SimpleNamespace(keep_alive_on_exit=False)),
        background_tasks=SimpleNamespace(kill_all_active=lambda reason: calls.append(reason) or []),
    )

    cli.shutdown_background_tasks()

    assert calls == ["CLI session ended"]
