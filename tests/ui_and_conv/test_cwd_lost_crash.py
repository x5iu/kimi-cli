from __future__ import annotations

import types
from io import StringIO
from typing import Any, cast

import pytest
from rich.console import Console

import kimi_cli.ui.shell as shell_module
from kimi_cli.loop import AgentLoop
from kimi_cli.loop.agent import Runtime
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop
from kimi_cli.ui.shell import Shell
from kimi_cli.ui.shell.prompt import CwdLostError, UserInput


def _fake_agent_loop(**overrides: Any) -> AgentLoop:
    base: dict[str, Any] = {
        "name": "Test",
        "model_name": None,
        "model_capabilities": set(),
        "thinking": False,
        "status": types.SimpleNamespace(context_usage=0.0, context_tokens=0, max_context_tokens=0),
        "available_slash_commands": [],
        "run": None,
    }
    base.update(overrides)
    return cast(AgentLoop, types.SimpleNamespace(**base))


@pytest.mark.asyncio
async def test_print_cwd_lost_crash_shows_unknown_for_non_kimi_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = Shell(_fake_agent_loop())
    cap = Console(file=StringIO(), width=120, record=True, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr(shell_module.console, "print", cap.print)
    shell._print_cwd_lost_crash()
    out = cap.export_text()
    assert "unknown" in out
    assert "Session crashed" in out


@pytest.mark.asyncio
async def test_print_cwd_lost_crash_shows_session_for_kimi_loop(
    runtime: Runtime,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kimi_cli.loop.agent import Agent
    from kimi_cli.loop.context import Context
    from llmkit.tooling.empty import EmptyToolset

    soul = KimiAgentLoop(
        Agent(
            name="t",
            system_prompt="p",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "h.jsonl"),
    )
    shell = Shell(soul)
    cap = Console(file=StringIO(), width=200, record=True, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr(shell_module.console, "print", cap.print)
    shell._print_cwd_lost_crash()
    out = cap.export_text()
    assert str(runtime.session.id) in out
    wd = str(runtime.session.work_dir)
    assert wd in out or wd.rstrip("/").split("/")[-1] in out


@pytest.mark.asyncio
async def test_shell_run_breaks_on_getcwd_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = Shell(_fake_agent_loop())

    class _Sess:
        def __enter__(self) -> _Sess:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        async def prompt(self) -> UserInput | None:
            raise AssertionError("unreachable")

        def execute_deferred_erase(self) -> None:
            return None

    monkeypatch.setattr(shell_module, "CustomPromptSession", lambda **_: _Sess())
    monkeypatch.setattr(shell_module, "_print_welcome_info", lambda *a, **k: None)
    monkeypatch.setattr(shell_module.os, "getcwd", lambda: (_ for _ in ()).throw(OSError(2, "ENOENT")))
    cap = Console(file=StringIO(), width=120, record=True, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr(shell_module.console, "print", cap.print)
    ok = await shell.run()
    assert ok is True
    assert "Session crashed" in cap.export_text()


def test_working_dir_text_getcwd_oserror_requests_app_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.prompt import CustomPromptSession as CPS

    class _StubApp:
        def __init__(self) -> None:
            self.exit_exceptions: list[BaseException] = []

        def exit(self, *, exception: BaseException | None = None, result: object | None = None) -> None:
            if exception is not None:
                self.exit_exceptions.append(exception)

    stub = _StubApp()
    s = CPS(
        status_provider=lambda: types.SimpleNamespace(
            context_usage=0.0, context_tokens=0, max_context_tokens=0
        ),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    monkeypatch.setattr(
        "kimi_cli.ui.shell.toolbar.os.getcwd",
        lambda: (_ for _ in ()).throw(OSError(2, "ENOENT")),
    )
    monkeypatch.setattr("kimi_cli.ui.shell.toolbar.get_app_or_none", lambda: stub)
    assert s._working_dir_text() == ""
    assert len(stub.exit_exceptions) == 1
    assert isinstance(stub.exit_exceptions[0], CwdLostError)


def test_working_dir_text_getcwd_oserror_without_app_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from kimi_cli.ui.shell.prompt import CustomPromptSession as CPS

    s = CPS(
        status_provider=lambda: types.SimpleNamespace(
            context_usage=0.0, context_tokens=0, max_context_tokens=0
        ),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    monkeypatch.setattr(
        "kimi_cli.ui.shell.toolbar.os.getcwd",
        lambda: (_ for _ in ()).throw(OSError(2, "ENOENT")),
    )
    monkeypatch.setattr("kimi_cli.ui.shell.toolbar.get_app_or_none", lambda: None)
    assert s._working_dir_text() == ""
