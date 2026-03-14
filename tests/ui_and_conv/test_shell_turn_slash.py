from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from kosong.chat_provider import APIStatusError
from kosong.tooling.empty import EmptyToolset
from rich.console import Console

from kimi_cli.soul import Soul
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell import Shell
from kimi_cli.ui.shell.prompt import CustomPromptSession, PromptMode, TurnSubmitResult, UserInput
from kimi_cli.utils.slashcmd import parse_slash_command_call
from kimi_cli.wire.types import ImageURLPart, TextPart


def _fake_soul(**overrides: Any) -> Soul:
    base: dict[str, Any] = {
        "name": "Test",
        "model_name": None,
        "model_capabilities": set(),
        "thinking": False,
        "status": SimpleNamespace(context_usage=0.0, context_tokens=0, max_context_tokens=0),
        "available_slash_commands": [],
        "run": None,
    }
    base.update(overrides)
    return cast(Soul, SimpleNamespace(**base))


def _fake_prompt_session(**overrides: Any) -> CustomPromptSession:
    return cast(CustomPromptSession, SimpleNamespace(**overrides))


@pytest.mark.asyncio
async def test_slash_command_submitted_during_turn_is_treated_as_steer_text(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(shell, "_echo_agent_input", lambda _: None)

    async def fake_run_turn_ui(*, submit_handler, live_view, **kwargs) -> None:
        assert submit_handler(
            UserInput(
                mode=PromptMode.AGENT,
                command="/help",
                content=[TextPart(text="/help")],
            )
        ) == TurnSubmitResult.accept(persist_history=True)
        rendered = live_view.render_ansi(80)
        assert "Reminder" in rendered
        assert "╭" in rendered
        assert "/help" in rendered

    async def fake_run_soul(soul_obj, user_input, ui_loop_fn, cancel_event, wire_file) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    recorded: list[object] = []

    def fake_steer(content) -> None:
        recorded.append(content)

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_soul", fake_run_soul)
    monkeypatch.setattr(soul, "steer", fake_steer)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert recorded == [[TextPart(text="/help")]]


@pytest.mark.asyncio
async def test_image_reminder_submitted_during_turn_shows_image_marker(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(shell, "_echo_agent_input", lambda _: None)

    image_content = [
        TextPart(text='<image path="/tmp/example.png">'),
        ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
        TextPart(text="</image>"),
    ]

    async def fake_run_turn_ui(*, submit_handler, live_view, **kwargs) -> None:
        assert submit_handler(
            UserInput(
                mode=PromptMode.AGENT,
                command="[image:abc123,10x10]",
                content=image_content,
            )
        ) == TurnSubmitResult.accept(persist_history=True)
        rendered = live_view.render_ansi(80)
        assert "Reminder" in rendered
        assert "╭" in rendered
        assert "[image]" in rendered
        assert "<image" not in rendered

    async def fake_run_soul(soul_obj, user_input, ui_loop_fn, cancel_event, wire_file) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    recorded: list[object] = []

    def fake_steer(content) -> None:
        recorded.append(content)

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_soul", fake_run_soul)
    monkeypatch.setattr(soul, "steer", fake_steer)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert recorded == [image_content]


def test_echo_agent_input_shows_image_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    render_console = Console(record=True, width=80, highlight=False)
    monkeypatch.setattr(shell_module, "console", render_console)

    shell = Shell(_fake_soul(available_slash_commands=[]))
    shell._echo_agent_input(
        UserInput(
            mode=PromptMode.AGENT,
            command="[image:abc123,10x10]",
            content=[
                TextPart(text='<image path="/tmp/example.png">'),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
                TextPart(text="</image>"),
            ],
        )
    )

    rendered = render_console.export_text()
    assert "User" in rendered
    assert "[image]" in rendered
    assert "╭" in rendered


@pytest.mark.asyncio
async def test_top_level_slash_command_is_not_echoed_as_user_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    printed: list[str] = []
    monkeypatch.setattr(
        shell_module.console, "print", lambda text, *args, **kwargs: printed.append(text)
    )

    shell = Shell(_fake_soul(status=SimpleNamespace()))
    calls: list[str | list[object]] = []

    async def fake_run_slash_command(command_call) -> None:
        calls.append(command_call.name)

    cast(Any, shell)._run_slash_command = fake_run_slash_command

    keep_running = await shell._handle_agent_input(
        _fake_prompt_session(),
        UserInput(mode=PromptMode.AGENT, command="/help", content=[TextPart(text="/help")]),
    )

    assert keep_running is True
    assert calls == ["help"]
    assert printed == []


@pytest.mark.asyncio
async def test_top_level_soul_slash_command_uses_interactive_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)

    monkeypatch.setattr(shell_module.shell_slash_registry, "find_command", lambda name: None)

    shell = Shell(
        _fake_soul(
            available_slash_commands=[SimpleNamespace(name="clear", aliases=["reset"])],
            status=SimpleNamespace(),
        )
    )
    prompt_session = _fake_prompt_session()
    received: list[object] = []

    async def fake_run_interactive_turn(prompt_session_arg, user_input) -> bool:
        assert prompt_session_arg is prompt_session
        received.append(user_input)
        return True

    cast(Any, shell)._run_interactive_turn = fake_run_interactive_turn

    keep_running = await shell._handle_agent_input(
        prompt_session,
        UserInput(mode=PromptMode.AGENT, command="/reset", content=[TextPart(text="/reset")]),
    )

    assert keep_running is True
    assert received == ["/reset"]


@pytest.mark.asyncio
async def test_interactive_turn_keeps_shell_alive_on_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    printed: list[str] = []
    monkeypatch.setattr(
        shell_module.console, "print", lambda text, *args, **kwargs: printed.append(str(text))
    )

    shell = Shell(
        _fake_soul(
            status=SimpleNamespace(context_usage=0.0, context_tokens=0, max_context_tokens=0),
        )
    )

    async def fake_run_soul(*args, **kwargs) -> None:
        raise APIStatusError(503, "Service unavailable.")

    monkeypatch.setattr(shell_module, "run_soul", fake_run_soul)

    keep_running = await shell._run_interactive_turn(_fake_prompt_session(), "hello")

    assert keep_running is True
    assert any("LLM provider error" in line for line in printed)


@pytest.mark.asyncio
async def test_run_slash_command_accepts_soul_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell_module.shell_slash_registry, "find_command", lambda name: None)

    shell = Shell(
        _fake_soul(
            available_slash_commands=[SimpleNamespace(name="clear", aliases=["reset"])],
            status=SimpleNamespace(),
        )
    )
    calls: list[str | list[object]] = []

    async def fake_run_soul_command(user_input: str | list[object]) -> bool:
        calls.append(user_input)
        return True

    cast(Any, shell).run_soul_command = fake_run_soul_command

    command_call = parse_slash_command_call("/reset")
    assert command_call is not None
    await shell._run_slash_command(command_call)

    assert calls == ["/reset"]
