from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.console import Console

from kimi_cli.eventbus.types import ContentPart, ImageURLPart, TextPart
from kimi_cli.loop import AgentLoop
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop
from kimi_cli.ui.shell import Shell
from kimi_cli.ui.shell.prompt import CustomPromptSession, PromptMode, TurnSubmitResult, UserInput
from kimi_cli.utils.slashcmd import parse_slash_command_call
from llmkit.chat_provider import APIStatusError
from llmkit.tooling.empty import EmptyToolset


def _fake_soul(**overrides: Any) -> AgentLoop:
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
    return cast(AgentLoop, SimpleNamespace(**base))


def _fake_prompt_session(**overrides: Any) -> CustomPromptSession:
    overrides.setdefault("execute_deferred_erase", lambda: None)
    return cast(CustomPromptSession, SimpleNamespace(**overrides))


@pytest.mark.asyncio
async def test_slash_command_submitted_during_turn_is_treated_as_steer_text(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiAgentLoop(
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
        assert kwargs["turn_prompt"] == "hello"
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

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    recorded: list[object] = []

    def fake_steer(content) -> None:
        recorded.append(content)

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)
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
    soul = KimiAgentLoop(
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
        assert kwargs["turn_prompt"] == "hello"
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

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    recorded: list[object] = []

    def fake_steer(content) -> None:
        recorded.append(content)

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(soul, "steer", fake_steer)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert recorded == [image_content]


@pytest.mark.asyncio
async def test_reminder_submitted_during_turn_rejects_when_llm_not_set(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime.llm = None
    soul = KimiAgentLoop(
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

    results: list[TurnSubmitResult] = []
    recorded: list[object] = []

    async def fake_run_turn_ui(*, submit_handler, **kwargs) -> None:
        results.append(
            submit_handler(
                UserInput(
                    mode=PromptMode.AGENT,
                    command="later",
                    content=[TextPart(text="later")],
                )
            )
        )

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(soul, "steer", lambda content: recorded.append(content))

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert results == [TurnSubmitResult.reject('LLM not set, send "/setup" to configure')]
    assert recorded == []


@pytest.mark.asyncio
async def test_image_reminder_submitted_during_turn_rejects_when_model_lacks_capability(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = set()
    soul = KimiAgentLoop(
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

    image_content: list[ContentPart] = [
        ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")),
    ]
    results: list[TurnSubmitResult] = []
    recorded: list[object] = []

    async def fake_run_turn_ui(*, submit_handler, **kwargs) -> None:
        results.append(
            submit_handler(
                UserInput(
                    mode=PromptMode.AGENT,
                    command="[image]",
                    content=image_content,
                )
            )
        )

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(soul, "steer", lambda content: recorded.append(content))

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert len(results) == 1
    assert results[0].accepted is False
    assert "image_in" in results[0].feedback
    assert recorded == []


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

    async def fake_run_slash_command(command_call, **kwargs) -> None:
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

    async def fake_run_interactive_turn(prompt_session_arg, user_input, **kwargs) -> bool:
        assert prompt_session_arg is prompt_session
        received.append(user_input)
        return True

    cast(Any, shell)._run_interactive_turn = fake_run_interactive_turn

    keep_running = await shell._handle_agent_input(
        prompt_session,
        UserInput(mode=PromptMode.AGENT, command="/reset", content=[TextPart(text="/reset")]),
    )

    assert keep_running is True
    assert received == [[TextPart(text="/reset")]]


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

    async def fake_run_agent_loop(*args, **kwargs) -> None:
        raise APIStatusError(503, "Service unavailable.")

    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)

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

    async def fake_run_agent_loop_command(user_input: str | list[object]) -> bool:
        calls.append(user_input)
        return True

    cast(Any, shell).run_agent_loop_command = fake_run_agent_loop_command

    command_call = parse_slash_command_call("/reset")
    assert command_call is not None
    await shell._run_slash_command(command_call)

    assert calls == ["/reset"]


@pytest.mark.asyncio
async def test_turn_allowed_command_dispatches_to_shell_registry(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A /task command during a turn should run synchronously and echo output."""
    soul = KimiAgentLoop(
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

    captured_results: list[TurnSubmitResult] = []

    async def fake_run_turn_ui(*, submit_handler, live_view, **kwargs) -> None:
        result = submit_handler(
            UserInput(
                mode=PromptMode.AGENT,
                command="/task",
                content=[TextPart(text="/task")],
            )
        )
        captured_results.append(result)

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert len(captured_results) == 1
    assert captured_results[0].accepted is True


@pytest.mark.asyncio
async def test_skill_command_during_turn_queues_and_cancels(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/skill:xxx during a turn should queue the input and cancel the turn."""
    soul = KimiAgentLoop(
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

    cancel_was_set = False
    captured_results: list[TurnSubmitResult] = []

    async def fake_run_turn_ui(*, submit_handler, live_view, **kwargs) -> None:
        result = submit_handler(
            UserInput(
                mode=PromptMode.AGENT,
                command="/skill:kimi-cli-help",
                content=[TextPart(text="/skill:kimi-cli-help")],
            )
        )
        captured_results.append(result)

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        nonlocal cancel_was_set

        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())
        cancel_was_set = cancel_event.is_set()
        if cancel_event.is_set():
            from kimi_cli.loop import RunCancelled

            raise RunCancelled()

    # Intercept queued _handle_agent_input to prevent actual execution
    queued_inputs: list[UserInput] = []

    async def fake_handle(ps, user_input, **kw):
        queued_inputs.append(user_input)
        return True

    monkeypatch.setattr(shell, "_handle_agent_input", fake_handle)

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert len(captured_results) == 1
    assert captured_results[0].accepted is True
    assert cancel_was_set is True
    assert len(queued_inputs) == 1
    assert queued_inputs[0].command == "/skill:kimi-cli-help"


@pytest.mark.asyncio
async def test_async_turn_command_is_guarded(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a turn-allowed command returns a coroutine, it should be closed with an error."""
    soul = KimiAgentLoop(
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

    # Register a fake async command in shell_mode_registry
    from kimi_cli.utils.slashcmd import SlashCommand as SC

    async def _async_task(app, args):
        pass  # pragma: no cover

    shell_module = importlib.import_module("kimi_cli.ui.shell")
    original_find = shell_module.shell_mode_registry.find_command

    def patched_find(name):
        if name == "task":
            return SC(name="task", description="test", func=_async_task, aliases=[])
        return original_find(name)

    monkeypatch.setattr(shell_module.shell_mode_registry, "find_command", patched_find)

    captured_results: list[TurnSubmitResult] = []
    info_texts: list[str] = []

    async def fake_run_turn_ui(*, submit_handler, live_view, **kwargs) -> None:
        monkeypatch.setattr(live_view, "echo_info", lambda text: info_texts.append(text))
        result = submit_handler(
            UserInput(
                mode=PromptMode.AGENT,
                command="/task",
                content=[TextPart(text="/task")],
            )
        )
        captured_results.append(result)

    async def fake_run_agent_loop(
        loop_obj, user_input, ui_loop_fn, cancel_event, event_log
    ) -> None:
        class _FakeWire:
            @staticmethod
            def ui_side(merge: bool = False):
                return None

        await ui_loop_fn(_FakeWire())

    monkeypatch.setattr(shell_module, "run_agent_loop", fake_run_agent_loop)

    prompt_session = _fake_prompt_session(run_turn_ui=fake_run_turn_ui)
    await shell._run_interactive_turn(prompt_session, "hello")

    assert captured_results[0].accepted is True
    assert any("cannot run during a turn" in t for t in info_texts)


@pytest.mark.asyncio
async def test_slash_command_with_pasted_text_placeholder_expands_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a slash command contains a pasted-text placeholder, the agent loop
    must receive the expanded content parts, not the raw placeholder string."""
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell_module.shell_slash_registry, "find_command", lambda name: None)

    shell = Shell(
        _fake_soul(
            available_slash_commands=[SimpleNamespace(name="skill:orchestrator", aliases=[])],
            status=SimpleNamespace(),
        )
    )
    prompt_session = _fake_prompt_session()
    received: list[object] = []

    async def fake_run_interactive_turn(prompt_session_arg, user_input, **kwargs) -> bool:
        received.append(user_input)
        return True

    cast(Any, shell)._run_interactive_turn = fake_run_interactive_turn

    pasted_text = "line1\nline2\nline3\nline4\nline5"
    expanded_content: list[ContentPart] = [
        TextPart(text="/skill:orchestrator "),
        TextPart(text=pasted_text),
    ]

    keep_running = await shell._handle_agent_input(
        prompt_session,
        UserInput(
            mode=PromptMode.AGENT,
            command="/skill:orchestrator [Pasted text #1 +5 lines]",
            content=expanded_content,
        ),
    )

    assert keep_running is True
    assert len(received) == 1
    assert received[0] == expanded_content
    # Verify the actual pasted text is present, not the placeholder
    texts = [p.text for p in received[0] if isinstance(p, TextPart)]
    assert any(pasted_text in t for t in texts)
    assert "[Pasted text" not in " ".join(texts)


@pytest.mark.asyncio
async def test_slash_command_with_image_placeholder_expands_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a slash command contains an image placeholder, the agent loop
    must receive the resolved ImageURLPart, not the raw placeholder string."""
    shell_module = importlib.import_module("kimi_cli.ui.shell")
    monkeypatch.setattr(shell_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell_module.shell_slash_registry, "find_command", lambda name: None)

    shell = Shell(
        _fake_soul(
            available_slash_commands=[SimpleNamespace(name="skill:orchestrator", aliases=[])],
            status=SimpleNamespace(),
        )
    )
    prompt_session = _fake_prompt_session()
    received: list[object] = []

    async def fake_run_interactive_turn(prompt_session_arg, user_input, **kwargs) -> bool:
        received.append(user_input)
        return True

    cast(Any, shell)._run_interactive_turn = fake_run_interactive_turn

    image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="data:image/png;base64,AAAA"))
    expanded_content: list[ContentPart] = [
        TextPart(text="/skill:orchestrator "),
        image_part,
    ]

    keep_running = await shell._handle_agent_input(
        prompt_session,
        UserInput(
            mode=PromptMode.AGENT,
            command="/skill:orchestrator [image:test.png,100x100]",
            content=expanded_content,
        ),
    )

    assert keep_running is True
    assert len(received) == 1
    assert received[0] == expanded_content
    # Verify the image part is present
    image_parts = [p for p in received[0] if isinstance(p, ImageURLPart)]
    assert len(image_parts) == 1
    assert image_parts[0].image_url.url == "data:image/png;base64,AAAA"
