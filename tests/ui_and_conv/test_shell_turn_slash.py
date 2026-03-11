from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.ui.shell import Shell
from kimi_cli.ui.shell.prompt import PromptMode, UserInput
from kimi_cli.wire.types import TextPart


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
        assert (
            submit_handler(
                UserInput(
                    mode=PromptMode.AGENT,
                    command="/help",
                    content=[TextPart(text="/help")],
                )
            )
            is True
        )
        rendered = live_view.render_ansi(80)
        assert "Reminder:" in rendered
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

    prompt_session = SimpleNamespace(run_turn_ui=fake_run_turn_ui)
    keep_running = await shell._run_interactive_turn(prompt_session, "hello")

    assert keep_running is True
    assert recorded == [[TextPart(text="/help")]]
