from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome


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

    turn_id = soul._begin_turn()
    try:
        soul.steer("also do this")

        consumed = await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    assert consumed is True
    assert [message.role for message in soul.context.history] == ["user"]

    reminder_text = soul.context.history[-1].extract_text(" ")
    assert "<system-reminder>" in reminder_text
    assert "additional user instruction" in reminder_text
    assert "do not stop, summarize, or conclude" in reminder_text
    assert "do not explicitly acknowledge, answer, or quote the reminder" in reminder_text
    assert "Do not use meta phrasing such as 'based on your reminder'" in reminder_text
    assert "Keep the final response centered on the user's original turn-opening request" in reminder_text
    assert "also do this" in reminder_text


@pytest.mark.asyncio
async def test_early_turn_steer_is_not_dropped_before_agent_loop_starts(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    oauth_started = asyncio.Event()
    release_oauth = asyncio.Event()
    step_calls = 0

    async def fake_ensure_fresh(_runtime) -> None:
        oauth_started.set()
        await release_oauth.wait()

    async def fake_step(self: KimiSoul) -> StepOutcome:
        nonlocal step_calls
        step_calls += 1
        return StepOutcome(
            stop_reason="no_tool_calls",
            assistant_message=Message(role="assistant", content=f"step {step_calls}"),
        )

    monkeypatch.setattr(runtime.oauth, "ensure_fresh", fake_ensure_fresh)
    monkeypatch.setattr(KimiSoul, "_step", fake_step)
    monkeypatch.setattr("kimi_cli.soul.kimisoul.wire_send", lambda _: None)

    run_task = asyncio.create_task(soul.run("hello"))
    await oauth_started.wait()
    soul.steer("also do this")
    release_oauth.set()
    await run_task

    reminder_messages = [
        message.extract_text(" ")
        for message in soul.context.history
        if message.role == "user" and "additional user instruction" in message.extract_text(" ")
    ]
    assert reminder_messages == [
        (
            "<system-reminder>\n"
            "The user sent a new reminder during the current turn. "
            "Treat it as an additional user instruction for this task. "
            "Incorporate it into the ongoing turn, but do not stop, summarize, or conclude "
            "the turn only because of this reminder. "
            "Use the reminder as hidden steering: do not explicitly acknowledge, answer, or "
            "quote the reminder by itself in the final response unless the original turn prompt "
            "directly asks for that. Do not use meta phrasing such as 'based on your reminder', "
            "'you just added', or 'you mentioned later'. Keep the final response centered on the "
            "user's original turn-opening request.\n\n"
            "Reminder:\nalso do this\n"
            "</system-reminder>"
        )
    ]
    assert step_calls == 2
