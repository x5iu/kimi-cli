from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul, StepOutcome
from kimi_cli.soul.message import INTERNAL_USER_NAME
from kimi_cli.wire.types import ImageURLPart, TextPart


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
    assert soul.context.history[-1].name == INTERNAL_USER_NAME

    reminder_text = soul.context.history[-1].extract_text(" ")
    assert "<system-reminder>" in reminder_text
    assert "additional user instruction" in reminder_text
    assert "do not stop, summarize, or conclude" in reminder_text
    assert "do not explicitly acknowledge, answer, or quote the reminder" in reminder_text
    assert "Do not use meta phrasing such as 'based on your reminder'" in reminder_text
    assert (
        "Keep the final response centered on the user's original turn-opening request"
        in reminder_text
    )
    assert "also do this" in reminder_text


@pytest.mark.asyncio
async def test_consume_pending_steer_preserves_non_text_content(
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

    image_part = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url="https://example.com/reminder.png")
    )
    turn_id = soul._begin_turn()
    try:
        soul.steer([TextPart(text="look at this image"), image_part])
        consumed = await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    assert consumed is True
    reminder_message = soul.context.history[-1]
    assert reminder_message.role == "user"
    assert reminder_message.name == INTERNAL_USER_NAME
    assert reminder_message.content[0] == TextPart(
        text=(
            "<system-reminder>\n"
            f"{KimiSoul._steer_instruction_text()}\n\n"
            "Reminder content follows in the rest of this message.\n"
            "</system-reminder>"
        )
    )
    assert reminder_message.content[1:] == [TextPart(text="look at this image"), image_part]


@pytest.mark.asyncio
async def test_consume_pending_steer_downgrades_unsupported_media_to_text(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    assert runtime.llm is not None
    runtime.llm.capabilities = set()
    soul = KimiSoul(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    image_part = ImageURLPart(
        image_url=ImageURLPart.ImageURL(url="https://example.com/reminder.png")
    )
    turn_id = soul._begin_turn()
    try:
        soul.steer([image_part])
        consumed = await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    assert consumed is True
    assert len(soul.context.history) == 1
    reminder_text = soul.context.history[-1].extract_text(" ")
    assert "Reminder:" in reminder_text
    assert "[image]" in reminder_text


@pytest.mark.asyncio
async def test_consume_pending_steer_string_and_list_use_same_wrapper_semantics(
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

    string_message = soul._build_steer_message("plain text reminder")
    list_message = soul._build_steer_message([TextPart(text="plain text reminder")])

    assert string_message.name == INTERNAL_USER_NAME
    assert list_message.name == INTERNAL_USER_NAME
    assert string_message.content[0] == list_message.content[0]
    assert string_message.content[1:] == [TextPart(text="plain text reminder")]
    assert list_message.content[1:] == [TextPart(text="plain text reminder")]


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
        message
        for message in soul.context.history
        if message.role == "user" and "additional user instruction" in message.extract_text(" ")
    ]
    assert len(reminder_messages) == 1
    assert reminder_messages[0].name == INTERNAL_USER_NAME
    reminder_text = reminder_messages[0].extract_text(" ")
    assert "<system-reminder>" in reminder_text
    assert "Reminder content follows in the rest of this message." in reminder_text
    assert reminder_text.endswith("also do this")
    assert step_calls == 2
