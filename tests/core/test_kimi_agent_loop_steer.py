from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from llmkit.message import Message
from llmkit.tooling.empty import EmptyToolset

from kimi_cli.eventbus.types import ImageURLPart, TextPart
from kimi_cli.loop.agent import Agent, Runtime
from kimi_cli.loop.context import Context
from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop, StepOutcome
from kimi_cli.loop.message import INTERNAL_USER_NAME


@pytest.mark.asyncio
async def test_consume_pending_steer_appends_user_reminder(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    soul = KimiAgentLoop(
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
    soul = KimiAgentLoop(
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
            f"{KimiAgentLoop._steer_instruction_text()}\n\n"
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
    soul = KimiAgentLoop(
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
    soul = KimiAgentLoop(
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
    soul = KimiAgentLoop(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    step_started = asyncio.Event()
    release_step = asyncio.Event()
    step_calls = 0

    async def fake_step(self: KimiAgentLoop) -> StepOutcome:
        nonlocal step_calls
        step_calls += 1
        if step_calls == 1:
            step_started.set()
            await release_step.wait()
        return StepOutcome(
            stop_reason="no_tool_calls",
            assistant_message=Message(role="assistant", content=f"step {step_calls}"),
        )

    monkeypatch.setattr(KimiAgentLoop, "_step", fake_step)
    monkeypatch.setattr("kimi_cli.loop.kimi_agent_loop.bus_send", lambda _: None)

    run_task = asyncio.create_task(soul.run("hello"))
    await step_started.wait()
    soul.steer("also do this")
    release_step.set()
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


@pytest.mark.asyncio
async def test_second_steer_uses_abbreviated_instruction(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    """The second steer in a turn should use abbreviated instruction text."""
    soul = KimiAgentLoop(
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
        soul.steer("first reminder")
        await soul._consume_pending_steers()

        soul.steer("second reminder")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    # First steer gets full instruction
    first_text = soul.context.history[0].extract_text(" ")
    assert "additional user instruction" in first_text

    # Second steer gets abbreviated instruction
    second_text = soul.context.history[1].extract_text(" ")
    assert "same handling rules" in second_text
    assert "additional user instruction" not in second_text
    assert "second reminder" in second_text


@pytest.mark.asyncio
async def test_steer_count_resets_on_new_turn(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    """Steer count resets when a new turn begins, so the first steer gets full instruction."""
    soul = KimiAgentLoop(
        Agent(
            name="Steer Test Agent",
            system_prompt="Test system prompt.",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    # First turn
    turn_id = soul._begin_turn()
    try:
        soul.steer("turn1 steer")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    # Second turn
    turn_id2 = soul._begin_turn()
    try:
        soul.steer("turn2 first steer")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id2)

    # Both first steers of each turn should have the full instruction
    first_text = soul.context.history[0].extract_text(" ")
    assert "additional user instruction" in first_text

    second_text = soul.context.history[1].extract_text(" ")
    assert "additional user instruction" in second_text


@pytest.mark.asyncio
async def test_steer_count_resets_after_compaction(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    """After compaction mid-turn, steer count resets so next steer gets full instruction."""
    soul = KimiAgentLoop(
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
        # First steer — full instruction
        soul.steer("first reminder")
        await soul._consume_pending_steers()
        assert soul._turn_steer_count == 1

        # Simulate compaction: increment generation and reset count
        soul._compaction_generation += 1
        soul._turn_steer_count = 0

        # Next steer after compaction — should get full instruction again
        soul.steer("post-compaction reminder")
        await soul._consume_pending_steers()
    finally:
        soul._end_turn(turn_id)

    # The post-compaction steer (history[-1]) should have full instruction, not brief
    post_compaction_text = soul.context.history[-1].extract_text(" ")
    assert "additional user instruction" in post_compaction_text
    assert "same handling rules" not in post_compaction_text
    assert "post-compaction reminder" in post_compaction_text
