from __future__ import annotations

import importlib

import pytest
from llmkit.message import Message
from rich.console import Console

from kimi_cli.eventbus.types import ImageURLPart, StepBegin, TextPart
from kimi_cli.loop.message import internal_user_message, system
from kimi_cli.ui.shell.replay import _build_replay_turns_from_history, replay_recent_history


def test_replay_ignores_internal_user_reminders() -> None:
    turns = _build_replay_turns_from_history(
        [
            Message(role="user", content="original task"),
            Message(role="assistant", content="working"),
            internal_user_message(
                [
                    TextPart(
                        text=(
                            "<system-reminder>\n"
                            "The user sent a new reminder during the current turn.\n"
                            "</system-reminder>"
                        )
                    )
                ]
            ),
            internal_user_message([system("Reminder: suggested skill")]),
            Message(role="assistant", content="done"),
        ]
    )

    assert len(turns) == 1
    assert turns[0].user_message.extract_text(" ") == "original task"
    assert turns[0].events == [
        StepBegin(n=1),
        TextPart(text="working"),
        StepBegin(n=2),
        TextPart(text="done"),
    ]


def test_replay_keeps_literal_system_like_user_text() -> None:
    turns = _build_replay_turns_from_history(
        [
            Message(role="user", content="<system>literal user text</system>"),
            Message(role="assistant", content="done"),
        ]
    )

    assert len(turns) == 1
    assert turns[0].user_message.extract_text(" ") == "<system>literal user text</system>"


@pytest.mark.asyncio
async def test_replay_renders_image_marker_literal(monkeypatch) -> None:
    replay_module = importlib.import_module("kimi_cli.ui.shell.replay")
    render_console = Console(record=True, width=80, highlight=False)
    monkeypatch.setattr(replay_module, "console", render_console)

    async def fake_visualize(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(replay_module, "visualize", fake_visualize)

    await replay_recent_history(
        [
            Message(
                role="user",
                content=[
                    TextPart(text='<image path="/tmp/example.png">'),
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(url="https://example.com/image.png")
                    ),
                    TextPart(text="</image>"),
                ],
            ),
            Message(role="assistant", content="done"),
        ]
    )

    assert "[image]" in render_console.export_text()
