from __future__ import annotations

from kosong.message import Message

from kimi_cli.soul.message import system
from kimi_cli.ui.shell.replay import _build_replay_turns_from_history
from kimi_cli.wire.types import StepBegin, TextPart


def test_replay_ignores_internal_user_reminders() -> None:
    turns = _build_replay_turns_from_history(
        [
            Message(role="user", content="original task"),
            Message(role="assistant", content="working"),
            Message(
                role="user",
                content=[
                    TextPart(
                        text=(
                            "<system-reminder>\n"
                            "The user sent a new reminder during the current turn.\n"
                            "</system-reminder>"
                        )
                    )
                ],
            ),
            Message(role="user", content=[system("Reminder: suggested skill")]),
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
