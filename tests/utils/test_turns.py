from __future__ import annotations

from kimi_cli.eventbus.types import TextPart
from kimi_cli.loop.message import INTERNAL_USER_NAME, internal_user_message, system
from kimi_cli.utils.turns import (
    is_internal_user_message,
    is_internal_user_record,
    is_real_user_turn_start_message,
    is_real_user_turn_start_record,
)
from llmkit.message import Message


def test_named_internal_user_message_is_not_real_turn() -> None:
    message = internal_user_message([TextPart(text="hidden")])

    assert message.name == INTERNAL_USER_NAME
    assert is_internal_user_message(message)
    assert not is_real_user_turn_start_message(message)


def test_legacy_internal_messages_still_do_not_start_real_turns() -> None:
    messages = [
        Message(
            role="user",
            content=[
                TextPart(
                    text="<system-reminder>\nPlan mode is active. Keep reading only.\n</system-reminder>"
                )
            ],
        ),
        Message(
            role="user",
            content=[
                TextPart(
                    text=(
                        "<system-reminder>\n"
                        "Plan mode still active (see full instructions earlier).\n"
                        "</system-reminder>"
                    )
                )
            ],
        ),
        Message(role="user", content=[system("The user has imported context from session 's1'.")]),
        Message(
            role="user",
            content=[system("Previous context has been compacted. Here is the compaction output:")],
        ),
        Message(role="user", content=[system("You just got a D-Mail from your future self.")]),
    ]

    for message in messages:
        assert is_internal_user_message(message)
        assert not is_real_user_turn_start_message(message)


def test_legacy_internal_records_still_do_not_start_real_turns() -> None:
    records = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<system>The user has imported context from session 's1'.</system>",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<system-reminder>\n"
                        "Plan mode still active (see full instructions earlier).\n"
                        "</system-reminder>"
                    ),
                }
            ],
        },
    ]

    for record in records:
        assert is_internal_user_record(record)
        assert not is_real_user_turn_start_record(record)


def test_literal_system_like_user_text_is_still_real_turn() -> None:
    message = Message(role="user", content=[TextPart(text="<system>literal user text</system>")])
    record = {"role": "user", "content": "<system>literal user text</system>"}

    assert not is_internal_user_message(message)
    assert is_real_user_turn_start_message(message)
    assert not is_internal_user_record(record)
    assert is_real_user_turn_start_record(record)
