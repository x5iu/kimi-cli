from __future__ import annotations

import pytest

from kimi_cli.ui.shell.visualize import LiveView
from kimi_cli.wire.types import (
    ApprovalRequest,
    QuestionItem,
    QuestionOption,
    QuestionRequest,
    StatusUpdate,
    StepBegin,
    TextPart,
    TurnBegin,
    TurnEnd,
)


@pytest.mark.asyncio
async def test_live_view_accepts_line_based_approval_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    assert "/more" not in view.input_hint
    request = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="WriteFile",
        action="edit files",
        description="Apply the requested changes.",
    )

    view.request_approval(request)

    assert view.has_pending_input_request is True
    assert view.input_mode == "approval"
    assert view.try_submit_line("2") is True
    assert await request.wait() == "approve_for_session"


@pytest.mark.asyncio
async def test_live_view_accepts_custom_single_select_answer() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-1",
        tool_call_id="tool-1",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
            )
        ],
    )

    view.request_question(request)

    assert view.input_mode == "question"
    assert view.try_submit_line("TOML") is True
    assert await request.wait() == {"Which format should I use?": "TOML"}


@pytest.mark.asyncio
async def test_live_view_switches_to_custom_answer_mode_for_other_option() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-other",
        tool_call_id="tool-other",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
            )
        ],
    )

    view.request_question(request)

    assert view.try_submit_line("3") is True
    assert view.input_mode == "question_other"
    assert view.try_submit_line("TOML") is True
    assert await request.wait() == {"Which format should I use?": "TOML"}


@pytest.mark.asyncio
async def test_live_view_accepts_multi_select_answer_with_custom_text() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-2",
        tool_call_id="tool-2",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                multi_select=True,
                options=[
                    QuestionOption(label="format"),
                    QuestionOption(label="lint"),
                    QuestionOption(label="tests"),
                ],
            )
        ],
    )

    view.request_question(request)

    assert view.try_submit_line("1, 3, smoke") is True
    assert await request.wait() == {"Which checks should I run?": "format, tests, smoke"}


def test_live_view_echoes_reminder_in_output() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.echo_reminder("please keep the answer short")
    rendered = view.render_ansi(80)

    assert "Reminder:" in rendered
    assert "please keep the answer short" in rendered

    view.dispatch_wire_message(StepBegin(n=2))
    rendered = view.render_ansi(80)

    assert "Reminder:" in rendered
    assert "please keep the answer short" in rendered


def test_live_view_keeps_turn_spinner_as_fallback_until_turn_end() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.dispatch_wire_message(TurnBegin(user_input="hello"))
    assert view.needs_periodic_refresh is True
    assert "Running..." in view.render_ansi(80)

    view.dispatch_wire_message(TextPart(text="working"))
    assert "Running..." not in view.render_ansi(80)

    view.flush_content()
    assert "Running..." in view.render_ansi(80)

    view.dispatch_wire_message(StepBegin(n=1))
    assert "Running..." not in view.render_ansi(80)

    view.dispatch_wire_message(TurnEnd())
    assert "Running..." not in view.render_ansi(80)
