from __future__ import annotations

import importlib

import pytest

from kimi_cli.ui.shell.keyboard import KeyEvent
from kimi_cli.ui.shell.visualize import LiveView
from kimi_cli.wire.types import (
    ApprovalRequest,
    QuestionItem,
    QuestionOption,
    QuestionRequest,
    SkillReminderNotice,
    StatusUpdate,
    StepBegin,
    TextPart,
    ThinkPart,
    ToolCall,
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
async def test_live_view_switches_to_custom_answer_mode_for_keyboard_selected_other() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-other",
        tool_call_id="tool-keyboard-other",
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
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert view.input_mode == "question_other"
    assert view.try_submit_line("TOML") is True
    assert await request.wait() == {"Which format should I use?": "TOML"}


@pytest.mark.asyncio
async def test_live_view_expand_hint_mentions_ctrl_e() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-expand-hint",
        tool_call_id="tool-expand-hint",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
                body="Long details",
            )
        ],
    )

    view.request_question(request)

    assert "Ctrl-E" in view.input_hint
    assert "/more" in view.input_hint


def test_live_view_ctrl_e_expands_current_panel(monkeypatch) -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-expand-key",
        tool_call_id="tool-expand-key",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
                body="Long details",
            )
        ],
    )
    view.request_question(request)
    calls: list[str] = []
    monkeypatch.setattr(view, "show_more", lambda: calls.append("expand") or True)

    view.dispatch_keyboard_event(KeyEvent.CTRL_E)

    assert calls == ["expand"]


def test_live_view_enter_toggles_multi_select_option_before_submit() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-multi-toggle",
        tool_call_id="tool-keyboard-multi-toggle",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                multi_select=True,
                options=[
                    QuestionOption(label="format"),
                    QuestionOption(label="lint"),
                ],
            )
        ],
    )

    view.request_question(request)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    panel = view._current_question_panel
    assert panel is not None
    assert panel._multi_selected == {0}
    assert view.input_mode == "question"
    assert request.resolved is False


@pytest.mark.asyncio
async def test_live_view_enter_submits_multi_select_when_current_option_already_checked() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-multi-submit",
        tool_call_id="tool-keyboard-multi-submit",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                multi_select=True,
                options=[
                    QuestionOption(label="format"),
                    QuestionOption(label="lint"),
                ],
            )
        ],
    )

    view.request_question(request)
    view.dispatch_keyboard_event(KeyEvent.ENTER)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert request.resolved is True


@pytest.mark.asyncio
async def test_live_view_switches_to_custom_answer_mode_for_keyboard_multi_select_other_without_precheck() -> (
    None
):
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-multi-other-no-space",
        tool_call_id="tool-keyboard-multi-other-no-space",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                multi_select=True,
                options=[
                    QuestionOption(label="format"),
                    QuestionOption(label="lint"),
                ],
            )
        ],
    )

    view.request_question(request)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert view.input_mode == "question_other"
    assert view.try_submit_line("smoke") is True
    assert await request.wait() == {"Which checks should I run?": "smoke"}


@pytest.mark.asyncio
async def test_live_view_switches_to_custom_answer_mode_for_keyboard_multi_select_other() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-multi-other",
        tool_call_id="tool-keyboard-multi-other",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                multi_select=True,
                options=[
                    QuestionOption(label="format"),
                    QuestionOption(label="lint"),
                ],
            )
        ],
    )

    view.request_question(request)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.SPACE)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert view.input_mode == "question_other"
    assert view.try_submit_line("smoke") is True
    assert await request.wait() == {"Which checks should I run?": "smoke"}


@pytest.mark.asyncio
async def test_live_view_accepts_keyboard_multi_select_submission() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-keyboard-multi-select",
        tool_call_id="tool-keyboard-multi-select",
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
    view.dispatch_keyboard_event(KeyEvent.SPACE)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.SPACE)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert await request.wait() == {"Which checks should I run?": "format, tests"}


@pytest.mark.asyncio
async def test_live_view_accepts_multi_select_line_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-2-line-input",
        tool_call_id="tool-2-line-input",
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

    assert view.try_submit_line("1, 3") is True
    assert await request.wait() == {"Which checks should I run?": "format, tests"}


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
    view.dispatch_keyboard_event(KeyEvent.ENTER)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.DOWN)
    view.dispatch_keyboard_event(KeyEvent.ENTER)

    assert view.input_mode == "question_other"
    assert view.try_submit_line("smoke") is True
    assert await request.wait() == {"Which checks should I run?": "format, smoke"}


def test_live_view_echoes_reminder_in_output() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.echo_reminder("please keep the answer short")
    rendered = view.render_ansi(80)

    assert "Reminder" in rendered
    assert "╭" in rendered
    assert "please keep the answer short" in rendered
    assert "Reminder" in view.render_reminders_ansi(80)
    assert "please keep the answer short" in view.render_reminders_ansi(80)

    view.dispatch_wire_message(StepBegin(n=2))
    rendered = view.render_ansi(80)

    assert "Reminder" in rendered
    assert "╭" in rendered
    assert "please keep the answer short" in rendered
    assert view.has_reminders is True


def test_live_view_renders_reminder_inline_with_body_output() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="before reminder"))
    view.echo_reminder("please keep the answer short")
    view.append_content(TextPart(text="after reminder"))

    rendered = view.render_ansi(80, include_running_indicators=False)

    assert rendered.index("before reminder") < rendered.index("Reminder")
    assert rendered.index("Reminder") < rendered.index("after reminder")


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


def test_live_view_pauses_periodic_refresh_while_waiting_for_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.dispatch_wire_message(TurnBegin(user_input="hello"))
    view.append_content(TextPart(text="still streaming"))
    assert view.needs_periodic_refresh is True

    view.request_question(
        QuestionRequest(
            id="question-pause-refresh",
            tool_call_id="tool-pause-refresh",
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
    )

    assert view.needs_periodic_refresh is False


def test_live_view_compose_body_keeps_recent_blocks_while_waiting_for_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="older block"))
    view.flush_content()
    view.append_content(TextPart(text="recent block"))
    view.flush_content()
    view.request_question(
        QuestionRequest(
            id="question-show-recent-history",
            tool_call_id="tool-show-recent-history",
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
    )

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=1,
        ),
        80,
    )

    assert "older block" not in rendered
    assert "recent block" in rendered
    assert "Which format should I use?" in rendered
    assert "Recent output only" in rendered


def test_live_view_bumps_render_revision_for_same_height_content_updates() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    assert view.render_revision == 0

    view.dispatch_wire_message(TextPart(text="ab"))
    first_revision = view.render_revision
    assert first_revision > 0

    view.dispatch_wire_message(TextPart(text="cd"))
    assert view.render_revision == first_revision + 1
    assert "abcd" in view.render_ansi(80, include_running_indicators=False)


def test_live_view_refresh_soon_bumps_render_revision() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.echo_reminder("keep going")
    first_revision = view.render_revision
    assert first_revision > 0

    view.finish_turn()
    assert view.render_revision == first_revision

    view.dispatch_wire_message(TurnBegin(user_input="hello"))
    assert view.render_revision == first_revision + 1

    view.finish_turn()
    assert view.render_revision == first_revision + 2


def test_live_view_reports_activity_indicator_for_running_states() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.dispatch_wire_message(TurnBegin(user_input="hello"))
    assert view.activity_indicator == ("running", "Running...")

    view.dispatch_wire_message(ThinkPart(think="analyzing"))
    assert view.activity_indicator == ("thinking", "Thinking...")

    view.dispatch_wire_message(StepBegin(n=1))
    assert view.activity_indicator == ("moon", "Running...")

    view.dispatch_wire_message(TurnEnd())
    assert view.activity_indicator is None


def test_live_view_activity_indicator_highlights_ready_todo_execute_target() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_tool_call(
        ToolCall(
            id="call-ready-todo",
            function=ToolCall.FunctionBody(
                name="SetTodoList",
                arguments=(
                    '{"todos":[{"title":"Inspect parser","status":"pending",'
                    '"executor":"task","subagent_name":"coder"},'
                    '{"title":"Share findings","status":"pending"}]}'
                ),
            ),
        )
    )

    assert view.activity_indicator == (
        "tool",
        "Updating Todo List (ready: Inspect parser @coder)",
    )


def test_live_view_can_hide_running_indicators_in_body() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.dispatch_wire_message(TurnBegin(user_input="hello"))
    assert "Running..." not in view.render_ansi(80, include_running_indicators=False)

    view.dispatch_wire_message(ThinkPart(think="analyzing"))
    rendered = view.render_ansi(80, include_running_indicators=False)
    assert "Thinking..." not in rendered
    assert "analyzing" in rendered

    view.dispatch_wire_message(StepBegin(n=1))
    assert "Running..." not in view.render_ansi(80, include_running_indicators=False)


def test_live_view_accepts_expand_command_for_question(monkeypatch: pytest.MonkeyPatch) -> None:
    visualize_module = importlib.import_module("kimi_cli.ui.shell.visualize")
    shown: list[str] = []
    monkeypatch.setattr(
        visualize_module,
        "_show_question_body_in_pager",
        lambda panel: shown.append(panel.current_question_text),
    )

    view = LiveView(StatusUpdate(context_usage=0.0))
    request = QuestionRequest(
        id="question-expand-open",
        tool_call_id="tool-expand-open",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
                body="Long details",
            )
        ],
    )

    view.request_question(request)

    assert view.try_submit_line("/more") is True
    assert shown == ["Which format should I use?"]
    assert request.resolved is False


def test_live_view_hides_expand_prompts_when_expansion_disabled() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request = QuestionRequest(
        id="question-expand",
        tool_call_id="tool-expand",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[
                    QuestionOption(label="JSON"),
                    QuestionOption(label="YAML"),
                ],
                body="Long details",
            )
        ],
    )

    view.request_question(request)
    rendered = view.render_ansi(80)

    assert "/more" not in rendered
    assert view.try_submit_line("/more") is False


def test_live_view_compose_body_can_limit_to_recent_blocks() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="older block"))
    view.flush_content()
    view.append_content(TextPart(text="recent block"))
    view.flush_content()

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=1,
        ),
        80,
    )

    assert "older block" not in rendered
    assert "recent block" in rendered
    assert "Recent output only" in rendered


def test_live_view_tail_mode_notice_stays_visible_without_actual_truncation() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="only block"))
    view.flush_content()

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=10,
        ),
        80,
    )

    assert "only block" in rendered
    assert "Ctrl-Y" in rendered
    assert "Ctrl-Y for history" in rendered


def test_live_view_compose_body_can_hide_previous_blocks_while_waiting_for_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="older block"))
    view.flush_content()
    view.request_question(
        QuestionRequest(
            id="question-hide-history",
            tool_call_id="tool-hide-history",
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
    )

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=0,
        ),
        80,
    )

    assert "older block" not in rendered
    assert "Which format should I use?" in rendered
    assert "Recent output only" in rendered


def test_live_view_compose_body_hides_streaming_content_while_waiting_for_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="streaming block"))
    view.request_question(
        QuestionRequest(
            id="question-hide-streaming",
            tool_call_id="tool-hide-streaming",
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
    )

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=0,
        ),
        80,
    )

    assert "streaming block" not in rendered
    assert "Which format should I use?" in rendered
    assert "Recent output only" in rendered


def test_live_view_compose_active_body_can_reveal_streaming_content_behind_question() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="streaming block"))
    view.request_question(
        QuestionRequest(
            id="question-reveal-streaming",
            tool_call_id="tool-reveal-streaming",
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
    )

    body = view.compose_active_body(
        include_running_indicators=False,
        focus_pending_input_panel=False,
    )
    assert body is not None
    rendered = view._renderable_to_ansi(body, 80)

    assert "streaming block" in rendered
    assert "Which format should I use?" not in rendered


def test_live_view_compose_body_hides_tool_calls_while_waiting_for_input() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_tool_call(
        ToolCall(
            id="call-hide-tool",
            function=ToolCall.FunctionBody(
                name="Shell",
                arguments='{"command": "echo hidden"}',
            ),
        )
    )
    view.request_question(
        QuestionRequest(
            id="question-hide-tool",
            tool_call_id="tool-hide-tool",
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
    )

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            tail_block_limit=0,
        ),
        80,
    )

    assert "Using Shell" not in rendered
    assert "echo hidden" not in rendered
    assert "Which format should I use?" in rendered
    assert "Recent output only" in rendered


def test_live_view_compose_body_can_limit_current_content_to_tail_chars() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.append_content(TextPart(text="prefix-" * 50 + "tail-end"))

    rendered = view._renderable_to_ansi(
        view.compose_body(
            include_running_indicators=False,
            content_char_limit=16,
        ),
        80,
    )

    assert "tail-end" in rendered
    assert "prefix-prefix-prefix" not in rendered
    assert "Recent output only" in rendered


def test_live_view_renders_skill_reminder_notice() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)

    view.dispatch_wire_message(SkillReminderNotice(skills=["/skill:gen-docs"]))
    rendered = view.render_ansi(80)

    assert "Reminder" in rendered
    assert "╭" in rendered
    assert "/skill:gen-docs" in rendered
    assert "main flow" in rendered


@pytest.mark.asyncio
async def test_live_view_approve_for_session_resolves_matching_queued_approvals() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request1 = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="WriteFile",
        action="edit files",
        description="Apply the requested changes.",
    )
    request2 = ApprovalRequest(
        id="req-2",
        tool_call_id="tool-2",
        sender="Edit",
        action="edit files",
        description="Apply the requested changes.",
    )
    request3 = ApprovalRequest(
        id="req-3",
        tool_call_id="tool-3",
        sender="Shell",
        action="run command",
        description="Run command.",
    )

    view.request_approval(request1)
    view.request_approval(request2)
    view.request_approval(request3)

    assert view.try_submit_line("2") is True
    assert await request1.wait() == "approve_for_session"
    assert await request2.wait() == "approve_for_session"
    assert request3.resolved is False
    assert view._current_approval_request_panel is not None
    assert view._current_approval_request_panel.request is request3

    view.cleanup(is_interrupt=False)
    assert await request3.wait() == "reject"


@pytest.mark.asyncio
async def test_live_view_rejects_queued_and_following_approvals() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request1 = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="WriteFile",
        action="edit files",
        description="Apply the requested changes.",
    )
    request2 = ApprovalRequest(
        id="req-2",
        tool_call_id="tool-2",
        sender="Edit",
        action="edit files",
        description="Apply the requested changes.",
    )
    request3 = ApprovalRequest(
        id="req-3",
        tool_call_id="tool-3",
        sender="Shell",
        action="run command",
        description="Run command.",
    )

    view.request_approval(request1)
    view.request_approval(request2)

    assert view.try_submit_line("3") is True
    assert await request1.wait() == "reject"
    assert await request2.wait() == "reject"

    view.request_approval(request3)
    assert await request3.wait() == "reject"


@pytest.mark.asyncio
async def test_live_view_cleanup_rejects_current_and_queued_approvals() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request1 = ApprovalRequest(
        id="req-1",
        tool_call_id="tool-1",
        sender="WriteFile",
        action="edit files",
        description="Apply the requested changes.",
    )
    request2 = ApprovalRequest(
        id="req-2",
        tool_call_id="tool-2",
        sender="Shell",
        action="run command",
        description="Run command.",
    )

    view.request_approval(request1)
    view.request_approval(request2)
    view.cleanup(is_interrupt=True)

    assert await request1.wait() == "reject"
    assert await request2.wait() == "reject"


@pytest.mark.asyncio
async def test_live_view_advances_to_next_queued_question_request() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request1 = QuestionRequest(
        id="question-1",
        tool_call_id="tool-1",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[QuestionOption(label="JSON"), QuestionOption(label="YAML")],
            )
        ],
    )
    request2 = QuestionRequest(
        id="question-2",
        tool_call_id="tool-2",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                options=[QuestionOption(label="format"), QuestionOption(label="lint")],
            )
        ],
    )

    view.request_question(request1)
    view.request_question(request2)

    assert view.try_submit_line("1") is True
    assert await request1.wait() == {"Which format should I use?": "JSON"}
    assert view._current_question_panel is not None
    assert view._current_question_panel.request is request2
    assert view.has_pending_input_request is True

    assert view.try_submit_line("2") is True
    assert await request2.wait() == {"Which checks should I run?": "lint"}
    assert view.has_pending_input_request is False


@pytest.mark.asyncio
async def test_live_view_cleanup_resolves_current_and_queued_questions() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), allow_expand=False)
    request1 = QuestionRequest(
        id="question-1",
        tool_call_id="tool-1",
        questions=[
            QuestionItem(
                question="Which format should I use?",
                options=[QuestionOption(label="JSON"), QuestionOption(label="YAML")],
            )
        ],
    )
    request2 = QuestionRequest(
        id="question-2",
        tool_call_id="tool-2",
        questions=[
            QuestionItem(
                question="Which checks should I run?",
                options=[QuestionOption(label="format"), QuestionOption(label="lint")],
            )
        ],
    )

    view.request_question(request1)
    view.request_question(request2)
    view.cleanup(is_interrupt=True)

    assert await request1.wait() == {}
    assert await request2.wait() == {}
    assert view.has_pending_input_request is False
