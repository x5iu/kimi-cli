import asyncio
import contextlib
from types import SimpleNamespace
from typing import cast

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import FloatContainer
from prompt_toolkit.utils import get_cwidth

from kimi_cli.soul import StatusSnapshot
from kimi_cli.ui.shell import prompt as shell_prompt
from kimi_cli.ui.shell.prompt import (
    STEADY_INPUT_CURSOR,
    CustomPromptSession,
    InputBoxState,
    PromptMode,
    TurnSubmitResult,
    _build_toolbar_tips,
    _toast_queues,
)
from kimi_cli.ui.shell.visualize import LiveView
from kimi_cli.wire.types import StatusUpdate, TextPart, ThinkPart, ToolCallPart


def test_build_toolbar_tips_without_clipboard():
    assert _build_toolbar_tips(clipboard_available=False) == [
        "ctrl-x: toggle mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
        "ctrl-l: redraw",
        "ctrl-y: history",
        "@: mention files",
    ]


def test_build_toolbar_tips_with_clipboard():
    assert _build_toolbar_tips(clipboard_available=True) == [
        "ctrl-x: toggle mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
        "ctrl-l: redraw",
        "ctrl-y: history",
        "ctrl-v: paste clipboard",
        "@: mention files",
    ]


def test_shell_style_dict_includes_completion_detail_styles() -> None:
    style = shell_prompt._shell_style_dict()

    assert style["slash-completion-menu.detail.prefix"]
    assert style["slash-completion-menu.detail"]


def test_frame_title_is_prompt() -> None:
    rendered = CustomPromptSession._render_prompt_title()
    plain = "".join(fragment[1] for fragment in rendered)

    assert "PROMPT" in plain


def test_history_view_position_formatter_handles_empty_and_nonempty() -> None:
    assert CustomPromptSession._format_history_view_position(top_line=0, total_lines=0) == "0/0"
    assert CustomPromptSession._format_history_view_position(top_line=3, total_lines=10) == "3/10"
    assert CustomPromptSession._format_history_view_position(top_line=99, total_lines=10) == "10/10"


def test_history_view_hint_includes_position_and_controls() -> None:
    hint = CustomPromptSession._format_history_view_hint(top_line=3, total_lines=10)

    assert "3/10" in hint
    assert "PgUp/PgDn" in hint
    assert "Home/End" in hint
    assert "Ctrl-Y" in hint


def test_live_turn_hint_can_include_recent_output_notice() -> None:
    hint = CustomPromptSession._format_live_turn_hint(show_recent_output_notice=True)

    assert "Recent output only" in hint
    assert "Ctrl-Y" in hint


def test_live_turn_hint_can_combine_input_hint_with_recent_output_notice() -> None:
    hint = CustomPromptSession._format_live_turn_hint(
        input_hint="Use ↑/↓ to focus and Enter to choose.",
        show_recent_output_notice=True,
    )

    assert "Use ↑/↓" in hint
    assert "Ctrl-Y" in hint
    assert "Recent output only" in hint


def test_turn_prompt_title_reflects_pending_input_mode() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    rendered = prompt_session._render_turn_prompt_title(
        SimpleNamespace(has_pending_input_request=True, input_mode="approval")
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "APPROVAL" in plain


def test_render_message_shows_active_turn_badge() -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._input_box_state_provider = lambda: InputBoxState(
        active=True,
        mode="reminder",
        hint="Turn is running.",
    )

    rendered = prompt_session._render_message()
    plain = "".join(fragment[1] for fragment in rendered)

    assert "REMINDER" in plain


def test_active_turn_input_box_uses_steady_cursor() -> None:
    assert STEADY_INPUT_CURSOR.cursor_shape == CursorShape.BEAM


def test_rich_from_ansi_strips_escape_sequences() -> None:
    rendered = shell_prompt._rich_from_ansi("\x1b[31mhello\x1b[0m")

    assert rendered.plain == "hello"


def test_route_live_navigation_when_question_panel_open() -> None:
    live_view = SimpleNamespace(has_pending_input_request=True, input_mode="question")

    assert CustomPromptSession._should_route_live_navigation(live_view, "") is True
    assert CustomPromptSession._should_route_live_navigation(live_view, "  ") is True
    assert CustomPromptSession._should_route_live_navigation(live_view, "1") is False
    assert CustomPromptSession._should_route_live_navigation(live_view, "/more") is False


def test_do_not_route_approval_navigation_when_buffer_has_text() -> None:
    live_view = SimpleNamespace(has_pending_input_request=True, input_mode="approval")

    assert CustomPromptSession._should_route_live_navigation(live_view, "") is True
    assert CustomPromptSession._should_route_live_navigation(live_view, "1") is False


def test_do_not_route_live_navigation_in_custom_answer_mode() -> None:
    live_view = SimpleNamespace(has_pending_input_request=True, input_mode="question_other")

    assert CustomPromptSession._should_route_live_navigation(live_view, "") is False


def test_prompt_application_shows_completion_menu(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    app, _ = prompt_session._build_prompt_application()

    assert isinstance(app.layout.container, FloatContainer)
    assert len(app.layout.container.floats) >= 1


def test_prompt_session_uses_slow_terminal_size_polling(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )

    assert (
        prompt_session._session.app.terminal_size_polling_interval
        == shell_prompt._TERMINAL_SIZE_POLLING_INTERVAL
    )


@pytest.mark.asyncio
async def test_run_turn_ui_avoids_recent_output_hint_recursion(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    live_view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False, allow_expand=True)
    for i in range(shell_prompt.MAX_ACTIVE_TURN_FLUSHED_BLOCKS + 1):
        live_view.echo_reminder(f"reminder {i}")

    async def _fake_run_async(self) -> None:
        return None

    class _FakeWire:
        async def receive(self):
            await asyncio.Future()

    printed: list[object] = []
    monkeypatch.setattr(shell_prompt.Application, "run_async", _fake_run_async)
    monkeypatch.setattr(shell_prompt, "patch_stdout", lambda raw=True: contextlib.nullcontext())
    monkeypatch.setattr(shell_prompt.console, "print", lambda *args, **kwargs: printed.append(args))

    await prompt_session.run_turn_ui(
        wire=_FakeWire(),
        live_view=live_view,
        submit_handler=lambda _input: TurnSubmitResult.accept(),
        cancel_handler=lambda: None,
        turn_prompt="hello",
    )

    assert printed


def test_prompt_force_turn_full_repaint_resets_last_screen() -> None:
    calls: list[str] = []
    renderer = SimpleNamespace(_last_screen="screen")
    app = SimpleNamespace(renderer=renderer, invalidate=lambda: calls.append("invalidate"))

    shell_prompt.CustomPromptSession._force_turn_full_repaint(app)

    assert renderer._last_screen is None
    assert calls == ["invalidate"]


def test_prompt_hard_redraw_prefers_renderer_erase() -> None:
    calls: list[tuple[str, object | None]] = []

    class _Renderer:
        _last_screen = "screen"

        def erase(self, leave_alternate_screen: bool = True) -> None:
            calls.append(("erase", leave_alternate_screen))

    class _DummyApp:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            calls.append(("invalidate", None))

    shell_prompt.CustomPromptSession._hard_redraw(_DummyApp())

    assert calls == [("erase", False), ("invalidate", None)]


def test_prompt_redraw_for_layout_change_triggers_full_repaint() -> None:
    calls: list[str] = []
    renderer = SimpleNamespace(_last_screen="screen")
    app = SimpleNamespace(renderer=renderer, invalidate=lambda: calls.append("invalidate"))

    result = shell_prompt.CustomPromptSession._redraw_for_layout_change(
        app,
        signature=("agent", 80, False, 1),
        last_signature=("agent", 80, True, 2),
    )

    assert renderer._last_screen is None
    assert calls == ["invalidate"]
    assert result == ("agent", 80, False, 1)


def test_turn_ui_refresh_interval_only_bursts_for_visible_text_stream() -> None:
    composing_view = SimpleNamespace(activity_indicator=("composing", "Composing..."))
    thinking_view = SimpleNamespace(activity_indicator=("thinking", "Thinking..."))

    assert (
        CustomPromptSession._turn_ui_refresh_interval(composing_view)
        == shell_prompt._TURN_UI_FAST_REFRESH_INTERVAL
    )
    assert (
        CustomPromptSession._turn_ui_refresh_interval(composing_view, burst_active=True)
        == shell_prompt._TURN_UI_BURST_REFRESH_INTERVAL
    )
    assert (
        CustomPromptSession._turn_ui_refresh_interval(thinking_view)
        == shell_prompt._TURN_UI_FAST_REFRESH_INTERVAL
    )
    assert (
        CustomPromptSession._turn_ui_refresh_interval(thinking_view, burst_active=True)
        == shell_prompt._TURN_UI_FAST_REFRESH_INTERVAL
    )
    assert (
        CustomPromptSession._turn_ui_refresh_interval(
            SimpleNamespace(activity_indicator=("tool", "Using Shell")),
            burst_active=True,
        )
        == shell_prompt._TURN_UI_REFRESH_INTERVAL
    )
    assert (
        CustomPromptSession._turn_ui_refresh_interval(
            SimpleNamespace(activity_indicator=("approval", "Awaiting approval..."))
        )
        == shell_prompt._TURN_UI_REFRESH_INTERVAL
    )


def test_turn_stream_push_thresholds_prioritize_visible_text_stream() -> None:
    composing_view = SimpleNamespace(activity_indicator=("composing", "Composing..."))

    assert CustomPromptSession._turn_stream_push_thresholds(
        SimpleNamespace(activity_indicator=("thinking", "Thinking..."))
    ) == (
        shell_prompt._TURN_UI_FAST_STREAM_PUSH_PARTS,
        shell_prompt._TURN_UI_FAST_STREAM_PUSH_CHARS,
    )
    assert CustomPromptSession._turn_stream_push_thresholds(composing_view) == (
        shell_prompt._TURN_UI_COMPOSE_STREAM_PUSH_PARTS,
        shell_prompt._TURN_UI_COMPOSE_STREAM_PUSH_CHARS,
    )
    assert CustomPromptSession._turn_stream_push_thresholds(
        composing_view,
        burst_active=True,
    ) == (
        shell_prompt._TURN_UI_BURST_STREAM_PUSH_PARTS,
        shell_prompt._TURN_UI_BURST_STREAM_PUSH_CHARS,
    )
    assert CustomPromptSession._turn_stream_push_thresholds(
        SimpleNamespace(activity_indicator=("tool", "Using Shell")),
        burst_active=True,
    ) == (
        shell_prompt._TURN_UI_STREAM_PUSH_PARTS,
        shell_prompt._TURN_UI_STREAM_PUSH_CHARS,
    )


def test_turn_stream_refresh_metrics_count_visible_stream_updates() -> None:
    assert CustomPromptSession._turn_stream_refresh_metrics(TextPart(text="hello")) == (1, 5, False)
    assert CustomPromptSession._turn_stream_refresh_metrics(ThinkPart(think="a\nb")) == (1, 3, True)
    assert CustomPromptSession._turn_stream_refresh_metrics(
        ToolCallPart(arguments_part='{"command":"echo hi"}')
    ) == (1, len('{"command":"echo hi"}'), False)
    assert CustomPromptSession._turn_stream_refresh_metrics(ToolCallPart(arguments_part="")) == (
        0,
        0,
        False,
    )


def test_turn_stream_burst_kind_only_treats_visible_text_as_burstable() -> None:
    assert CustomPromptSession._turn_stream_burst_kind(TextPart(text="hello")) == "composing"
    assert CustomPromptSession._turn_stream_burst_kind(ThinkPart(think="thought")) is None
    assert (
        CustomPromptSession._turn_stream_burst_kind(
            ToolCallPart(arguments_part='{"command":"echo hi"}')
        )
        is None
    )


def test_should_start_turn_stream_burst_for_new_or_resumed_visible_text() -> None:
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="composing",
            msg_kind="composing",
            current_burst_kind=None,
            now=10.0,
            burst_until=0.0,
            last_burstable_stream_at=0.0,
        )
        is True
    )
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="composing",
            msg_kind="composing",
            current_burst_kind="composing",
            now=10.0,
            burst_until=10.2,
            last_burstable_stream_at=9.95,
        )
        is False
    )
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="composing",
            msg_kind="composing",
            current_burst_kind="composing",
            now=10.0,
            burst_until=9.7,
            last_burstable_stream_at=10.0 - shell_prompt._TURN_UI_STREAM_RESUME_BURST_GAP - 0.01,
        )
        is True
    )


def test_should_not_start_turn_stream_burst_for_non_visible_or_too_soon_resume() -> None:
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="thinking",
            msg_kind="composing",
            current_burst_kind=None,
            now=10.0,
            burst_until=0.0,
            last_burstable_stream_at=0.0,
        )
        is False
    )
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="composing",
            msg_kind=None,
            current_burst_kind="composing",
            now=10.0,
            burst_until=9.7,
            last_burstable_stream_at=9.0,
        )
        is False
    )
    assert (
        CustomPromptSession._should_start_turn_stream_burst(
            active_kind="composing",
            msg_kind="composing",
            current_burst_kind="composing",
            now=10.0,
            burst_until=9.7,
            last_burstable_stream_at=10.0 - shell_prompt._TURN_UI_STREAM_RESUME_BURST_GAP + 0.01,
        )
        is False
    )


def test_custom_prompt_app_uses_slow_terminal_size_polling(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    app, _ = prompt_session._build_prompt_application()

    assert app.terminal_size_polling_interval == shell_prompt._TERMINAL_SIZE_POLLING_INTERVAL


def test_custom_prompt_app_clears_rendered_input_when_done(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    app, _ = prompt_session._build_prompt_application()

    assert app.erase_when_done is True


def test_prompt_application_cache_reuses_top_level_app(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )

    first_app, first_text_area = prompt_session._get_prompt_application()
    second_app, second_text_area = prompt_session._get_prompt_application()

    assert first_app is second_app
    assert first_text_area is second_text_area


def test_prepare_prompt_application_clears_buffer_and_applies_mode(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )

    app, text_area = prompt_session._get_prompt_application()
    text_area.buffer.document = shell_prompt.Document(text="stale", cursor_position=5)
    prompt_session._mode = PromptMode.SHELL

    prepared_app, prepared_text_area = prompt_session._prepare_prompt_application()

    assert prepared_app is app
    assert prepared_text_area is text_area
    assert prepared_text_area.buffer.text == ""
    assert prepared_text_area.buffer.completer is prompt_session._shell_mode_completer


def test_prepare_prompt_application_uses_hard_redraw(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )

    app, _ = prompt_session._get_prompt_application()
    redraw_calls: list[object] = []
    monkeypatch.setattr(
        prompt_session, "_hard_redraw", lambda target_app: redraw_calls.append(target_app)
    )

    prompt_session._prepare_prompt_application()

    assert redraw_calls == [app]


def test_custom_prompt_app_binds_ctrl_l_to_redraw(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    app, _ = prompt_session._build_prompt_application()

    assert app.key_bindings is not None
    bindings = app.key_bindings.get_bindings_for_keys((Keys.ControlL,))
    assert bindings


def test_custom_prompt_app_binds_ctrl_c_to_keyboard_interrupt(
    temp_work_dir,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    prompt_session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[],
        shell_mode_slash_commands=[],
    )
    app, _ = prompt_session._build_prompt_application()

    assert app.key_bindings is not None
    bindings = app.key_bindings.get_bindings_for_keys((Keys.ControlC,))
    assert bindings


def test_dismiss_input_clears_buffer_text_and_completion() -> None:
    class _DummyBuffer:
        def __init__(self) -> None:
            self.text = "pending input"
            self.complete_state = object()
            self.document = shell_prompt.Document(text=self.text, cursor_position=len(self.text))

        def cancel_completion(self) -> None:
            self.complete_state = None

        @property
        def document(self):
            return self._document

        @document.setter
        def document(self, value) -> None:
            self._document = value
            self.text = value.text

    buffer = _DummyBuffer()

    handled = CustomPromptSession._dismiss_input(cast(Buffer, buffer))

    assert handled is True
    assert buffer.text == ""
    assert buffer.complete_state is None


def test_dismiss_input_returns_false_when_nothing_to_clear() -> None:
    class _DummyBuffer:
        text = ""
        complete_state = None

        def cancel_completion(self) -> None:
            raise AssertionError("cancel_completion should not be called")

    assert CustomPromptSession._dismiss_input(cast(Buffer, _DummyBuffer())) is False


def test_turn_input_hint_text_shows_pending_question_hint() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    hint = prompt_session._turn_input_hint_text(
        live_view=SimpleNamespace(
            input_mode="question",
            input_hint="Use ↑/↓ to focus and Enter to choose.",
        ),
        buffer_text="",
        feedback_message="",
    )

    assert "Use ↑/↓" in hint


def test_turn_input_hint_text_keeps_pending_question_hint_while_typing() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    hint = prompt_session._turn_input_hint_text(
        live_view=SimpleNamespace(
            input_mode="question",
            input_hint="Use ↑/↓ to focus and Enter to choose.",
        ),
        buffer_text="typing",
        feedback_message="",
    )

    assert "Use ↑/↓" in hint


def test_turn_input_hint_text_still_hides_reminder_hint_while_typing() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    hint = prompt_session._turn_input_hint_text(
        live_view=SimpleNamespace(
            input_mode="reminder",
            input_hint="Turn is running. Type a message and press Enter to send a reminder.",
        ),
        buffer_text="typing",
        feedback_message="",
    )

    assert hint == ""


def test_turn_body_cursor_line_pins_pending_questions_to_top_until_output_reveal() -> None:
    assert (
        CustomPromptSession._turn_body_cursor_line(
            0,
            has_pending_input_request=True,
            reveal_latest_output=False,
        )
        is None
    )
    assert (
        CustomPromptSession._turn_body_cursor_line(
            7,
            has_pending_input_request=True,
            reveal_latest_output=False,
        )
        == 0
    )
    assert (
        CustomPromptSession._turn_body_cursor_line(
            7,
            has_pending_input_request=True,
            reveal_latest_output=True,
        )
        == 6
    )
    assert (
        CustomPromptSession._turn_body_cursor_line(
            7,
            has_pending_input_request=False,
            reveal_latest_output=False,
        )
        == 6
    )
    assert (
        CustomPromptSession._turn_body_cursor_line(
            7,
            has_pending_input_request=False,
            reveal_latest_output=True,
            history_view_enabled=True,
        )
        == 0
    )


def test_immediate_toast_replaces_older_messages() -> None:
    _toast_queues["left"].clear()

    shell_prompt.toast("older toast", position="left")
    shell_prompt.toast("new toast", position="left", immediate=True)

    assert len(_toast_queues["left"]) == 1
    toast = shell_prompt._current_toast("left")
    assert toast is not None
    assert toast.message == "new toast"


def test_render_toast_line_shows_left_and_right_toasts(monkeypatch) -> None:
    width = 80
    prompt_session = object.__new__(CustomPromptSession)

    class _DummyOutput:
        @staticmethod
        def get_size():
            return SimpleNamespace(columns=width)

    dummy_app = SimpleNamespace(output=_DummyOutput())
    monkeypatch.setattr(shell_prompt, "get_app_or_none", lambda: dummy_app)
    _toast_queues["left"].clear()
    _toast_queues["right"].clear()
    shell_prompt.toast("left toast", position="left", immediate=True)
    shell_prompt.toast("right toast", position="right", immediate=True)

    rendered = prompt_session._render_toast_line()
    plain = "".join(fragment[1] for fragment in rendered)

    assert "left toast" in plain
    assert "right toast" in plain
    assert get_cwidth(plain) <= width


def test_refresh_turn_application_repaints_for_visible_toast() -> None:
    invalidate_calls = 0
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._has_toasts = lambda: True

    class _App:
        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    refreshed = prompt_session._refresh_turn_application(
        _App(),
        live_view=SimpleNamespace(needs_periodic_refresh=False),
    )

    assert refreshed is True
    assert invalidate_calls == 1


def test_refresh_turn_application_clears_expired_toast() -> None:
    invalidate_calls = 0
    prompt_session = object.__new__(CustomPromptSession)
    states = iter([True, False])
    prompt_session._has_toasts = lambda: next(states)

    class _App:
        def __init__(self) -> None:
            self._kimi_had_toasts = False

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()

    assert (
        prompt_session._refresh_turn_application(
            app,
            live_view=SimpleNamespace(needs_periodic_refresh=False),
        )
        is True
    )
    assert (
        prompt_session._refresh_turn_application(
            app,
            live_view=SimpleNamespace(needs_periodic_refresh=False),
        )
        is True
    )
    assert invalidate_calls == 2


def test_reminder_mode_hides_hint_line(monkeypatch) -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.1)
    prompt_session._tips = []
    prompt_session._tip_rotation_index = 0
    prompt_session._input_box_state_provider = lambda: InputBoxState(
        active=True,
        mode="reminder",
        hint="Turn is running. Type a message and press Enter to send a reminder.",
    )

    class _DummyBuffer:
        text = ""

    class _DummyTextArea:
        buffer = _DummyBuffer()

    assert prompt_session._render_hint_line(_DummyTextArea()) == ""


def test_active_turn_input_box_renders_static_hint(monkeypatch) -> None:
    width = 80
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.1)
    prompt_session._tips = []
    prompt_session._tip_rotation_index = 0
    prompt_session._input_box_state_provider = lambda: InputBoxState(
        active=True,
        mode="reminder",
        hint="Type a message and press Enter to send a reminder.",
    )

    class _DummyOutput:
        @staticmethod
        def get_size():
            return SimpleNamespace(columns=width)

    dummy_app = SimpleNamespace(output=_DummyOutput())
    monkeypatch.setattr(shell_prompt, "get_app_or_none", lambda: dummy_app)
    _toast_queues["left"].clear()
    _toast_queues["right"].clear()

    message = "".join(fragment[1] for fragment in prompt_session._render_message())
    toolbar = "".join(fragment[1] for fragment in prompt_session._render_bottom_toolbar())
    placeholder = "".join(fragment[1] for fragment in prompt_session._render_placeholder())
    rprompt = "".join(fragment[1] for fragment in prompt_session._render_rprompt())

    assert "╭─" in message
    assert "│ " in message
    assert "╰─" in toolbar
    assert "send a reminder" not in toolbar
    assert "send a reminder" in placeholder
    assert rprompt == "│"


def test_active_turn_footer_shows_working_directory() -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._working_dir_provider = lambda: "/tmp/project"

    rendered = prompt_session._render_turn_footer(
        80,
        status=StatusSnapshot(context_usage=0.0),
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "agent (kimi)" in plain
    assert "/tmp/project" in plain
    assert "Using Shell (make test)" not in plain


def test_bottom_toolbar_shows_working_directory(monkeypatch) -> None:
    width = 100
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._working_dir_provider = lambda: "/tmp/project"
    prompt_session._tips = []
    prompt_session._tip_rotation_index = 0
    prompt_session._input_box_state_provider = lambda: InputBoxState()

    class _DummyOutput:
        @staticmethod
        def get_size():
            return SimpleNamespace(columns=width)

    dummy_app = SimpleNamespace(output=_DummyOutput())
    monkeypatch.setattr(shell_prompt, "get_app_or_none", lambda: dummy_app)
    _toast_queues["left"].clear()
    _toast_queues["right"].clear()

    plain = "".join(fragment[1] for fragment in prompt_session._render_bottom_toolbar())
    second_line = plain.split("\n", 1)[1]

    assert "/tmp/project" in second_line
    assert get_cwidth(second_line) <= width


def test_bottom_toolbar_handles_cjk_working_directory_width(monkeypatch) -> None:
    width = 40
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._working_dir_provider = lambda: "/tmp/项目/子目录/更深的目录"
    prompt_session._tips = []
    prompt_session._tip_rotation_index = 0
    prompt_session._input_box_state_provider = lambda: InputBoxState()

    class _DummyOutput:
        @staticmethod
        def get_size():
            return SimpleNamespace(columns=width)

    dummy_app = SimpleNamespace(output=_DummyOutput())
    monkeypatch.setattr(shell_prompt, "get_app_or_none", lambda: dummy_app)
    _toast_queues["left"].clear()
    _toast_queues["right"].clear()

    plain = "".join(fragment[1] for fragment in prompt_session._render_bottom_toolbar())
    second_line = plain.split("\n", 1)[1]

    assert "录" in second_line
    assert get_cwidth(second_line) <= width


def test_turn_footer_handles_cjk_working_directory_width() -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._working_dir_provider = lambda: "/tmp/项目/子目录/更深的目录"

    rendered = prompt_session._render_turn_footer(
        40,
        status=StatusSnapshot(context_usage=0.0),
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "录" in plain
    assert get_cwidth(plain) <= 40


def test_active_turn_activity_line_shows_running_indicator() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    rendered = prompt_session._render_turn_activity(
        SimpleNamespace(
            has_pending_input_request=False,
            activity_indicator=("tool", "Using Shell (make test)"),
        )
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "Using Shell (make test)" in plain


def test_active_turn_activity_line_hides_while_waiting_for_input() -> None:
    prompt_session = object.__new__(CustomPromptSession)

    assert (
        prompt_session._render_turn_activity(
            SimpleNamespace(
                has_pending_input_request=True,
                activity_indicator=("question", "Awaiting answer..."),
            )
        )
        == ""
    )


def test_rich_renderable_control_tracks_width_and_line_count() -> None:
    calls: list[str] = []
    control = shell_prompt._RichRenderableControl(lambda: calls.append("render") or "head\nbody")

    content = control.create_content(80, None)

    assert calls == ["render"]
    assert content.line_count == 2
    assert control.line_count(80) == 2
    assert calls == ["render"]


def test_rich_renderable_control_invalidates_cache_when_revision_changes() -> None:
    revision = 0
    calls: list[int] = []

    def _render() -> str:
        calls.append(revision)
        return "head\nbody"

    control = shell_prompt._RichRenderableControl(
        _render,
        get_cache_revision=lambda: revision,
    )

    first_content = control.create_content(80, None)
    assert first_content.line_count == 2
    assert control.line_count(80) == 2
    assert calls == [0]

    revision = 1
    second_content = control.create_content(80, None)
    assert second_content.line_count == 2
    assert control.line_count(80) == 2
    assert calls == [0, 1]


def test_stacked_rich_renderable_control_rerenders_only_changed_section() -> None:
    revision_a = 0
    revision_b = 0
    calls: list[tuple[str, int]] = []

    control = shell_prompt._StackedRichRenderableControl(
        [
            shell_prompt._RichRenderableControl(
                lambda: calls.append(("a", revision_a)) or "head",
                get_cache_revision=lambda: revision_a,
            ),
            shell_prompt._RichRenderableControl(
                lambda: calls.append(("b", revision_b)) or "body",
                get_cache_revision=lambda: revision_b,
            ),
        ]
    )

    content = control.create_content(80, None)
    assert content.line_count == 2
    assert calls == [("a", 0), ("b", 0)]

    revision_b = 1
    content = control.create_content(80, None)
    assert content.line_count == 2
    assert calls == [("a", 0), ("b", 0), ("b", 1)]


def test_rich_style_to_prompt_toolkit_maps_basic_styles() -> None:
    style = shell_prompt._rich_style_to_prompt_toolkit(
        shell_prompt.RichStyle.parse("bold italic underline #ff0000 on #0000ff")
    )

    assert "fg:#ff0000" in style
    assert "bg:#0000ff" in style
    assert "bold" in style
    assert "italic" in style
    assert "underline" in style


def test_force_turn_full_repaint_resets_renderer_cache() -> None:
    invalidate_calls = 0

    class _Renderer:
        def __init__(self) -> None:
            self._last_screen = object()

    class _App:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    CustomPromptSession._force_turn_full_repaint(app)

    assert app.renderer._last_screen is None
    assert invalidate_calls == 1


def test_redraw_for_layout_change_can_skip_full_repaint() -> None:
    invalidate_calls = 0

    class _Renderer:
        def __init__(self) -> None:
            self._last_screen = object()

    class _App:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    signature = ("body", 10)

    updated = CustomPromptSession._redraw_for_layout_change(
        app,
        signature=signature,
        last_signature=("body", 9),
        full_repaint_on_change=False,
    )

    assert updated == signature
    assert app.renderer._last_screen is not None
    assert invalidate_calls == 1


def test_refresh_turn_application_uses_incremental_redraw() -> None:
    invalidate_calls = 0

    class _Renderer:
        def __init__(self) -> None:
            self._last_screen = object()

    class _App:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._has_toasts = lambda: False
    refreshed = prompt_session._refresh_turn_application(
        app,
        live_view=SimpleNamespace(needs_periodic_refresh=True),
    )

    assert refreshed is True
    assert app.renderer._last_screen is not None
    assert invalidate_calls == 1


def test_refresh_turn_application_stops_repainting_when_idle() -> None:
    invalidate_calls = 0

    class _App:
        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._has_toasts = lambda: False
    refreshed = prompt_session._refresh_turn_application(
        app,
        live_view=SimpleNamespace(needs_periodic_refresh=False),
    )

    assert refreshed is False
    assert invalidate_calls == 0


def test_append_history_entry_updates_in_memory_history(tmp_path) -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._history_file = tmp_path / "history.jsonl"
    prompt_session._history = shell_prompt.InMemoryHistory()
    prompt_session._last_history_content = None

    prompt_session._append_history_entry("hello")
    prompt_session._append_history_entry("hello")

    assert list(prompt_session._history.get_strings()) == ["hello"]
    assert prompt_session._last_history_content == "hello"
    assert prompt_session._history_file.read_text(encoding="utf-8").count("hello") == 1


def test_bottom_toolbar_no_overflow_when_tip_would_exactly_fill_old_available(monkeypatch) -> None:
    width = 60
    mode_text = "agent"
    right_text = CustomPromptSession._render_right_span(StatusSnapshot(context_usage=0.0))
    tip_len = width - len(mode_text) - len(right_text) - 4

    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = None
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)
    prompt_session._tips = ["t" * tip_len]
    prompt_session._tip_rotation_index = 0
    prompt_session._input_box_state_provider = lambda: InputBoxState()

    class _DummyOutput:
        @staticmethod
        def get_size():
            return SimpleNamespace(columns=width)

    dummy_app = SimpleNamespace(output=_DummyOutput())
    monkeypatch.setattr(shell_prompt, "get_app_or_none", lambda: dummy_app)
    _toast_queues["left"].clear()
    _toast_queues["right"].clear()

    rendered = prompt_session._render_bottom_toolbar()
    plain = "".join(fragment[1] for fragment in rendered)
    second_line = plain.split("\n", 1)[1]

    assert get_cwidth(second_line) <= width
