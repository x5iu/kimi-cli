from types import SimpleNamespace

from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import FloatContainer

from kimi_cli.soul import StatusSnapshot
from kimi_cli.ui.shell import prompt as shell_prompt
from kimi_cli.ui.shell.prompt import (
    STEADY_INPUT_CURSOR,
    CustomPromptSession,
    InputBoxState,
    PromptMode,
    _build_toolbar_tips,
    _toast_queues,
)


def test_build_toolbar_tips_without_clipboard():
    assert _build_toolbar_tips(clipboard_available=False) == [
        "ctrl-x: toggle mode",
        "shift-tab: plan mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
        "ctrl-l: redraw",
        "@: mention files",
    ]


def test_build_toolbar_tips_with_clipboard():
    assert _build_toolbar_tips(clipboard_available=True) == [
        "ctrl-x: toggle mode",
        "shift-tab: plan mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
        "ctrl-l: redraw",
        "ctrl-v: paste media",
        "@: mention files",
    ]


def test_frame_title_is_prompt() -> None:
    rendered = CustomPromptSession._render_prompt_title()
    plain = "".join(fragment[1] for fragment in rendered)

    assert "PROMPT" in plain


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


def test_prompt_force_turn_full_repaint_resets_last_screen() -> None:
    calls: list[str] = []
    renderer = SimpleNamespace(_last_screen="screen")
    app = SimpleNamespace(renderer=renderer, invalidate=lambda: calls.append("invalidate"))

    shell_prompt.CustomPromptSession._force_turn_full_repaint(app)

    assert renderer._last_screen is None
    assert calls == ["invalidate"]


def test_prompt_hard_redraw_prefers_resize_path() -> None:
    calls: list[str] = []

    class _DummyApp:
        def _on_resize(self) -> None:
            calls.append("resize")

    shell_prompt.CustomPromptSession._hard_redraw(_DummyApp())

    assert calls == ["resize"]


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


def test_active_turn_footer_only_shows_fixed_status() -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)

    rendered = prompt_session._render_turn_footer(
        80,
        status=StatusSnapshot(context_usage=0.0),
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "agent (kimi)" in plain
    assert "Using Shell (make test)" not in plain


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
    refreshed = CustomPromptSession._refresh_turn_application(
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
    refreshed = CustomPromptSession._refresh_turn_application(
        app,
        live_view=SimpleNamespace(needs_periodic_refresh=False),
    )

    assert refreshed is False
    assert invalidate_calls == 0


def test_target_turn_body_bottom_scroll_tracks_latest_output() -> None:
    body_window = SimpleNamespace(render_info=SimpleNamespace(window_height=4, window_width=20))

    assert (
        CustomPromptSession._target_turn_body_bottom_scroll(
            body_window,
            line_count=11,
            current_scroll=0,
        )
        == 7
    )
    assert (
        CustomPromptSession._target_turn_body_bottom_scroll(
            body_window,
            line_count=11,
            current_scroll=7,
        )
        is None
    )


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

    assert len(second_line) <= width
