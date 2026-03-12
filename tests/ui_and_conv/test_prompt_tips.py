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
        "@: mention files",
    ]


def test_build_toolbar_tips_with_clipboard():
    assert _build_toolbar_tips(clipboard_available=True) == [
        "ctrl-x: toggle mode",
        "shift-tab: plan mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
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


def test_route_live_navigation_when_panel_open_and_buffer_empty() -> None:
    live_view = SimpleNamespace(has_pending_input_request=True, input_mode="question")

    assert CustomPromptSession._should_route_live_navigation(live_view, "") is True
    assert CustomPromptSession._should_route_live_navigation(live_view, "  ") is True
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


def test_prompt_session_disables_terminal_size_polling(
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

    assert prompt_session._session.app.terminal_size_polling_interval is None


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


def test_active_turn_footer_includes_fixed_running_indicator() -> None:
    prompt_session = object.__new__(CustomPromptSession)
    prompt_session._mode = PromptMode.AGENT
    prompt_session._model_name = "kimi"
    prompt_session._thinking = False
    prompt_session._status_provider = lambda: StatusSnapshot(context_usage=0.0)

    rendered = prompt_session._render_turn_footer(
        80,
        status=StatusSnapshot(context_usage=0.0),
        live_view=SimpleNamespace(footer_indicator=("tool", "Using Shell (make test)")),
    )
    plain = "".join(fragment[1] for fragment in rendered)

    assert "agent (kimi)" in plain
    assert "Using Shell (make test)" in plain


def test_refresh_turn_application_forces_periodic_full_repaint() -> None:
    invalidate_calls = 0

    class _Renderer:
        def __init__(self) -> None:
            self._last_screen = object()
            self.erase_calls = 0
            self.reset_calls = 0

        def erase(self, *, leave_alternate_screen: bool = True) -> None:
            self.erase_calls += 1

        def reset(self) -> None:
            self.reset_calls += 1

    class _App:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    live_view = SimpleNamespace(needs_periodic_refresh=True)

    last_repaint = CustomPromptSession._refresh_turn_application(
        app,
        live_view=live_view,
        last_full_repaint_at=None,
        now=10.0,
    )
    assert last_repaint == 10.0
    assert app.renderer._last_screen is not None
    assert app.renderer.erase_calls == 0
    assert app.renderer.reset_calls == 0
    assert invalidate_calls == 1

    last_repaint = CustomPromptSession._refresh_turn_application(
        app,
        live_view=live_view,
        last_full_repaint_at=last_repaint,
        now=10.5,
    )
    assert last_repaint == 10.0
    assert app.renderer._last_screen is not None
    assert app.renderer.erase_calls == 0
    assert app.renderer.reset_calls == 0
    assert invalidate_calls == 2

    last_repaint = CustomPromptSession._refresh_turn_application(
        app,
        live_view=live_view,
        last_full_repaint_at=last_repaint,
        now=11.1,
    )
    assert last_repaint == 11.1
    assert app.renderer._last_screen is None
    assert app.renderer.erase_calls == 0
    assert app.renderer.reset_calls == 0
    assert invalidate_calls == 3


def test_refresh_turn_application_stops_repainting_when_idle() -> None:
    invalidate_calls = 0

    class _Renderer:
        def __init__(self) -> None:
            self._last_screen = object()
            self.erase_calls = 0
            self.reset_calls = 0

        def erase(self, *, leave_alternate_screen: bool = True) -> None:
            self.erase_calls += 1

        def reset(self) -> None:
            self.reset_calls += 1

    class _App:
        def __init__(self) -> None:
            self.renderer = _Renderer()

        def invalidate(self) -> None:
            nonlocal invalidate_calls
            invalidate_calls += 1

    app = _App()
    last_repaint = CustomPromptSession._refresh_turn_application(
        app,
        live_view=SimpleNamespace(needs_periodic_refresh=False),
        last_full_repaint_at=10.0,
        now=10.5,
    )

    assert last_repaint is None
    assert app.renderer._last_screen is not None
    assert app.renderer.erase_calls == 0
    assert app.renderer.reset_calls == 0
    assert invalidate_calls == 0


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
