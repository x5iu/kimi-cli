"""Tests for slash command completer behavior."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import ConditionalContainer, FloatContainer, HSplit, Window
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.utils import get_cwidth

import kimi_cli.ui.shell.completion as completion_mod
from kimi_cli.loop import StatusSnapshot
from kimi_cli.ui.shell.completion import find_prompt_float_container, wrap_to_width
from kimi_cli.ui.shell.prompt import (
    CustomPromptSession,
    SlashCommandCompleter,
    SlashCommandMenuControl,
)
from kimi_cli.utils.slashcmd import SlashCommand


def _noop(app: object, args: str) -> None:
    pass


def _make_command(
    name: str, *, aliases: Iterable[str] = ()
) -> SlashCommand[Callable[[object, str], None]]:
    return SlashCommand(
        name=name,
        description=f"{name} command",
        func=_noop,
        aliases=list(aliases),
    )


def _completion_texts(completer: Completer, text: str) -> list[str]:
    document = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return [completion.text for completion in completer.get_completions(document, event)]


def _completions(completer: SlashCommandCompleter, text: str):
    document = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return list(completer.get_completions(document, event))


def test_exact_command_match_hides_completions():
    """Exact matches should not show completions."""
    completer = SlashCommandCompleter(
        [
            _make_command("mcp"),
            _make_command("mcp-server"),
            _make_command("help", aliases=["h"]),
        ]
    )

    texts = _completion_texts(completer, "/mcp")

    assert not texts


def test_exact_alias_match_hides_completions():
    """Exact alias matches should not show completions."""
    completer = SlashCommandCompleter(
        [
            _make_command("help", aliases=["h"]),
            _make_command("history"),
        ]
    )

    texts = _completion_texts(completer, "/h")

    assert not texts


def test_turn_input_completer_disables_slash_commands():
    session = CustomPromptSession(
        status_provider=lambda: StatusSnapshot(context_usage=0.0),
        model_capabilities=set(),
        model_name=None,
        thinking=False,
        agent_mode_slash_commands=[_make_command("help", aliases=["h"])],
        shell_mode_slash_commands=[_make_command("exit")],
    )

    texts = _completion_texts(session._turn_mode_completer, "/h")

    assert not texts


def test_should_complete_only_for_root_slash_token():
    assert SlashCommandCompleter.should_complete(Document(text="/", cursor_position=1))
    assert SlashCommandCompleter.should_complete(Document(text="  /he", cursor_position=5))
    assert not SlashCommandCompleter.should_complete(Document(text="test /he", cursor_position=8))
    assert not SlashCommandCompleter.should_complete(Document(text="@src", cursor_position=4))
    assert not SlashCommandCompleter.should_complete(Document(text="/he next", cursor_position=8))


def test_completion_display_uses_canonical_command_name():
    completer = SlashCommandCompleter(
        [
            _make_command("help", aliases=["h", "?"]),
            _make_command("history"),
        ]
    )

    completions = _completions(completer, "/he")

    assert len(completions) == 1
    assert completions[0].text == "/help"
    assert completions[0].display_text == "/help"
    assert completions[0].display_meta_text == "help command"


def testwrap_to_width_respects_width():
    lines = wrap_to_width(
        "Help address review issue comments on the open GitHub PR",
        18,
    )

    assert len(lines) > 1
    assert all(get_cwidth(line) <= 18 for line in lines)


def testwrap_to_width_respects_max_lines():
    lines = wrap_to_width(
        "Help address review issue comments on the open GitHub PR for the current branch",
        20,
        max_lines=2,
    )

    assert len(lines) == 2
    assert all(get_cwidth(line) <= 20 for line in lines)
    assert lines[-1].endswith("...")


def test_slash_menu_preserves_unselected_state(monkeypatch):
    completions = [
        Completion(
            text="/model",
            start_position=0,
            display="/model",
            display_meta="Switch LLM model or thinking mode",
        ),
        Completion(
            text="/exit",
            start_position=0,
            display="/exit",
            display_meta="Exit the application",
        ),
    ]
    complete_state = SimpleNamespace(completions=completions, complete_index=None)
    app = SimpleNamespace(current_buffer=SimpleNamespace(complete_state=complete_state))
    monkeypatch.setattr(completion_mod, "get_app_or_none", lambda: app)

    control = SlashCommandMenuControl(left_padding=lambda: 0)
    content = control.create_content(width=80, height=6)

    rendered_lines = [
        "".join(fragment[1] for fragment in content.get_line(i)) for i in range(content.line_count)
    ]

    assert content.line_count == 6
    assert content.cursor_position.y == 0
    assert "›" not in rendered_lines[1]
    assert "›" not in rendered_lines[2]
    assert "Switch" in rendered_lines[1]
    assert rendered_lines[1].count("/model") == 1
    assert rendered_lines[-1].strip() == ""


def test_completion_menu_uses_full_width_when_meta_missing() -> None:
    completions = [
        Completion(
            text="src/kimi_cli/ui/shell/prompt.py",
            start_position=0,
            display="src/kimi_cli/ui/shell/prompt.py",
            display_meta="",
        )
    ]
    control = SlashCommandMenuControl(left_padding=lambda: 0)

    command_width = control._command_column_width(
        completions,
        menu_width=40,
        marker_width=2,
        has_meta=False,
    )

    assert command_width == 38


def test_completion_menu_height_is_stable_for_selected_description(monkeypatch) -> None:
    completions = [
        Completion(
            text="/model",
            start_position=0,
            display="/model",
            display_meta="Switch LLM model or thinking mode and save preference",
        ),
        Completion(
            text="/exit",
            start_position=0,
            display="/exit",
            display_meta="Exit the application",
        ),
    ]
    control = SlashCommandMenuControl(left_padding=lambda: 0)

    selected_app = SimpleNamespace(
        current_buffer=SimpleNamespace(
            complete_state=SimpleNamespace(completions=completions, complete_index=0)
        )
    )
    unselected_app = SimpleNamespace(
        current_buffer=SimpleNamespace(
            complete_state=SimpleNamespace(completions=completions, complete_index=None)
        )
    )

    monkeypatch.setattr(completion_mod, "get_app_or_none", lambda: selected_app)
    selected_height = control.preferred_height(80, 10, False, None)
    selected_content = control.create_content(width=80, height=6)

    monkeypatch.setattr(completion_mod, "get_app_or_none", lambda: unselected_app)
    unselected_height = control.preferred_height(80, 10, False, None)
    unselected_content = control.create_content(width=80, height=6)

    assert selected_height == 10
    assert unselected_height == 10
    assert selected_content.line_count == 6
    assert unselected_content.line_count == 6
    detail_line = "".join(fragment[1] for fragment in selected_content.get_line(5))
    assert "╰─" in detail_line
    assert "Switch" in detail_line


def testfind_prompt_float_container_supports_conditional_container_shape():
    float_container = FloatContainer(content=Window(), floats=[])
    root = HSplit(
        [
            ConditionalContainer(
                content=Window(),
                filter=True,
                alternative_content=float_container,
            )
        ]
    )

    assert find_prompt_float_container(root) is float_container


def testfind_prompt_float_container_supports_direct_float_container_shape():
    float_container = FloatContainer(content=Window(), floats=[])
    root = HSplit([float_container])

    assert find_prompt_float_container(root) is float_container


def test_prompt_session_wraps_root_layout_for_slash_menu(
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
        agent_mode_slash_commands=[_make_command("help")],
        shell_mode_slash_commands=[],
    )

    root_container = prompt_session._session.layout.container

    assert isinstance(root_container, FloatContainer)
    assert isinstance(root_container.content, HSplit)
    assert root_container.floats

    slash_float = root_container.floats[0]
    assert slash_float.left == 0
    assert slash_float.right == 0
    assert slash_float.ycursor is True

    inner_float_container = find_prompt_float_container(root_container.content)
    assert isinstance(inner_float_container, FloatContainer)
    assert not any(
        isinstance(float_.content, CompletionsMenu) for float_ in inner_float_container.floats
    )
    assert any(
        isinstance(float_.content, ConditionalContainer)
        and isinstance(float_.content.content, CompletionsMenu)
        for float_ in inner_float_container.floats
    )


def test_prompt_application_uses_inline_slash_menu(
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
        agent_mode_slash_commands=[_make_command("help")],
        shell_mode_slash_commands=[],
    )

    app, _ = prompt_session._build_prompt_application()

    slash_windows = [
        container
        for container in app.layout.walk()
        if isinstance(container, Window) and isinstance(container.content, SlashCommandMenuControl)
    ]

    assert len(slash_windows) == 1
