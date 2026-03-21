from __future__ import annotations

from prompt_toolkit.data_structures import Point
from rich.style import Style
from rich.text import Text

from kimi_cli.ui.shell.panels import QuestionRequestPanel
from kimi_cli.ui.shell.rich_ptk import _RichRenderableControl, _StackedRichRenderableControl
from kimi_cli.utils.rich.diff import render_diff_block
from kimi_cli.utils.rich.markdown import Markdown
from kimi_cli.wire.types import DiffDisplayBlock, QuestionItem, QuestionOption, QuestionRequest


def _section(text: str) -> _RichRenderableControl:
    return _RichRenderableControl(lambda: Text(text))


def _content_lines(content) -> list[str]:
    return [
        "".join(fragment[1] for fragment in content.get_line(i)) for i in range(content.line_count)
    ]


def test_stacked_rich_renderable_control_exposes_bottom_cursor_for_autoscroll() -> None:
    control = _StackedRichRenderableControl(
        [_section("first"), _section("second")],
        get_cursor_line=lambda line_count: line_count - 1,
    )

    content = control.create_content(width=80, height=None)

    assert content.line_count == 2
    assert content.cursor_position == Point(x=0, y=1)


def test_stacked_rich_renderable_control_clamps_cursor_line() -> None:
    control = _StackedRichRenderableControl(
        [_section("only line")],
        get_cursor_line=lambda _line_count: 99,
    )

    content = control.create_content(width=80, height=None)

    assert content.line_count == 1
    assert content.cursor_position == Point(x=0, y=0)


def test_stacked_rich_renderable_control_can_limit_visible_lines_from_bottom() -> None:
    control = _StackedRichRenderableControl(
        [_section("1\n2\n3"), _section("4\n5")],
        get_cursor_line=lambda line_count: line_count - 1,
        get_max_line_count=lambda _width: 3,
    )

    content = control.create_content(width=80, height=None)

    assert content.line_count == 3
    assert content.cursor_position == Point(x=0, y=2)
    assert _content_lines(content) == ["3", "4", "5"]


def test_stacked_rich_renderable_control_can_limit_visible_lines_from_top() -> None:
    control = _StackedRichRenderableControl(
        [_section("1\n2\n3"), _section("4\n5")],
        get_cursor_line=lambda _line_count: 0,
        get_max_line_count=lambda _width: 2,
    )

    content = control.create_content(width=80, height=None)

    assert content.line_count == 2
    assert content.cursor_position == Point(x=0, y=0)
    assert _content_lines(content) == ["1", "2"]


def test_stacked_rich_renderable_control_can_pin_visible_window_to_custom_start() -> None:
    control = _StackedRichRenderableControl(
        [_section("1\n2\n3\n4\n5")],
        get_cursor_line=lambda _line_count: 1,
        get_max_line_count=lambda _width: 2,
        get_window_start=lambda _line_count, _visible_count: 1,
    )

    content = control.create_content(width=80, height=None)

    assert content.line_count == 2
    assert content.cursor_position == Point(x=0, y=0)
    assert _content_lines(content) == ["2", "3"]


def test_stacked_rich_renderable_control_reports_total_line_count() -> None:
    control = _StackedRichRenderableControl(
        [_section("1\n2\n3"), _section("4\n5")],
        get_cursor_line=lambda line_count: line_count - 1,
        get_max_line_count=lambda _width: 2,
    )

    assert control.total_line_count(80) == 5
    assert control.line_count(80) == 2


def test_source_numbered_diff_uses_soft_colors() -> None:
    renderable = render_diff_block(
        DiffDisplayBlock(
            path="src/example.py",
            old_text="before\n",
            new_text="after\n",
            old_start_line=42,
            new_start_line=42,
        ),
        source_line_numbers=True,
    )
    control = _RichRenderableControl(lambda: renderable)
    content = control.create_content(width=80, height=None)
    lines = [content.get_line(i) for i in range(content.line_count)]

    deleted_line = next(
        line for line in lines if any("-before" in fragment[1] for fragment in line)
    )
    inserted_line = next(
        line for line in lines if any("+after" in fragment[1] for fragment in line)
    )

    assert any(
        "-before" in fragment[1] and fragment[0] == "fg:#ff8a8a" for fragment in deleted_line
    )
    assert any(
        "+after" in fragment[1] and fragment[0] == "fg:#8fcd8f" for fragment in inserted_line
    )


def test_unified_diff_syntax_uses_soft_colors() -> None:
    renderable = render_diff_block(
        DiffDisplayBlock(
            path="src/example.py",
            old_text="before\n",
            new_text="after\n",
        ),
        source_line_numbers=False,
    )
    control = _RichRenderableControl(lambda: renderable)
    content = control.create_content(width=80, height=None)
    lines = [content.get_line(i) for i in range(content.line_count)]

    deleted_line = next(
        line for line in lines if any("-before" in fragment[1] for fragment in line)
    )
    inserted_line = next(
        line for line in lines if any("+after" in fragment[1] for fragment in line)
    )

    assert any(
        "-before" in fragment[1] and fragment[0] == "fg:#ff8a8a" for fragment in deleted_line
    )
    assert any(
        "+after" in fragment[1] and fragment[0] == "fg:#8fcd8f" for fragment in inserted_line
    )


def test_rich_style_omits_default_terminal_colors() -> None:
    control = _RichRenderableControl(lambda: Text("default", style=Style(color="default")))
    content = control.create_content(width=80, height=None)
    line = content.get_line(0)

    assert any(fragment[1] == "default" and fragment[0] == "" for fragment in line)


def test_rich_style_keeps_explicit_colors_but_drops_default_background() -> None:
    control = _RichRenderableControl(
        lambda: Text("color", style=Style.parse("#ff0000 on default bold"))
    )
    content = control.create_content(width=80, height=None)
    line = content.get_line(0)

    assert any(fragment[1] == "color" and fragment[0] == "fg:#ff0000 bold" for fragment in line)


def test_markdown_code_block_keeps_default_terminal_foreground() -> None:
    renderable = Markdown("```sh\nmake format\nmake check\nmake test\n```")
    control = _RichRenderableControl(lambda: renderable)
    content = control.create_content(width=80, height=None)
    lines = [content.get_line(i) for i in range(content.line_count)]

    assert any("make format" in fragment[1] and fragment[0] == "" for fragment in lines[0])
    assert any("make check" in fragment[1] and fragment[0] == "" for fragment in lines[1])
    assert any("make " in fragment[1] and fragment[0] == "" for fragment in lines[2])
    assert not any(fragment[0] == "fg:#000000" for line in lines for fragment in line)
    assert not any("bg:" in fragment[0] for line in lines for fragment in line)


def test_question_body_markdown_code_block_uses_shared_fix() -> None:
    panel = QuestionRequestPanel(
        QuestionRequest(
            id="question-markdown-code",
            tool_call_id="tool-question-markdown-code",
            questions=[
                QuestionItem(
                    question="Which command should I run?",
                    options=[QuestionOption(label="Continue")],
                    body="```sh\nmake format\nmake check\nmake test\n```",
                )
            ],
        )
    )
    renderable = panel.render_full_body()[0]
    control = _RichRenderableControl(lambda: renderable)
    content = control.create_content(width=80, height=None)
    lines = [content.get_line(i) for i in range(content.line_count)]

    assert any("make format" in fragment[1] and fragment[0] == "" for fragment in lines[0])
    assert any("make check" in fragment[1] and fragment[0] == "" for fragment in lines[1])
    assert not any(fragment[0] == "fg:#000000" for line in lines for fragment in line)
