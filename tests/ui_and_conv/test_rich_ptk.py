from __future__ import annotations

from prompt_toolkit.data_structures import Point
from rich.text import Text

from kimi_cli.ui.shell.rich_ptk import _RichRenderableControl, _StackedRichRenderableControl
from kimi_cli.utils.rich.diff import render_diff_block
from kimi_cli.wire.types import DiffDisplayBlock


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

    deleted_line = next(line for line in lines if any("-before" in text for _, text in line))
    inserted_line = next(line for line in lines if any("+after" in text for _, text in line))

    assert any("-before" in text and style == "fg:#ff8a8a" for style, text in deleted_line)
    assert any("+after" in text and style == "fg:#8fcd8f" for style, text in inserted_line)


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

    deleted_line = next(line for line in lines if any("-before" in text for _, text in line))
    inserted_line = next(line for line in lines if any("+after" in text for _, text in line))

    assert any("-before" in text and style == "fg:#ff8a8a" for style, text in deleted_line)
    assert any("+after" in text and style == "fg:#8fcd8f" for style, text in inserted_line)
