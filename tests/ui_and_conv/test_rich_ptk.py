from __future__ import annotations

from prompt_toolkit.data_structures import Point
from rich.text import Text

from kimi_cli.ui.shell.rich_ptk import _RichRenderableControl, _StackedRichRenderableControl


def _section(text: str) -> _RichRenderableControl:
    return _RichRenderableControl(lambda: Text(text))


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
