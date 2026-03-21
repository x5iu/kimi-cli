from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import UIContent, UIControl
from rich.color import Color as RichColor
from rich.color import ColorType
from rich.console import Console as RichConsole
from rich.console import RenderableType
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text as RichText

from kimi_cli.ui.shell.console import RIGHT_PADDING


def _no_cursor_line(_line_count: int) -> int | None:
    return None


def _no_max_line_count(_width: int) -> int | None:
    return None


def _no_window_start(_line_count: int, _visible_count: int) -> int | None:
    return None


def _rich_from_ansi(text: str) -> RichText:
    return RichText.from_ansi(text)


def _rich_color_to_prompt_toolkit(
    color: RichColor | None, *, background: bool = False
) -> str | None:
    if color is None or color.type == ColorType.DEFAULT:
        return None
    truecolor = color.get_truecolor()
    prefix = "bg" if background else "fg"
    return f"{prefix}:#{truecolor.red:02x}{truecolor.green:02x}{truecolor.blue:02x}"


def _rich_style_to_prompt_toolkit(style: RichStyle | None) -> str:
    if style is None:
        return ""
    parts: list[str] = []
    if fg := _rich_color_to_prompt_toolkit(style.color):
        parts.append(fg)
    if bg := _rich_color_to_prompt_toolkit(style.bgcolor, background=True):
        parts.append(bg)
    if style.bold:
        parts.append("bold")
    if style.italic:
        parts.append("italic")
    if style.underline:
        parts.append("underline")
    if style.strike:
        parts.append("strike")
    if style.reverse:
        parts.append("reverse")
    if style.blink:
        parts.append("blink")
    return " ".join(parts)


class _RichRenderableControl(UIControl):
    """Render Rich renderables directly with the actual width assigned by prompt_toolkit."""

    _CACHE_UNSET = object()

    def __init__(
        self,
        get_renderable: Callable[[], RenderableType | None],
        *,
        get_cache_revision: Callable[[], object] | None = None,
    ) -> None:
        self._get_renderable = get_renderable
        self._get_cache_revision = get_cache_revision or (lambda: None)
        self._console = RichConsole(force_terminal=True, color_system="truecolor", highlight=False)
        self._cached_revision: object = self._CACHE_UNSET
        self._rendered_lines_cache: dict[int, tuple[tuple[tuple[str, str], ...], ...]] = {}
        self._content_cache: dict[int, UIContent] = {}

    def _invalidate_cache_if_needed(self) -> None:
        revision = self._get_cache_revision()
        if revision == self._cached_revision:
            return
        self._cached_revision = revision
        self._rendered_lines_cache.clear()
        self._content_cache.clear()

    def _render_lines(self, width: int) -> tuple[tuple[tuple[str, str], ...], ...]:
        normalized_width = max(20, width - RIGHT_PADDING)
        self._invalidate_cache_if_needed()
        cached = self._rendered_lines_cache.get(normalized_width)
        if cached is not None:
            return cached

        renderable = self._get_renderable()
        if renderable is None:
            cached = ()
            self._rendered_lines_cache[normalized_width] = cached
            return cached
        options = self._console.options.update_width(normalized_width)
        lines = self._console.render_lines(renderable, options=options, pad=False, new_lines=False)
        rendered_lines: list[tuple[tuple[str, str], ...]] = []
        for line in lines or [[]]:
            fragments: list[tuple[str, str]] = []
            for segment in Segment.simplify(line):
                if segment.control or not segment.text:
                    continue
                fragments.append((_rich_style_to_prompt_toolkit(segment.style), segment.text))
            rendered_lines.append(tuple(fragments))
        cached = tuple(rendered_lines)
        self._rendered_lines_cache[normalized_width] = cached
        return cached

    def render_lines(self, width: int) -> tuple[tuple[tuple[str, str], ...], ...]:
        return self._render_lines(width)

    def line_count(self, width: int) -> int:
        return len(self._render_lines(width))

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int | None:
        return min(self.line_count(width), max_available_height)

    def create_content(self, width: int, height: int | None) -> UIContent:
        normalized_width = max(20, width)
        self._invalidate_cache_if_needed()
        cached = self._content_cache.get(normalized_width)
        if cached is not None:
            return cached

        key_lines = self._render_lines(normalized_width)
        cached = UIContent(
            get_line=lambda i: list(key_lines[i]) if 0 <= i < len(key_lines) else [],
            line_count=len(key_lines),
            show_cursor=False,
        )
        self._content_cache[normalized_width] = cached
        return cached


class _StackedRichRenderableControl(UIControl):
    """Stack multiple Rich renderables vertically while caching each section separately."""

    def __init__(
        self,
        sections: Sequence[_RichRenderableControl],
        *,
        get_cursor_line: Callable[[int], int | None] | None = None,
        get_max_line_count: Callable[[int], int | None] | None = None,
        get_window_start: Callable[[int, int], int | None] | None = None,
    ) -> None:
        self._sections = list(sections)
        self._get_cursor_line = get_cursor_line or _no_cursor_line
        self._get_max_line_count = get_max_line_count or _no_max_line_count
        self._get_window_start = get_window_start or _no_window_start

    def _visible_window(
        self,
        *,
        width: int,
        line_count: int,
    ) -> tuple[int, int, int | None]:
        if line_count <= 0:
            return 0, 0, None

        max_line_count = self._get_max_line_count(width)
        if max_line_count is None or max_line_count <= 0 or line_count <= max_line_count:
            return 0, line_count, self._get_cursor_line(line_count)

        visible_count = min(line_count, max_line_count)
        max_start = max(0, line_count - visible_count)
        start_override = self._get_window_start(line_count, visible_count)
        if start_override is not None:
            start = max(0, min(max_start, start_override))
            cursor_line = self._get_cursor_line(line_count)
            if cursor_line is None:
                return start, start + visible_count, None
            clamped_cursor = max(start, min(start + visible_count - 1, cursor_line))
            return start, start + visible_count, clamped_cursor - start

        cursor_line = self._get_cursor_line(line_count)
        if cursor_line is None:
            start = line_count - visible_count
            return start, start + visible_count, None

        clamped_cursor = max(0, min(line_count - 1, cursor_line))
        if clamped_cursor <= 0:
            start = 0
        elif clamped_cursor >= line_count - 1:
            start = line_count - visible_count
        else:
            start = max(0, min(clamped_cursor - visible_count // 2, line_count - visible_count))
        return start, start + visible_count, clamped_cursor - start

    def _section_lines(self, width: int) -> list[tuple[tuple[tuple[str, str], ...], ...]]:
        return [section.render_lines(width) for section in self._sections]

    def total_line_count(self, width: int) -> int:
        return sum(len(lines) for lines in self._section_lines(width))

    def line_count(self, width: int) -> int:
        total_lines = self.total_line_count(width)
        start, end, _ = self._visible_window(width=width, line_count=total_lines)
        return end - start

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int | None:
        return min(self.line_count(width), max_available_height)

    def create_content(self, width: int, height: int | None) -> UIContent:
        normalized_width = max(20, width)
        sections = self._section_lines(normalized_width)
        offsets: list[tuple[int, tuple[tuple[tuple[str, str], ...], ...]]] = []
        line_offset = 0
        for lines in sections:
            offsets.append((line_offset, lines))
            line_offset += len(lines)

        visible_start, visible_end, visible_cursor_line = self._visible_window(
            width=normalized_width,
            line_count=line_offset,
        )
        visible_line_count = max(0, visible_end - visible_start)

        def _get_line(i: int) -> StyleAndTextTuples:
            actual_index = visible_start + i
            for start, lines in offsets:
                end = start + len(lines)
                if start <= actual_index < end:
                    return cast(StyleAndTextTuples, list(lines[actual_index - start]))
            return []

        cursor_position = None
        if visible_line_count > 0 and visible_cursor_line is not None:
            cursor_position = Point(
                x=0,
                y=max(0, min(visible_line_count - 1, visible_cursor_line)),
            )

        return UIContent(
            get_line=_get_line,
            line_count=visible_line_count,
            show_cursor=False,
            cursor_position=cursor_position,
        )


RichRenderableControl = _RichRenderableControl
StackedRichRenderableControl = _StackedRichRenderableControl
rich_from_ansi = _rich_from_ansi
rich_style_to_prompt_toolkit = _rich_style_to_prompt_toolkit
