from __future__ import annotations

import re

from rich.console import RenderableType
from rich.text import Text

from kimi_cli.tools.display import DiffDisplayBlock
from kimi_cli.utils.diff import format_unified_diff
from kimi_cli.utils.rich.syntax import KimiSyntax

EDIT_DIFF_LINE_NUMBER_TOOLS = frozenset({"Edit", "StrReplaceFile"})
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_GUTTER_SEPARATOR = " │ "
_GUTTER_STYLE = "bright_black"


def render_diff_block(
    block: DiffDisplayBlock,
    *,
    source_line_numbers: bool = False,
) -> RenderableType:
    diff_text = format_unified_diff(
        block.old_text,
        block.new_text,
        block.path,
        include_file_header=False,
        old_start_line=block.old_start_line,
        new_start_line=block.new_start_line,
    ).rstrip("\n")
    if not source_line_numbers:
        return KimiSyntax(diff_text, "diff")
    return _render_source_numbered_diff(diff_text, block)


def _render_source_numbered_diff(diff_text: str, block: DiffDisplayBlock) -> Text:
    text = Text(no_wrap=False)
    lines = diff_text.splitlines()
    gutter_width = _gutter_width(block)
    current_old = 0
    current_new = 0

    for index, line in enumerate(lines):
        if match := _HUNK_HEADER_RE.match(line):
            current_old = int(match.group("old_start"))
            current_new = int(match.group("new_start"))
            text.append(_format_gutter("", "", gutter_width), style=_GUTTER_STYLE)
            text.append(line, style="cyan")
        elif line.startswith(" "):
            text.append(
                _format_gutter(str(current_old), str(current_new), gutter_width),
                style=_GUTTER_STYLE,
            )
            text.append(line)
            current_old += 1
            current_new += 1
        elif line.startswith("-"):
            text.append(_format_gutter(str(current_old), "", gutter_width), style=_GUTTER_STYLE)
            text.append(line, style="red")
            current_old += 1
        elif line.startswith("+"):
            text.append(_format_gutter("", str(current_new), gutter_width), style=_GUTTER_STYLE)
            text.append(line, style="green")
            current_new += 1
        else:
            text.append(_format_gutter("", "", gutter_width), style=_GUTTER_STYLE)
            text.append(line, style="yellow")

        if index != len(lines) - 1:
            text.append("\n")

    return text


def _gutter_width(block: DiffDisplayBlock) -> int:
    max_line_number = max(
        _max_line_number(block.old_start_line, block.old_text),
        _max_line_number(block.new_start_line, block.new_text),
        0,
    )
    return max(1, len(str(max_line_number)))


def _max_line_number(start_line: int, text: str) -> int:
    if start_line <= 0:
        return 0
    line_count = len(text.splitlines())
    if line_count <= 0:
        return 0
    return start_line + line_count - 1


def _format_gutter(old_line: str, new_line: str, width: int) -> str:
    return f"{old_line:>{width}} {new_line:>{width}}{_GUTTER_SEPARATOR}"
