from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from typing import Any, cast

import streamingjson  # pyright: ignore[reportMissingTypeStubs]
from rich.console import Group, RenderableType
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

try:
    from markdown_it import MarkdownIt as _MarkdownIt

    _HAS_MARKDOWN_IT = True
except ImportError:  # pragma: no cover
    _HAS_MARKDOWN_IT = False

from kimi_cli.eventbus.types import (
    BackgroundTaskDisplayBlock,
    BriefDisplayBlock,
    ContentPart,
    DiffDisplayBlock,
    StatusUpdate,
    TextPart,
    TodoDisplayBlock,
    TodoDisplayItem,
    ToolCall,
    ToolReturnValue,
)
from kimi_cli.loop import format_context_status
from kimi_cli.tools import extract_key_argument
from kimi_cli.tools.todo_text import todo_label
from kimi_cli.tools.utils import truncate_line
from kimi_cli.utils.rich.columns import BulletColumns
from kimi_cli.utils.rich.diff import SOURCE_LINE_NUMBER_DIFF_TOOLS, render_diff_block
from kimi_cli.utils.rich.markdown import Markdown

MAX_TOOL_ERROR_OUTPUT_LINES = 12
MAX_TOOL_ERROR_OUTPUT_CHARS = 4000
MAX_TOOL_OUTPUT_TAIL_LINES = 8
MAX_TOOL_OUTPUT_TAIL_CHARS = 4000
MAX_TOOL_OUTPUT_LINE_CHARS = 400
MAX_FILE_CONTENT_DISPLAY_LINES = 12
MAX_FILE_CONTENT_DISPLAY_CHARS = 4000
_FILE_CONTENT_TOOLS = frozenset({"ReadFile"})
_OUTPUT_GUTTER_SEPARATOR = " │ "
_OUTPUT_GUTTER_STYLE = "bright_black"
_OUTPUT_STDOUT_STYLE = "#b8b8b8"
_OUTPUT_STDERR_STYLE = "#ff8a8a"


_TOOL_HEADLINE_VERBS: dict[str, tuple[str, str]] = {
    "SetTodoList": ("Updating", "Updated"),
    "ExecuteTodo": ("Executing", "Executed"),
}

_TOOL_HEADLINE_NAMES: dict[str, str] = {
    "SetTodoList": "Todo List",
    "ExecuteTodo": "Todo",
}


def _set_todo_list_activity_argument(argument: str | None) -> str | None:
    if not argument:
        return None
    marker = "; ready: "
    if marker not in argument:
        return None
    return f"ready: {argument.split(marker, 1)[1]}"


_SELF_CLOSING_BLOCKS = frozenset(
    {
        "fence",
        "code_block",
        "hr",
        "html_block",
        "table",
    }
)

_md_parser_instance: Any = None


def _get_md_parser() -> Any:
    """Lazy-initialise and return a markdown-it-py parser instance."""
    if not _HAS_MARKDOWN_IT:
        return None
    global _md_parser_instance
    if _md_parser_instance is None:
        _md_parser_instance = _MarkdownIt()
    return _md_parser_instance


def _find_committed_boundary(text: str) -> int | None:
    """Find the character offset up to which top-level blocks can be committed.

    Returns ``None`` when there are fewer than two complete top-level blocks,
    meaning nothing can be safely frozen yet.
    """
    md = _get_md_parser()
    if md is None:
        return None
    tokens = md.parse(text)

    depth = 0
    block_ends: list[int] = []
    for tok in tokens:
        if tok.nesting == 1:
            depth += 1
        elif tok.nesting == -1:
            depth -= 1
        if (
            depth == 0
            and (tok.type.endswith("_close") or tok.type in _SELF_CLOSING_BLOCKS)
            and tok.map is not None
        ):
            block_ends.append(tok.map[1])

    # Need at least 2 closed blocks to commit the earlier ones
    if len(block_ends) < 2:
        return None

    commit_line = block_ends[-2]
    lines = text.split("\n")
    return sum(len(line) + 1 for line in lines[:commit_line])


class _ContentBlock:
    def __init__(self, is_think: bool):
        self.is_think = is_think
        self._spinner = Spinner("dots", self.status_text)
        self._chunks: list[str] = []
        self._raw_text_cache: str | None = ""
        self._raw_text_length = 0
        # Incremental markdown streaming state
        self._committed_text: str = ""
        self._committed_renderable: RenderableType | None = None
        self._last_boundary_check_len: int = 0

    @property
    def raw_text(self) -> str:
        if self._raw_text_cache is None:
            self._raw_text_cache = "".join(self._chunks)
        return self._raw_text_cache

    @property
    def status_text(self) -> str:
        return "Thinking..." if self.is_think else "Composing..."

    def _mk_style(self) -> str:
        return "grey50 italic" if self.is_think else ""

    def _mk_bullet_style(self) -> str | None:
        return "grey50" if self.is_think else None

    def _try_advance_commit(self) -> None:
        """Advance the committed boundary if new top-level blocks are closed."""
        full = self.raw_text
        if len(full) - self._last_boundary_check_len < 128:
            return
        self._last_boundary_check_len = len(full)
        boundary = _find_committed_boundary(full)
        if boundary is not None and boundary > len(self._committed_text):
            self._committed_text = full[:boundary]
            self._committed_renderable = Markdown(
                self._committed_text,
                style=self._mk_style(),
            )

    @property
    def _pending_text(self) -> str:
        """Return the portion of text that has not been committed yet."""
        full = self.raw_text
        if self._committed_text:
            return full[len(self._committed_text) :]
        return full

    def _compose_incremental(self) -> RenderableType:
        """Build a renderable combining frozen committed blocks and the live tail."""
        parts: list[RenderableType] = []
        if self._committed_renderable is not None:
            parts.append(self._committed_renderable)
        pending = self._pending_text
        if pending:
            parts.append(Markdown(pending, style=self._mk_style()))
        if not parts:
            return Markdown("", style=self._mk_style())
        return Group(*parts) if len(parts) > 1 else parts[0]

    def compose(self, *, show_indicator: bool = True) -> RenderableType:
        if show_indicator:
            return self._spinner
        # Use incremental rendering when committed content exists
        if self._committed_renderable is not None:
            return BulletColumns(
                self._compose_incremental(),
                bullet_style=self._mk_bullet_style(),
            )
        return self.compose_final()

    def compose_final(self) -> RenderableType:
        # On final compose, render the full text (no incremental needed)
        return BulletColumns(
            Markdown(
                self.raw_text,
                style=self._mk_style(),
            ),
            bullet_style=self._mk_bullet_style(),
        )

    def compose_tail(
        self,
        *,
        max_chars: int,
        show_indicator: bool = True,
    ) -> tuple[RenderableType, bool]:
        if show_indicator:
            return self._spinner, False

        text, truncated = self._tail_text(max_chars)
        if truncated:
            text = f"...\n\n{text}"
        renderable = BulletColumns(
            Markdown(
                text,
                style=self._mk_style(),
            ),
            bullet_style=self._mk_bullet_style(),
        )
        return renderable, truncated

    def _tail_text(self, max_chars: int) -> tuple[str, bool]:
        if max_chars <= 0 or self._raw_text_length <= max_chars:
            return self.raw_text, False

        remaining = max_chars
        parts: list[str] = []
        for chunk in reversed(self._chunks):
            if remaining <= 0:
                break
            if len(chunk) <= remaining:
                parts.append(chunk)
                remaining -= len(chunk)
            else:
                parts.append(chunk[-remaining:])
                remaining = 0
        return "".join(reversed(parts)), True

    def append(self, content: str) -> None:
        self._chunks.append(content)
        self._raw_text_cache = None
        self._raw_text_length += len(content)
        self._try_advance_commit()


class _ToolCallBlock:
    def __init__(self, tool_call: ToolCall):
        self._tool_name = tool_call.function.name
        self._lexer = streamingjson.Lexer()
        if tool_call.function.arguments is not None:
            self._lexer.append_string(tool_call.function.arguments)

        self._argument = extract_key_argument(self._lexer, self._tool_name)
        self._full_url = self._extract_full_url(tool_call.function.arguments, self._tool_name)
        self._result: ToolReturnValue | None = None

        self._output_tail = deque[tuple[int, str, str]](maxlen=MAX_TOOL_OUTPUT_TAIL_LINES)
        self._output_tail_chars = 0
        self._output_tail_truncated = False
        self._output_line_count = 0

        self._spinning_dots = Spinner("dots", text="")
        self._renderable: RenderableType = self._compose()

    def compose(self, *, show_indicator: bool = True) -> RenderableType:
        if show_indicator:
            return self._renderable
        return self._compose(show_indicator=False)

    @property
    def finished(self) -> bool:
        return self._result is not None

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def append_args_part(self, args_part: str) -> bool:
        if self.finished:
            return False
        self._lexer.append_string(args_part)
        argument = extract_key_argument(self._lexer, self._tool_name)
        if not argument or argument == self._argument:
            return False
        self._argument = argument
        self._full_url = self._extract_full_url(self._lexer.complete_json(), self._tool_name)
        self._renderable = BulletColumns(
            self._build_headline_text(),
            bullet=self._spinning_dots,
        )
        return True

    def append_output(self, text: str, *, stream: str = "stdout") -> bool:
        if not text:
            return False

        updated = False
        for line in text.splitlines(keepends=True) or [text]:
            line = truncate_line(line, MAX_TOOL_OUTPUT_LINE_CHARS)
            self._output_line_count += 1
            if (
                self._output_tail.maxlen is not None
                and len(self._output_tail) == self._output_tail.maxlen
            ):
                _, removed, _ = self._output_tail.popleft()
                self._output_tail_chars -= len(removed)
                self._output_tail_truncated = True
            self._output_tail.append((self._output_line_count, line, stream))
            self._output_tail_chars += len(line)
            updated = True

        while self._output_tail_chars > MAX_TOOL_OUTPUT_TAIL_CHARS and len(self._output_tail) > 1:
            _, removed, _ = self._output_tail.popleft()
            self._output_tail_chars -= len(removed)
            self._output_tail_truncated = True

        if updated and not self.finished:
            self._renderable = self._compose()
        return updated

    def finish(self, result: ToolReturnValue):
        self._result = result
        self._renderable = self._compose()

    def _compose(self, *, show_indicator: bool = True) -> RenderableType:
        lines: list[RenderableType] = [self._build_headline_text()]

        if output_tail := self._render_output_tail():
            lines.append(output_tail)

        if self._result is not None:
            lines.extend(self._render_result_display(self._result))

        if self.finished:
            assert self._result is not None
            return BulletColumns(
                Group(*lines),
                bullet_style="green" if not self._result.is_error else "red",
            )
        if show_indicator:
            return BulletColumns(
                Group(*lines),
                bullet=self._spinning_dots,
            )
        return BulletColumns(Group(*lines), bullet_style="grey50")

    @staticmethod
    def _extract_full_url(arguments: str | None, tool_name: str) -> str | None:
        if tool_name != "FetchURL" or not arguments:
            return None
        try:
            args = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(args, dict):
            url = cast(dict[str, Any], args).get("url")
            if url:
                return str(url)
        return None

    @property
    def status_text(self) -> str:
        if self._tool_name == "SetTodoList":
            activity_argument = _set_todo_list_activity_argument(self._argument)
            if activity_argument:
                return f"Updating Todo List ({activity_argument})"
        return self._headline_plain(finished=False)

    def _headline_plain(self, *, finished: bool) -> str:
        present, past = _TOOL_HEADLINE_VERBS.get(self._tool_name, ("Using", "Used"))
        display_name = _TOOL_HEADLINE_NAMES.get(self._tool_name, self._tool_name)
        text = f"{past if finished else present} {display_name}"
        if self._argument:
            text += f" ({self._argument})"
        return text

    @staticmethod
    def _format_output_gutter(line_no: str, width: int) -> str:
        return f"{line_no:>{width}}{_OUTPUT_GUTTER_SEPARATOR}"

    def _build_headline_text(self) -> Text:
        present, past = _TOOL_HEADLINE_VERBS.get(self._tool_name, ("Using", "Used"))
        display_name = _TOOL_HEADLINE_NAMES.get(self._tool_name, self._tool_name)
        text = Text()
        text.append(f"{past if self.finished else present} ")
        text.append(display_name, style="blue")
        if self._argument:
            text.append(" (", style="grey50")
            arg_style = Style(color="grey50", link=self._full_url) if self._full_url else "grey50"
            text.append(self._argument, style=arg_style)
            text.append(")", style="grey50")
        return text

    @staticmethod
    def _extract_error_message(result: ToolReturnValue) -> str:
        if result.message:
            return result.message.strip()

        for block in result.display:
            if isinstance(block, BriefDisplayBlock) and block.text:
                return block.text.strip()

        return ""

    def _has_visible_output_tail(self) -> bool:
        return bool(self._output_tail and any(line.strip() for _, line, _ in self._output_tail))

    def _render_output_tail(self) -> RenderableType | None:
        if not self._has_visible_output_tail():
            return None

        gutter_width = len(str(self._output_tail[-1][0]))
        rendered = Text(no_wrap=False)
        if self._output_tail_truncated:
            rendered.append(
                self._format_output_gutter("", gutter_width),
                style=_OUTPUT_GUTTER_STYLE,
            )
            rendered.append("… older output omitted", style="grey50 italic")
            rendered.append("\n")

        for index, (line_no, line, stream) in enumerate(self._output_tail):
            rendered.append(
                self._format_output_gutter(str(line_no), gutter_width),
                style=_OUTPUT_GUTTER_STYLE,
            )
            line_text = line.rstrip("\n")
            rendered.append(
                line_text,
                style=_OUTPUT_STDERR_STYLE if stream == "stderr" else _OUTPUT_STDOUT_STYLE,
            )
            if index != len(self._output_tail) - 1:
                rendered.append("\n")

        return Group(Text("Output tail", style="cyan dim"), rendered)

    def _output_tail_text(self) -> str:
        return "".join(line for _, line, _ in self._output_tail).strip("\n")

    def _should_suppress_error_output(
        self,
        result: ToolReturnValue,
        *,
        tool_name: str,
    ) -> bool:
        if (
            tool_name != "Shell"
            and tool_name != self._tool_name
            or not self._has_visible_output_tail()
            or self._output_tail_truncated
        ):
            return False

        return self._output_tail_text() == self._stringify_output(result.output).strip("\n")

    def _render_result_display(
        self,
        result: ToolReturnValue,
        *,
        tool_name: str | None = None,
    ) -> list[RenderableType]:
        lines: list[RenderableType] = []
        tool_name = tool_name or self._tool_name
        if result.is_error:
            error_message = self._extract_error_message(result)
            if error_message:
                lines.append(Text(error_message, style="red", overflow="fold"))

            error_output = self._extract_error_output(result)
            if error_output and not self._should_suppress_error_output(result, tool_name=tool_name):
                lines.append(Text(error_output, style="red", overflow="fold"))
        else:
            has_diff_display = any(isinstance(block, DiffDisplayBlock) for block in result.display)
            if result.message and has_diff_display:
                lines.append(Markdown(result.message, style="dim"))
            elif tool_name in _FILE_CONTENT_TOOLS:
                if result.message:
                    lines.append(Markdown(result.message, style="dim"))
                file_content = self._render_file_content(result)
                if file_content:
                    lines.append(file_content)

        last_diff_path: str | None = None
        for block in result.display:
            if isinstance(block, BriefDisplayBlock):
                last_diff_path = None
                if result.is_error:
                    continue
                if block.text:
                    lines.append(Markdown(block.text, style="grey50"))
            elif isinstance(block, TodoDisplayBlock):
                last_diff_path = None
                markdown = self._render_todo_markdown(block)
                if markdown:
                    lines.append(Markdown(markdown, style="grey50"))
            elif isinstance(block, BackgroundTaskDisplayBlock):
                last_diff_path = None
                lines.append(
                    Markdown(
                        f"`{block.task_id}` [{block.status}] {block.description}",
                        style="grey50",
                    )
                )
            elif isinstance(block, DiffDisplayBlock):
                if block.path != last_diff_path:
                    lines.append(Text(block.path, style="bold"))
                    last_diff_path = block.path
                else:
                    lines.append(Text("⋮", style="grey50"))

                lines.append(
                    render_diff_block(
                        block,
                        source_line_numbers=tool_name in SOURCE_LINE_NUMBER_DIFF_TOOLS,
                    )
                )
            else:
                last_diff_path = None

        return lines

    def _render_file_content(self, result: ToolReturnValue) -> RenderableType | None:
        raw = self._stringify_output(result.output).strip("\n")
        if not raw.strip():
            return None

        all_lines = raw.splitlines()
        if not all_lines:
            return None

        truncated = len(all_lines) > MAX_FILE_CONTENT_DISPLAY_LINES
        visible = all_lines[:MAX_FILE_CONTENT_DISPLAY_LINES] if truncated else all_lines

        # Enforce character budget
        total_chars = 0
        capped: list[str] = []
        for line in visible:
            total_chars += len(line)
            if total_chars > MAX_FILE_CONTENT_DISPLAY_CHARS and capped:
                truncated = True
                break
            capped.append(line)
        visible = capped

        # Parse lines – ReadFile format: "{line_num:6d}\t{content}"
        parsed: list[tuple[str, str]] = []
        for line in visible:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                parsed.append((parts[0].strip(), parts[1]))
            else:
                parsed.append(("", line))

        if not parsed:
            return None

        gutter_width = max(len(ln) for ln, _ in parsed)
        rendered = Text(no_wrap=False)

        for index, (line_no, content) in enumerate(parsed):
            rendered.append(
                self._format_output_gutter(line_no, gutter_width),
                style=_OUTPUT_GUTTER_STYLE,
            )
            rendered.append(
                truncate_line(content.rstrip("\n"), MAX_TOOL_OUTPUT_LINE_CHARS),
                style=_OUTPUT_STDOUT_STYLE,
            )
            if index < len(parsed) - 1:
                rendered.append("\n")

        if truncated:
            rendered.append("\n")
            rendered.append(
                self._format_output_gutter("", gutter_width),
                style=_OUTPUT_GUTTER_STYLE,
            )
            rendered.append("… remaining content omitted", style="grey50 italic")

        return Group(Text("File content", style="cyan dim"), rendered)

    @classmethod
    def _extract_error_output(cls, result: ToolReturnValue) -> str:
        text = cls._stringify_output(result.output).strip("\n")
        if not text.strip():
            return ""

        truncated = False
        if len(text) > MAX_TOOL_ERROR_OUTPUT_CHARS:
            text = text[:MAX_TOOL_ERROR_OUTPUT_CHARS].rstrip()
            truncated = True

        lines = text.splitlines()
        if len(lines) > MAX_TOOL_ERROR_OUTPUT_LINES:
            text = "\n".join(lines[:MAX_TOOL_ERROR_OUTPUT_LINES]).rstrip()
            truncated = True

        if truncated:
            text += "\n[...truncated]"
        return text

    @staticmethod
    def _stringify_output(output: str | ContentPart | Sequence[ContentPart]) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, TextPart):
            return output.text
        if isinstance(output, Sequence):
            return "".join(part.text for part in output if isinstance(part, TextPart))
        return ""

    @staticmethod
    def _render_todo_meta(todo: TodoDisplayItem) -> str:
        details: list[str] = []
        match todo.executor:
            case "background_shell":
                details.append("`Shell(bg)`")
            case "main":
                details.append("`main`")
            case None:
                pass
        if todo.done_when:
            details.append(f"when: {todo.done_when}")
        return f" — {' · '.join(details)}" if details else ""

    @staticmethod
    def _short_todo_label(todo: TodoDisplayItem) -> str:
        return todo_label(todo.title)

    def _render_todo_markdown(self, block: TodoDisplayBlock) -> str:
        lines: list[str] = []
        for todo in block.items:
            meta = self._render_todo_meta(todo)
            normalized = todo.status.replace("_", " ").lower()
            match normalized:
                case "pending":
                    lines.append(f"- {todo.title}{meta}")
                case "in progress":
                    lines.append(f"- {todo.title}{meta} ←")
                case "done":
                    lines.append(f"- ~~{todo.title}~~{meta}")
                case "blocked":
                    lines.append(f"- {todo.title}{meta} (blocked)")
                case _:
                    lines.append(f"- {todo.title}{meta}")

        return "\n".join(lines)


class _StatusBlock:
    def __init__(self, initial: StatusUpdate) -> None:
        self.text = Text("", justify="right")
        self._context_usage: float = 0.0
        self._context_tokens: int = 0
        self._max_context_tokens: int = 0
        self.update(initial)

    def render(self) -> RenderableType:
        return self.text

    def update(self, status: StatusUpdate) -> bool:
        has_update = False
        if status.context_usage is not None:
            self._context_usage = status.context_usage
            has_update = True
        if status.context_tokens is not None:
            self._context_tokens = status.context_tokens
            has_update = True
        if status.max_context_tokens is not None:
            self._max_context_tokens = status.max_context_tokens
            has_update = True
        if has_update:
            self.text.plain = format_context_status(
                self._context_usage,
                self._context_tokens,
                self._max_context_tokens,
            )
        return has_update


ContentBlock = _ContentBlock
ToolCallBlock = _ToolCallBlock
StatusBlock = _StatusBlock
