from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import streamingjson  # pyright: ignore[reportMissingTypeStubs]
from rich.console import Group, RenderableType
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

from kimi_cli.soul import format_context_status
from kimi_cli.tools import extract_key_argument
from kimi_cli.utils.diff import format_unified_diff
from kimi_cli.utils.rich.columns import BulletColumns
from kimi_cli.utils.rich.markdown import Markdown
from kimi_cli.utils.rich.syntax import KimiSyntax
from kimi_cli.wire.types import (
    BackgroundTaskDisplayBlock,
    BriefDisplayBlock,
    ContentPart,
    DiffDisplayBlock,
    StatusUpdate,
    TextPart,
    TodoDisplayBlock,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
)

MAX_SUBAGENT_TOOL_CALLS_TO_SHOW = 4
MAX_TOOL_ERROR_OUTPUT_LINES = 12
MAX_TOOL_ERROR_OUTPUT_CHARS = 4000
EDIT_DIFF_LINE_NUMBER_TOOLS = {"Edit", "StrReplaceFile"}


class _ContentBlock:
    def __init__(self, is_think: bool):
        self.is_think = is_think
        self._spinner = Spinner("dots", self.status_text)
        self._chunks: list[str] = []
        self._raw_text_cache: str | None = ""
        self._raw_text_length = 0

    @property
    def raw_text(self) -> str:
        if self._raw_text_cache is None:
            self._raw_text_cache = "".join(self._chunks)
        return self._raw_text_cache

    @property
    def status_text(self) -> str:
        return "Thinking..." if self.is_think else "Composing..."

    def compose(self, *, show_indicator: bool = True) -> RenderableType:
        if show_indicator:
            return self._spinner
        return self.compose_final()

    def compose_final(self) -> RenderableType:
        return BulletColumns(
            Markdown(
                self.raw_text,
                style="grey50 italic" if self.is_think else "",
            ),
            bullet_style="grey50" if self.is_think else None,
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
                style="grey50 italic" if self.is_think else "",
            ),
            bullet_style="grey50" if self.is_think else None,
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


class _ToolCallBlock:
    class FinishedSubCall(NamedTuple):
        call: ToolCall
        result: ToolReturnValue

    def __init__(self, tool_call: ToolCall):
        self._tool_name = tool_call.function.name
        self._lexer = streamingjson.Lexer()
        if tool_call.function.arguments is not None:
            self._lexer.append_string(tool_call.function.arguments)

        self._argument = extract_key_argument(self._lexer, self._tool_name)
        self._full_url = self._extract_full_url(tool_call.function.arguments, self._tool_name)
        self._result: ToolReturnValue | None = None

        self._ongoing_subagent_tool_calls: dict[str, ToolCall] = {}
        self._last_subagent_tool_call: ToolCall | None = None
        self._n_finished_subagent_tool_calls = 0
        self._finished_subagent_tool_calls = deque[_ToolCallBlock.FinishedSubCall](
            maxlen=MAX_SUBAGENT_TOOL_CALLS_TO_SHOW
        )

        self._spinning_dots = Spinner("dots", text="")
        self._renderable: RenderableType = self._compose()

    def compose(self, *, show_indicator: bool = True) -> RenderableType:
        if show_indicator:
            return self._renderable
        return self._compose(show_indicator=False)

    @property
    def finished(self) -> bool:
        return self._result is not None

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

    def finish(self, result: ToolReturnValue):
        self._result = result
        self._renderable = self._compose()

    def append_sub_tool_call(self, tool_call: ToolCall):
        self._ongoing_subagent_tool_calls[tool_call.id] = tool_call
        self._last_subagent_tool_call = tool_call

    def append_sub_tool_call_part(self, tool_call_part: ToolCallPart):
        if self._last_subagent_tool_call is None:
            return
        if not tool_call_part.arguments_part:
            return
        if self._last_subagent_tool_call.function.arguments is None:
            self._last_subagent_tool_call.function.arguments = tool_call_part.arguments_part
        else:
            self._last_subagent_tool_call.function.arguments += tool_call_part.arguments_part

    def finish_sub_tool_call(self, tool_result: ToolResult):
        self._last_subagent_tool_call = None
        sub_tool_call = self._ongoing_subagent_tool_calls.pop(tool_result.tool_call_id, None)
        if sub_tool_call is None:
            return

        self._finished_subagent_tool_calls.append(
            _ToolCallBlock.FinishedSubCall(
                call=sub_tool_call,
                result=tool_result.return_value,
            )
        )
        self._n_finished_subagent_tool_calls += 1
        self._renderable = self._compose()

    def _compose(self, *, show_indicator: bool = True) -> RenderableType:
        lines: list[RenderableType] = [self._build_headline_text()]

        if self._n_finished_subagent_tool_calls > MAX_SUBAGENT_TOOL_CALLS_TO_SHOW:
            n_hidden = self._n_finished_subagent_tool_calls - MAX_SUBAGENT_TOOL_CALLS_TO_SHOW
            lines.append(
                BulletColumns(
                    Text(
                        f"{n_hidden} more tool call{'s' if n_hidden > 1 else ''} ...",
                        style="grey50 italic",
                    ),
                    bullet_style="grey50",
                )
            )
        for sub_call, sub_result in self._finished_subagent_tool_calls:
            argument = extract_key_argument(
                sub_call.function.arguments or "",
                sub_call.function.name,
            )
            sub_url = self._extract_full_url(sub_call.function.arguments, sub_call.function.name)
            sub_text = Text()
            sub_text.append("Used ")
            sub_text.append(sub_call.function.name, style="blue")
            if argument:
                sub_text.append(" (", style="grey50")
                arg_style = Style(color="grey50", link=sub_url) if sub_url else "grey50"
                sub_text.append(argument, style=arg_style)
                sub_text.append(")", style="grey50")
            sub_lines = [cast(RenderableType, sub_text)]
            sub_lines.extend(
                self._render_result_display(sub_result, tool_name=sub_call.function.name)
            )
            lines.append(
                BulletColumns(
                    Group(*sub_lines),
                    bullet_style="green" if not sub_result.is_error else "red",
                )
            )

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
        return self._headline_plain(finished=False)

    def _headline_plain(self, *, finished: bool) -> str:
        text = f"{'Used' if finished else 'Using'} {self._tool_name}"
        if self._argument:
            text += f" ({self._argument})"
        return text

    def _build_headline_text(self) -> Text:
        text = Text()
        text.append("Used " if self.finished else "Using ")
        text.append(self._tool_name, style="blue")
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
            if error_output:
                lines.append(Text(error_output, style="red", overflow="fold"))
        else:
            has_diff_display = any(isinstance(block, DiffDisplayBlock) for block in result.display)
            if result.message and has_diff_display:
                lines.append(Markdown(result.message, style="dim"))

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

                diff_text = format_unified_diff(
                    block.old_text,
                    block.new_text,
                    block.path,
                    include_file_header=False,
                ).rstrip("\n")
                if diff_text:
                    lines.append(
                        KimiSyntax(
                            diff_text,
                            "diff",
                            line_numbers=tool_name in EDIT_DIFF_LINE_NUMBER_TOOLS,
                        )
                    )
            else:
                last_diff_path = None

        return lines

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

    def _render_todo_markdown(self, block: TodoDisplayBlock) -> str:
        lines: list[str] = []
        for todo in block.items:
            normalized = todo.status.replace("_", " ").lower()
            match normalized:
                case "pending":
                    lines.append(f"- {todo.title}")
                case "in progress":
                    lines.append(f"- {todo.title} ←")
                case "done":
                    lines.append(f"- ~~{todo.title}~~")
                case _:
                    lines.append(f"- {todo.title}")
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
