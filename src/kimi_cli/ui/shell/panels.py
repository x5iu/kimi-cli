from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from kimi_cli.ui.shell.console import console
from kimi_cli.utils.diff import format_unified_diff
from kimi_cli.utils.rich.markdown import Markdown
from kimi_cli.utils.rich.syntax import KimiSyntax
from kimi_cli.wire.types import (
    ApprovalRequest,
    ApprovalResponse,
    BriefDisplayBlock,
    DiffDisplayBlock,
    QuestionRequest,
    ShellDisplayBlock,
)

MAX_PREVIEW_LINES = 4
QUESTION_BODY_PREVIEW_LINES = 3
OTHER_OPTION_LABEL = "Other"


class _ApprovalContentBlock(NamedTuple):
    """A pre-rendered content block for approval request with line count."""

    text: str
    lines: int
    style: str = ""
    lexer: str = ""


def _preview_text(text: str, *, max_lines: int) -> tuple[str, bool]:
    stripped = text.rstrip("\n")
    if not stripped:
        return "", False

    lines = stripped.splitlines()
    if len(lines) <= max_lines:
        return stripped, False

    preview = "\n".join(lines[:max_lines]).rstrip()
    return f"{preview}\n...", True


class _ApprovalRequestPanel:
    def __init__(self, request: ApprovalRequest):
        self.request = request
        self.options: list[tuple[str, ApprovalResponse.Kind]] = [
            ("Approve once", "approve"),
            ("Approve for this session", "approve_for_session"),
            ("Reject", "reject"),
        ]
        self.selected_index = 0

        self._content_blocks: list[_ApprovalContentBlock] = []
        last_diff_path: str | None = None

        if request.description and not request.display:
            text = request.description.rstrip("\n")
            self._content_blocks.append(
                _ApprovalContentBlock(text=text, lines=text.count("\n") + 1)
            )

        for block in request.display:
            if isinstance(block, DiffDisplayBlock):
                if block.path != last_diff_path:
                    self._content_blocks.append(
                        _ApprovalContentBlock(text=block.path, lines=1, style="bold")
                    )
                    last_diff_path = block.path
                else:
                    self._content_blocks.append(
                        _ApprovalContentBlock(text="⋮", lines=1, style="dim")
                    )
                diff_text = format_unified_diff(
                    block.old_text,
                    block.new_text,
                    block.path,
                    include_file_header=False,
                ).rstrip("\n")
                self._content_blocks.append(
                    _ApprovalContentBlock(
                        text=diff_text,
                        lines=diff_text.count("\n") + 1,
                        lexer="diff",
                    )
                )
            elif isinstance(block, ShellDisplayBlock):
                text = block.command.rstrip("\n")
                self._content_blocks.append(
                    _ApprovalContentBlock(
                        text=text,
                        lines=text.count("\n") + 1,
                        lexer=block.language,
                    )
                )
                last_diff_path = None
            elif isinstance(block, BriefDisplayBlock) and block.text:
                text = block.text.rstrip("\n")
                self._content_blocks.append(
                    _ApprovalContentBlock(text=text, lines=text.count("\n") + 1, style="grey50")
                )
                last_diff_path = None

        self._total_lines = sum(b.lines for b in self._content_blocks)
        self.has_expandable_content = self._total_lines > MAX_PREVIEW_LINES

    def render(self, *, allow_expand: bool = True) -> RenderableType:
        content_lines: list[RenderableType] = [
            Text.from_markup(
                "[yellow]"
                f"{escape(self.request.sender)} is requesting approval to "
                f"{escape(self.request.action)}:[/yellow]"
            )
        ]
        content_lines.append(Text(""))

        remaining = MAX_PREVIEW_LINES
        for block in self._content_blocks:
            if remaining <= 0:
                break
            content_lines.append(self._render_block(block, remaining))
            remaining -= min(block.lines, remaining)

        if self.has_expandable_content and allow_expand:
            content_lines.append(
                Text("... (truncated, press Ctrl-E or type /more to expand)", style="dim italic")
            )

        lines: list[RenderableType] = []
        if content_lines:
            lines.append(Padding(Group(*content_lines), (0, 0, 0, 1)))

        if lines:
            lines.append(Text(""))
        for i, (option_text, _) in enumerate(self.options):
            num = i + 1
            if i == self.selected_index:
                lines.append(Text(f"→ [{num}] {option_text}", style="cyan"))
            else:
                lines.append(Text(f"  [{num}] {option_text}", style="grey50"))

        lines.append(Text(""))
        hint = "  Press 1/2/3 to choose, or use ↑/↓ and Enter"
        if self.has_expandable_content and allow_expand:
            hint += "  (Ctrl-E or /more to expand)"
        lines.append(Text(hint, style="dim"))

        return Panel(
            Group(*lines),
            border_style="bold yellow",
            title="[bold yellow]⚠ ACTION REQUIRED[/bold yellow]",
            title_align="left",
            padding=(0, 1),
        )

    def _render_block(
        self, block: _ApprovalContentBlock, max_lines: int | None = None
    ) -> RenderableType:
        text = block.text
        if max_lines is not None and block.lines > max_lines:
            text = "\n".join(text.split("\n")[:max_lines])

        if block.lexer:
            return KimiSyntax(text, block.lexer)
        return Text(text, style=block.style)

    def render_full(self) -> list[RenderableType]:
        return [self._render_block(block) for block in self._content_blocks]

    def move_up(self):
        self.selected_index = (self.selected_index - 1) % len(self.options)

    def move_down(self):
        self.selected_index = (self.selected_index + 1) % len(self.options)

    def get_selected_response(self) -> ApprovalResponse.Kind:
        return self.options[self.selected_index][1]


def _show_approval_in_pager(panel: _ApprovalRequestPanel) -> None:
    with console.screen(), console.pager(styles=True):
        console.print(
            Text.from_markup(
                "[yellow]⚠ "
                f"{escape(panel.request.sender)} is requesting approval to "
                f"{escape(panel.request.action)}:[/yellow]"
            )
        )
        console.print()
        for renderable in panel.render_full():
            console.print(renderable)


class _QuestionRequestPanel:
    """Renders structured questions for the user to answer interactively."""

    def __init__(self, request: QuestionRequest):
        self.request = request
        self._current_question_index = 0
        self._answers: dict[str, str] = {}
        self._saved_selections: dict[int, tuple[int, set[int]]] = {}
        self._selected_index = 0
        self._multi_selected: set[int] = set()
        self._body_text: str = ""
        self.has_expandable_content: bool = False
        self._setup_current_question()

    def _setup_current_question(self) -> None:
        q = self._current_question
        self._options = [(o.label, o.description) for o in q.options]
        other_label = q.other_label or OTHER_OPTION_LABEL
        other_desc = q.other_description or ""
        self._options.append((other_label, other_desc))
        idx = self._current_question_index
        if idx in self._saved_selections:
            saved_idx, saved_multi = self._saved_selections[idx]
            self._selected_index = min(saved_idx, len(self._options) - 1)
            self._multi_selected = saved_multi
        elif q.question in self._answers:
            answer = self._answers[q.question]
            if q.multi_select:
                answer_labels = [a.strip() for a in answer.split(", ")]
                known_labels = {label for label, _ in self._options[:-1]}
                self._multi_selected = set()
                for i, (label, _) in enumerate(self._options[:-1]):
                    if label in answer_labels:
                        self._multi_selected.add(i)
                if any(answer_label not in known_labels for answer_label in answer_labels):
                    self._multi_selected.add(len(self._options) - 1)
                self._selected_index = min(self._multi_selected) if self._multi_selected else 0
            else:
                for i, (label, _) in enumerate(self._options):
                    if label == answer:
                        self._selected_index = i
                        break
                else:
                    self._selected_index = len(self._options) - 1
                self._multi_selected = set()
        else:
            self._selected_index = 0
            self._multi_selected = set()
        self._recompute_body()

    def _recompute_body(self) -> None:
        body = self._current_question.body
        self._body_text = body.rstrip("\n") if body else ""
        self.has_expandable_content = bool(self._body_text)

    @property
    def _current_question(self):
        return self.request.questions[self._current_question_index]

    @property
    def is_other_selected(self) -> bool:
        return self._selected_index == len(self._options) - 1

    @property
    def is_multi_select(self) -> bool:
        return self._current_question.multi_select

    @property
    def current_question_text(self) -> str:
        return self._current_question.question

    @property
    def options(self) -> Sequence[tuple[str, str]]:
        return self._options

    @property
    def option_count(self) -> int:
        return len(self._options)

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @selected_index.setter
    def selected_index(self, value: int) -> None:
        self._selected_index = value

    @property
    def multi_selected(self) -> set[int]:
        return self._multi_selected

    def set_multi_selected(self, indices: set[int]) -> None:
        self._multi_selected = set(indices)

    def should_prompt_other_input(self) -> bool:
        if not self.is_multi_select:
            return self.is_other_selected
        other_idx = len(self._options) - 1
        return other_idx in self._multi_selected

    def select_index(self, index: int) -> bool:
        if not (0 <= index < len(self._options)):
            return False
        self._selected_index = index
        return True

    def render(self, *, allow_expand: bool = True) -> RenderableType:
        q = self._current_question
        lines: list[RenderableType] = []

        if len(self.request.questions) > 1:
            tab_parts: list[str] = []
            for i, qi in enumerate(self.request.questions):
                label = escape(qi.header or f"Q{i + 1}")
                if i == self._current_question_index:
                    icon, style = "●", "bold cyan"
                elif qi.question in self._answers:
                    icon, style = "✓", "green"
                else:
                    icon, style = "○", "grey50"
                tab_parts.append(f"[{style}]({icon}) {label}[/{style}]")
            lines.append(Text.from_markup("  ".join(tab_parts)))
            lines.append(Text(""))

        lines.append(Text.from_markup(f"[yellow]? {escape(q.question)}[/yellow]"))
        if q.multi_select:
            lines.append(Text("  (separate multiple selections with commas)", style="dim italic"))
        lines.append(Text(""))

        if self._body_text and allow_expand:
            body_preview, _ = _preview_text(
                self._body_text,
                max_lines=QUESTION_BODY_PREVIEW_LINES,
            )
            if body_preview:
                lines.append(Padding(Text(body_preview, style="grey50"), (0, 0, 0, 1)))
                lines.append(Text(""))
            lines.append(
                Text.from_markup(
                    "[bold cyan]  ▶ Press Ctrl-E or type /more to view full content[/bold cyan]"
                )
            )
            lines.append(Text(""))

        for i, (label, description) in enumerate(self._options):
            num = i + 1
            if q.multi_select:
                checked = "✓" if i in self._multi_selected else " "
                prefix = f"\\[{checked}]"
                if i == self._selected_index:
                    option_line = Text.from_markup(f"[cyan]{prefix} {escape(label)}[/cyan]")
                else:
                    option_line = Text.from_markup(f"[grey50]{prefix} {escape(label)}[/grey50]")
            else:
                if i == self._selected_index:
                    option_line = Text.from_markup(f"[cyan]→ \\[{num}] {escape(label)}[/cyan]")
                else:
                    option_line = Text.from_markup(f"[grey50]  \\[{num}] {escape(label)}[/grey50]")
            lines.append(option_line)

            if description:
                lines.append(Text(f"      {description}", style="dim"))

        lines.append(Text(""))
        hint = "  Use ↑/↓ to move focus and Enter to choose"
        if self.has_expandable_content and allow_expand:
            hint += "  (Ctrl-E or /more to expand)"
        lines.append(Text(hint, style="dim"))
        if q.multi_select:
            lines.append(
                Text(
                    "  Press Space or Enter to select. "
                    "Press Enter again on a checked option to submit.",
                    style="dim",
                )
            )
            lines.append(Text("  Select Other to enter custom text.", style="dim"))
        else:
            lines.append(Text("  Select Other to enter custom text.", style="dim"))

        return Panel(
            Group(*lines),
            border_style="bold cyan",
            title="[bold cyan]? QUESTION[/bold cyan]",
            title_align="left",
            padding=(0, 1),
        )

    def go_to(self, index: int) -> None:
        if index == self._current_question_index:
            return
        if not (0 <= index < len(self.request.questions)):
            return
        self._saved_selections[self._current_question_index] = (
            self._selected_index,
            set(self._multi_selected),
        )
        self._current_question_index = index
        self._setup_current_question()

    def next_tab(self) -> None:
        if self._current_question_index < len(self.request.questions) - 1:
            self.go_to(self._current_question_index + 1)

    def prev_tab(self) -> None:
        if self._current_question_index > 0:
            self.go_to(self._current_question_index - 1)

    def move_up(self) -> None:
        self._selected_index = (self._selected_index - 1) % len(self._options)

    def move_down(self) -> None:
        self._selected_index = (self._selected_index + 1) % len(self._options)

    def toggle_select(self) -> None:
        if not self.is_multi_select:
            return
        if self._selected_index in self._multi_selected:
            self._multi_selected.discard(self._selected_index)
        else:
            self._multi_selected.add(self._selected_index)

    def submit(self) -> bool:
        q = self._current_question
        if q.multi_select:
            other_idx = len(self._options) - 1
            if other_idx in self._multi_selected:
                return False
            selected_labels = [
                self._options[i][0] for i in sorted(self._multi_selected) if i < len(q.options)
            ]
            if not selected_labels:
                return False
            self._answers[q.question] = ", ".join(selected_labels)
        else:
            if self.is_other_selected:
                return False
            self._answers[q.question] = self._options[self._selected_index][0]
        self._saved_selections.pop(self._current_question_index, None)
        return self._advance()

    def submit_other(self, text: str) -> bool:
        q = self._current_question
        if q.multi_select:
            other_idx = len(self._options) - 1
            selected_labels = [
                self._options[i][0]
                for i in sorted(self._multi_selected)
                if i < len(q.options) and i != other_idx
            ]
            if text:
                selected_labels.append(text)
            self._answers[q.question] = ", ".join(selected_labels) if selected_labels else text
        else:
            self._answers[q.question] = text
        self._saved_selections.pop(self._current_question_index, None)
        return self._advance()

    def _advance(self) -> bool:
        total = len(self.request.questions)
        if len(self._answers) >= total:
            return True
        for offset in range(1, total + 1):
            idx = (self._current_question_index + offset) % total
            if self.request.questions[idx].question not in self._answers:
                self._current_question_index = idx
                self._setup_current_question()
                return False
        return True

    def get_answers(self) -> dict[str, str]:
        return self._answers

    def render_full_body(self) -> list[RenderableType]:
        if not self._body_text:
            return []
        return [Markdown(self._body_text)]


def _show_question_body_in_pager(panel: _QuestionRequestPanel) -> None:
    with console.screen(), console.pager(styles=True):
        console.print(Text.from_markup(f"[yellow]? {escape(panel.current_question_text)}[/yellow]"))
        console.print()
        for renderable in panel.render_full_body():
            console.print(renderable)
