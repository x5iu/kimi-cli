from __future__ import annotations

import asyncio
import json
import shlex
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from hashlib import md5
from pathlib import Path
from typing import Any, Literal, cast

from kaos.path import KaosPath
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.completion import merge_completers
from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, Filter, has_completions, has_focus, is_done
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame, TextArea
from pydantic import BaseModel, ValidationError
from rich.style import Style as _RichStyle

from kimi_cli.llm import ModelCapability
from kimi_cli.share import get_share_dir
from kimi_cli.soul import StatusSnapshot, format_context_status
from kimi_cli.ui.shell import placeholders as prompt_placeholders
from kimi_cli.ui.shell.completion import (
    LocalFileMentionCompleter,
    SlashCommandCompleter,
    SlashCommandMenuControl,
)
from kimi_cli.ui.shell.completion import (
    find_prompt_float_container as _find_prompt_float_container,
)
from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.keyboard import KeyEvent
from kimi_cli.ui.shell.placeholders import (
    PromptPlaceholderManager,
    normalize_pasted_text,
    sanitize_surrogates,
)
from kimi_cli.ui.shell.rich_ptk import (
    RichRenderableControl as _RichRenderableControl,
)
from kimi_cli.ui.shell.rich_ptk import (
    StackedRichRenderableControl as _StackedRichRenderableControl,
)
from kimi_cli.ui.shell.rich_ptk import (
    rich_from_ansi,
    rich_style_to_prompt_toolkit,
)
from kimi_cli.ui.shell.slash import SKILL_PREFIX as _skill_prefix
from kimi_cli.ui.shell.slash import TURN_ALLOWED_COMMANDS as _turn_allowed_commands
from kimi_cli.ui.shell.toast import current_toast as _current_toast
from kimi_cli.ui.shell.toast import toast, toast_queues
from kimi_cli.ui.shell.visualize import (
    MAX_ACTIVE_TURN_CONTENT_CHARS,
    MAX_ACTIVE_TURN_FLUSHED_BLOCKS,
    MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS,
    is_significant_for_render,
    render_user_prompt_block,
)
from kimi_cli.ui.shell.visualize import (
    recent_output_notice_text as _recent_output_notice_text,
)
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.clipboard import (
    grab_media_from_clipboard,
    is_clipboard_available,
)
from kimi_cli.utils.logging import logger
from kimi_cli.utils.slashcmd import SlashCommand
from kimi_cli.wire.types import ContentPart, StepInterrupted, TextPart, ThinkPart, ToolCallPart

AttachmentCache = prompt_placeholders.AttachmentCache
CachedAttachment = prompt_placeholders.CachedAttachment
_parse_attachment_kind = prompt_placeholders.parse_attachment_kind
_sanitize_surrogates = sanitize_surrogates  # backward compat re-export
_rich_from_ansi = rich_from_ansi
_rich_style_to_prompt_toolkit = rich_style_to_prompt_toolkit
_toast_queues = toast_queues
RichStyle = _RichStyle

# -- Terminal focus reporting -------------------------------------------------
# Register the CSI focus-in/focus-out escape sequences so prompt_toolkit's
# Vt100 parser recognises (and silently swallows) them instead of treating
# each byte as literal input.  The actual repaint on focus-in is handled by
# _install_focus_repaint which monkey-patches the input's read_keys method.
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES as _ANSI_SEQUENCES

_ANSI_SEQUENCES.setdefault("\x1b[I", Keys.Ignore)
_ANSI_SEQUENCES.setdefault("\x1b[O", Keys.Ignore)

_FOCUS_IN_SEQ = "\x1b[I"


class _ClearScreenRequest(Exception):
    """Raised when Ctrl-L is pressed to request full screen clear + history replay."""

    def __init__(self, buffer_text: str = ""):
        self.buffer_text = buffer_text
        super().__init__()

PROMPT_SYMBOL = "✨"
PROMPT_SYMBOL_SHELL = "$"
PROMPT_SYMBOL_THINKING = "💫"
PROMPT_SYMBOL_PLAN = "📋"
STEADY_INPUT_CURSOR = SimpleCursorShapeConfig(CursorShape.BEAM)

_DOTS_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_MOON_FRAMES = ("🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘")
_PAUSE_FRAMES = ("…",)
_INDICATOR_FRAMES: dict[str, tuple[str, ...]] = {
    "running": _DOTS_FRAMES,
    "thinking": _DOTS_FRAMES,
    "composing": _DOTS_FRAMES,
    "tool": _DOTS_FRAMES,
    "mcp": _DOTS_FRAMES,
    "compacting": _DOTS_FRAMES,
    "moon": _MOON_FRAMES,
    "approval": _PAUSE_FRAMES,
    "question": _PAUSE_FRAMES,
}
_INDICATOR_STYLES = {
    "running": "fg:#22c55e",
    "thinking": "fg:#22c55e",
    "composing": "fg:#22c55e",
    "tool": "fg:#22c55e",
    "mcp": "fg:#38bdf8",
    "compacting": "fg:#c084fc",
    "moon": "fg:#facc15",
    "approval": "fg:#f59e0b",
    "question": "fg:#22d3ee",
}
_TURN_UI_REFRESH_INTERVAL = 0.2
_TURN_UI_FAST_REFRESH_INTERVAL = 1 / 30
_TURN_UI_BURST_REFRESH_INTERVAL = 1 / 60
_TURN_UI_STREAM_BURST_SECONDS = 0.3
_TURN_UI_STREAM_RESUME_BURST_GAP = 0.15
_TURN_UI_BURST_STREAM_PUSH_PARTS = 1
_TURN_UI_BURST_STREAM_PUSH_CHARS = 32
_TURN_UI_COMPOSE_STREAM_PUSH_PARTS = 2
_TURN_UI_COMPOSE_STREAM_PUSH_CHARS = 64
_TURN_UI_FAST_STREAM_PUSH_PARTS = 4
_TURN_UI_FAST_STREAM_PUSH_CHARS = 128
_TURN_UI_STREAM_PUSH_PARTS = 8
_TURN_UI_STREAM_PUSH_CHARS = 256
_TERMINAL_SIZE_POLLING_INTERVAL = 1.0


class _HistoryEntry(BaseModel):
    content: str


def _load_history_entries(history_file: Path) -> list[_HistoryEntry]:
    entries: list[_HistoryEntry] = []
    if not history_file.exists():
        return entries

    try:
        with history_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse user history line; skipping: {line}",
                        line=line,
                    )
                    continue
                try:
                    entry = _HistoryEntry.model_validate(record)
                    entries.append(entry)
                except ValidationError:
                    logger.warning(
                        "Failed to validate user history entry; skipping: {line}",
                        line=line,
                    )
                    continue
    except OSError as exc:
        logger.warning(
            "Failed to load user history file: {file} ({error})",
            file=history_file,
            error=exc,
        )

    return entries


class PromptMode(Enum):
    AGENT = "agent"
    SHELL = "shell"

    def toggle(self) -> PromptMode:
        return PromptMode.SHELL if self == PromptMode.AGENT else PromptMode.AGENT

    def __str__(self) -> str:
        return self.value


class UserInput(BaseModel):
    mode: PromptMode
    command: str
    """The plain text representation of the user input."""
    resolved_command: str = ""
    """The text command after UI-only placeholders are expanded.
    Defaults to `command` when not explicitly set."""

    def model_post_init(self, __context: Any) -> None:
        if not self.resolved_command:
            self.resolved_command = self.command

    content: list[ContentPart]
    """The rich content parts."""

    def __str__(self) -> str:
        return self.command

    def __bool__(self) -> bool:
        return bool(self.command)


@dataclass(frozen=True, slots=True)
class TurnSubmitResult:
    accepted: bool
    persist_history: bool = False
    feedback: str = ""

    @classmethod
    def accept(cls, *, persist_history: bool = False) -> TurnSubmitResult:
        return cls(accepted=True, persist_history=persist_history)

    @classmethod
    def reject(cls, feedback: str = "") -> TurnSubmitResult:
        return cls(accepted=False, feedback=feedback)


@dataclass(frozen=True, slots=True)
class InputBoxState:
    active: bool = False
    mode: Literal["default", "reminder", "approval", "question", "question_other"] = "default"
    hint: str = ""


_REFRESH_INTERVAL = 1.0
_TURN_HISTORY_VIEW_EMPTY_POSITION = "0/0"


def _build_toolbar_tips(clipboard_available: bool) -> list[str]:
    tips = [
        "ctrl-x: toggle mode",
        "shift-tab: plan mode",
        "ctrl-o: editor",
        "ctrl-j: newline",
        "ctrl-l: redraw",
        "ctrl-y: history",
    ]
    if clipboard_available:
        tips.append("ctrl-v: paste clipboard")
    tips.append("@: mention files")
    return tips


_TIP_SEPARATOR = " | "


def _shell_style_dict() -> dict[str, str]:
    return {
        "bottom-toolbar": "noreverse",
        "frame.border": "fg:#38bdf8",
        "frame.label": "bold",
        "slash-completion-menu": "",
        "slash-completion-menu.separator": "fg:#4a5568",
        "slash-completion-menu.marker": "fg:#4a5568",
        "slash-completion-menu.marker.current": "fg:#4f9fff",
        "slash-completion-menu.command": "fg:#a6adba",
        "slash-completion-menu.meta": "fg:#7c8594",
        "slash-completion-menu.command.current": "fg:#6fb7ff bold",
        "slash-completion-menu.meta.current": "fg:#56a4ff",
        "slash-completion-menu.detail.prefix": "fg:#475569",
        "slash-completion-menu.detail": "fg:#93c5fd italic",
    }


class CustomPromptSession:
    def __init__(  # noqa: C901
        self,
        *,
        status_provider: Callable[[], StatusSnapshot],
        model_capabilities: set[ModelCapability],
        model_name: str | None,
        thinking: bool,
        agent_mode_slash_commands: Sequence[SlashCommand[Any]],
        shell_mode_slash_commands: Sequence[SlashCommand[Any]],
        editor_command_provider: Callable[[], str] = lambda: "",
        plan_mode_toggle_callback: Callable[[], Awaitable[bool]] | None = None,
        input_box_state_provider: Callable[[], InputBoxState] | None = None,
        working_dir_provider: Callable[[], str] | None = None,
        redraw_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        history_dir = get_share_dir() / "user-history"
        history_dir.mkdir(parents=True, exist_ok=True)
        work_dir_id = md5(str(KaosPath.cwd()).encode(encoding="utf-8")).hexdigest()
        self._deferred_erase_pending = False
        self._deferred_erase_x = 0
        self._deferred_erase_y = 0
        self._history_file = (history_dir / work_dir_id).with_suffix(".jsonl")
        self._status_provider = status_provider
        self._editor_command_provider = editor_command_provider
        self._plan_mode_toggle_callback = plan_mode_toggle_callback
        self._input_box_state_provider = input_box_state_provider or (lambda: InputBoxState())
        self._working_dir_provider = working_dir_provider or (lambda: str(KaosPath.cwd()))
        self._redraw_callback = redraw_callback
        self._model_capabilities = model_capabilities
        self._model_name = model_name
        self._last_history_content: str | None = None
        self._mode: PromptMode = PromptMode.AGENT
        self._thinking = thinking
        self._placeholder_manager = PromptPlaceholderManager()
        # Keep the old attribute for test compatibility and for any external imports.
        self._attachment_cache = self._placeholder_manager.attachment_cache
        self._tip_rotation_index: int = 0
        clipboard_available = is_clipboard_available()
        self._tips = _build_toolbar_tips(clipboard_available)

        history_entries = _load_history_entries(self._history_file)
        history = InMemoryHistory()
        for entry in history_entries:
            history.append_string(entry.content)
        self._history = history

        if history_entries:
            # for consecutive deduplication
            self._last_history_content = history_entries[-1].content

        # Build completers
        # Slash commands are only available at the top-level prompt. During an active
        # turn, the input box is reserved for reminders / approval answers, so keep
        # slash commands out of that UI entirely.
        self._file_mention_completer = LocalFileMentionCompleter(
            KaosPath.cwd().unsafe_to_local_path()
        )
        self._agent_slash_completer = SlashCommandCompleter(agent_mode_slash_commands)
        self._shell_slash_completer = SlashCommandCompleter(shell_mode_slash_commands)
        self._agent_mode_completer = merge_completers(
            [
                self._agent_slash_completer,
                # TODO(kaos): we need an async KaosFileMentionCompleter
                self._file_mention_completer,
            ],
            deduplicate=True,
        )
        _turn_slash_commands = [
            c
            for c in (*shell_mode_slash_commands, *agent_mode_slash_commands)
            if c.name in _turn_allowed_commands or c.name.startswith(_skill_prefix)
        ]
        self._turn_mode_completer = merge_completers(
            [SlashCommandCompleter(_turn_slash_commands), self._file_mention_completer],
            deduplicate=True,
        )
        self._shell_mode_completer = self._shell_slash_completer

        # Build key bindings
        _kb = KeyBindings()

        @_kb.add("c-x", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._mode = self._mode.toggle()
            toast(
                f"{self._mode.value} mode",
                topic="prompt_mode",
                duration=2.0,
                immediate=True,
            )
            # Apply mode-specific settings
            self._apply_mode(event)
            # Redraw UI
            event.app.invalidate()

        @_kb.add("s-tab", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Toggle plan mode with Shift+Tab."""
            if self._plan_mode_toggle_callback is not None:

                async def _toggle() -> None:
                    assert self._plan_mode_toggle_callback is not None
                    new_state = await self._plan_mode_toggle_callback()
                    if new_state:
                        toast("plan mode ON", topic="plan_mode", duration=3.0, immediate=True)
                    else:
                        toast("plan mode OFF", topic="plan_mode", duration=3.0, immediate=True)
                    event.app.invalidate()

                event.app.create_background_task(_toggle())
            event.app.invalidate()

        @_kb.add("escape", "enter", eager=True)
        @_kb.add("c-j", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Insert a newline when Alt-Enter or Ctrl-J is pressed."""
            event.current_buffer.insert_text("\n")

        @_kb.add("c-o", eager=True)
        def _(event: KeyPressEvent) -> None:
            """Open current buffer in external editor."""
            self._open_in_external_editor(event)

        @_kb.add("c-l", eager=True)
        def _(event: KeyPressEvent) -> None:
            if self._redraw_callback is not None:
                event.app.exit(exception=_ClearScreenRequest(event.current_buffer.text))
            else:
                self._hard_redraw(event.app)

        @_kb.add(Keys.BracketedPaste, eager=True)
        def _(event: KeyPressEvent) -> None:
            self._handle_bracketed_paste(event)

        if clipboard_available:

            @_kb.add("c-v", eager=True)
            def _(event: KeyPressEvent) -> None:
                if self._try_paste_media(event):
                    return
                clipboard_data = event.app.clipboard.get_data()
                self._insert_pasted_text(event.current_buffer, clipboard_data.text)
                event.app.invalidate()

            clipboard = PyperclipClipboard()
        else:
            clipboard = None

        self._clipboard = clipboard
        self._base_key_bindings = _kb

        self._session = PromptSession[str](
            message=self._render_message,
            prompt_continuation=self._render_prompt_continuation,
            placeholder=self._render_placeholder,
            rprompt=self._render_rprompt,
            cursor=STEADY_INPUT_CURSOR,
            completer=self._agent_mode_completer,
            complete_while_typing=True,
            reserve_space_for_menu=10,
            key_bindings=_kb,
            clipboard=clipboard,
            history=history,
            bottom_toolbar=self._render_bottom_toolbar,
            style=Style.from_dict(_shell_style_dict()),
        )
        # PromptSession defaults to polling terminal size every 0.5s, which causes
        # needless redraws/flicker in our persistent TUI input box. Keep a slower
        # polling interval so focus/window switches and resizes still recover cleanly.
        self._session.app.terminal_size_polling_interval = _TERMINAL_SIZE_POLLING_INTERVAL
        self._install_slash_completion_menu()

        # Allow completion to be triggered when the text is changed,
        # such as when backspace is used to delete text.
        @self._session.default_buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            self._maybe_start_completion(
                buffer,
                allow_slash=True,
                allow_mentions=self._mode == PromptMode.AGENT,
            )

        self._status_refresh_task: asyncio.Task[None] | None = None
        self._prompt_app: Application[str] | None = None
        self._prompt_text_area: TextArea | None = None

    def _install_slash_completion_menu(self) -> None:
        root_container = self._session.layout.container
        prompt_float_container = _find_prompt_float_container(root_container)
        outer_float_container = FloatContainer(content=root_container, floats=[])
        self._session.layout.container = outer_float_container
        self._install_slash_completion_menu_for_buffer(
            outer_float_container,
            buffer=self._session.default_buffer,
            hide_original_menu=False,
        )
        if isinstance(prompt_float_container, FloatContainer):
            self._hide_original_completion_menu(
                prompt_float_container,
                buffer=self._session.default_buffer,
            )

    def _hide_original_completion_menu(
        self,
        float_container: FloatContainer,
        *,
        buffer: Buffer,
    ) -> None:
        original_float = next(
            (
                float_
                for float_ in float_container.floats
                if isinstance(float_.content, CompletionsMenu)
            ),
            None,
        )
        if original_float is None:
            return
        original_float.content = ConditionalContainer(
            original_float.content,
            filter=~self._build_inline_completion_filter(buffer),
        )

    def _build_inline_completion_filter(self, buffer: Buffer) -> Filter:
        return has_focus(buffer) & has_completions & ~is_done

    def _build_inline_completion_menu(self, buffer: Buffer) -> ConditionalContainer:
        return ConditionalContainer(
            Window(
                content=SlashCommandMenuControl(left_padding=self._slash_menu_left_padding),
                dont_extend_height=True,
                height=Dimension(max=10),
                style="class:slash-completion-menu",
            ),
            filter=self._build_inline_completion_filter(buffer),
        )

    def _install_slash_completion_menu_for_buffer(
        self,
        float_container: FloatContainer,
        *,
        buffer: Buffer,
        hide_original_menu: bool = True,
    ) -> None:
        float_container.floats.insert(
            0,
            Float(
                left=0,
                right=0,
                ycursor=True,
                content=self._build_inline_completion_menu(buffer),
                z_index=10**8,
            ),
        )

        if hide_original_menu:
            self._hide_original_completion_menu(float_container, buffer=buffer)

    @staticmethod
    def _should_trigger_completion(
        document: Document,
        *,
        allow_slash: bool,
        allow_mentions: bool,
    ) -> bool:
        return (allow_slash and SlashCommandCompleter.should_complete(document)) or (
            allow_mentions and LocalFileMentionCompleter.should_complete(document)
        )

    def _current_slash_completer(self) -> SlashCommandCompleter:
        if self._mode == PromptMode.SHELL:
            return self._shell_slash_completer
        return self._agent_slash_completer

    def _maybe_start_completion(
        self,
        buffer: Buffer,
        *,
        allow_slash: bool,
        allow_mentions: bool,
    ) -> None:
        if not buffer.complete_while_typing():
            return
        if not self._should_trigger_completion(
            buffer.document,
            allow_slash=allow_slash,
            allow_mentions=allow_mentions,
        ):
            if buffer.complete_state is not None:
                buffer.cancel_completion()
            return
        if allow_slash and self._current_slash_completer().is_exact_match(buffer.document):
            if buffer.complete_state is not None:
                buffer.cancel_completion()
            return
        buffer.start_completion()

    def _slash_menu_left_padding(self) -> int:
        if self._mode == PromptMode.SHELL:
            return max(1, get_cwidth(f"{PROMPT_SYMBOL_SHELL} ") - 2)
        if self._status_provider().plan_mode:
            return max(1, get_cwidth(f"{PROMPT_SYMBOL_PLAN} ") - 2)
        symbol = PROMPT_SYMBOL_THINKING if self._thinking else PROMPT_SYMBOL
        return max(1, get_cwidth(f"{symbol} ") - 2)

    def _render_message(self) -> FormattedText:
        if self._mode == PromptMode.SHELL:
            return FormattedText([("bold", f"{PROMPT_SYMBOL_SHELL} ")])

        input_box_state = self._input_box_state_provider()
        if input_box_state.active:
            return self._render_active_input_box_message(input_box_state)

        status = self._status_provider()
        if status.plan_mode:
            return FormattedText([("fg:#00aaff", f"{PROMPT_SYMBOL_PLAN} ")])
        symbol = PROMPT_SYMBOL_THINKING if self._thinking else PROMPT_SYMBOL
        return FormattedText([("", f"{symbol} ")])

    @staticmethod
    def _input_box_appearance(state: InputBoxState) -> tuple[str, str, str]:
        appearances = {
            "reminder": ("REMINDER", "fg:#38bdf8 bold", "bg:#1d4ed8 #ffffff bold"),
            "approval": ("APPROVAL", "fg:#f59e0b bold", "bg:#f59e0b #111827 bold"),
            "question": ("QUESTION", "fg:#22d3ee bold", "bg:#0891b2 #ffffff bold"),
            "question_other": ("CUSTOM ANSWER", "fg:#f472b6 bold", "bg:#db2777 #ffffff bold"),
        }
        return appearances.get(state.mode, ("INPUT", "fg:#9ca3af bold", "bg:#374151 #ffffff bold"))

    @staticmethod
    def _truncate_text(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len == 1:
            return "…"
        return text[: max_len - 1] + "…"

    @staticmethod
    def _display_width(text: str) -> int:
        return get_cwidth(text)

    @classmethod
    def _truncate_display_text(cls, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if cls._display_width(text) <= max_width:
            return text
        ellipsis = "…"
        ellipsis_width = cls._display_width(ellipsis)
        if max_width <= ellipsis_width:
            return ellipsis
        available = max_width - ellipsis_width
        total = 0
        chars: list[str] = []
        for ch in text:
            ch_width = get_cwidth(ch)
            if total + ch_width > available:
                break
            chars.append(ch)
            total += ch_width
        return "".join(chars) + ellipsis

    @staticmethod
    def _take_prefix_width(text: str, width: int) -> str:
        if width <= 0:
            return ""
        total = 0
        chars: list[str] = []
        for ch in text:
            ch_width = get_cwidth(ch)
            if total + ch_width > width:
                break
            chars.append(ch)
            total += ch_width
        return "".join(chars)

    @staticmethod
    def _take_suffix_width(text: str, width: int) -> str:
        if width <= 0:
            return ""
        total = 0
        chars: deque[str] = deque()
        for ch in reversed(text):
            ch_width = get_cwidth(ch)
            if total + ch_width > width:
                break
            chars.appendleft(ch)
            total += ch_width
        return "".join(chars)

    @classmethod
    def _shorten_middle_display_text(cls, text: str, width: int) -> str:
        if width <= 0:
            return ""
        if cls._display_width(text) <= width:
            return text
        ellipsis = "..."
        ellipsis_width = cls._display_width(ellipsis)
        if width <= ellipsis_width:
            return "." * width
        available = width - ellipsis_width
        left_width = available // 2
        right_width = available - left_width
        return (
            cls._take_prefix_width(text, left_width)
            + ellipsis
            + cls._take_suffix_width(text, right_width)
        )

    def _render_active_input_box_message(self, state: InputBoxState) -> FormattedText:
        label, border_style, badge_style = self._input_box_appearance(state)
        app = get_app_or_none()
        columns = app.output.get_size().columns if app is not None else 80
        badge = f" {label} "
        used = self._display_width("╭─") + self._display_width(badge) + self._display_width("╮")
        filler = "─" * max(1, columns - used)
        return FormattedText(
            [
                (border_style, "╭─"),
                (badge_style, badge),
                (border_style, f"{filler}╮\n│ "),
            ]
        )

    def _render_prompt_continuation(
        self,
        width: int,
        line_number: int,
        wrap_count: int,
    ) -> FormattedText:
        del width, line_number, wrap_count
        state = self._input_box_state_provider()
        if state.active and self._mode != PromptMode.SHELL:
            _, border_style, _ = self._input_box_appearance(state)
            return FormattedText([(border_style, "│ ")])
        return FormattedText([("fg:#4d4d4d", "… ")])

    def _render_placeholder(self) -> FormattedText | str:
        state = self._input_box_state_provider()
        if not state.active or self._mode == PromptMode.SHELL:
            return ""
        hint = self._truncate_text(state.hint, 120)
        if not hint:
            return ""
        return FormattedText([("fg:#22d3ee italic", hint)])

    def _render_rprompt(self) -> FormattedText | str:
        state = self._input_box_state_provider()
        if not state.active or self._mode == PromptMode.SHELL:
            return ""
        _, border_style, _ = self._input_box_appearance(state)
        return FormattedText([(border_style, "│")])

    @staticmethod
    def _render_prompt_title() -> FormattedText:
        border_style = "fg:#38bdf8 bold"
        badge_style = "bg:#2563eb #ffffff bold"
        return FormattedText([(border_style, "─"), (badge_style, " PROMPT "), (border_style, "─")])

    def _render_turn_prompt_title(self, live_view: Any) -> FormattedText:
        if not getattr(live_view, "has_pending_input_request", False):
            return self._render_prompt_title()
        mode = cast(
            Literal["default", "reminder", "approval", "question", "question_other"],
            getattr(live_view, "input_mode", "default"),
        )
        label, border_style, badge_style = self._input_box_appearance(
            InputBoxState(active=True, mode=mode)
        )
        return FormattedText(
            [(border_style, "─"), (badge_style, f" {label} "), (border_style, "─")]
        )

    def _render_frame_title(self) -> FormattedText:
        return self._render_prompt_title()

    def _render_textarea_prompt(self) -> FormattedText | str:
        if self._mode == PromptMode.SHELL:
            return FormattedText([("bold", f"{PROMPT_SYMBOL_SHELL} ")])
        return ""

    def _show_agent_input_frame(self) -> bool:
        return self._mode == PromptMode.AGENT

    @staticmethod
    def _should_route_live_navigation(live_view: Any, buffer_text: str) -> bool:
        if not live_view.has_pending_input_request:
            return False
        if live_view.input_mode == "question_other":
            return False
        return not buffer_text.strip()

    def _render_hint_line(self, text_area: Any) -> FormattedText | str:
        if self._mode != PromptMode.AGENT:
            return ""
        if text_area.buffer.text:
            return ""
        state = self._input_box_state_provider()
        if state.mode == "reminder":
            return ""
        hint = state.hint.strip()
        if not hint:
            return ""
        hint = self._truncate_text(hint, 160)
        return FormattedText([("fg:#22d3ee italic", hint)])

    def _render_footer_line(self) -> FormattedText:
        status = self._status_provider()
        right_text = self._render_right_span(status)
        app = get_app_or_none()
        columns = app.output.get_size().columns if app is not None else 80
        left_text = self._render_footer_left_text(
            status=status,
            columns=max(1, columns - self._display_width("  ")),
            right_text=right_text,
        )
        padding = max(
            1,
            columns - self._display_width(left_text) - self._display_width(right_text),
        )
        return FormattedText(
            [
                ("fg:#38bdf8 bold", left_text),
                ("", " " * padding),
                ("fg:#9ca3af", right_text),
            ]
        )

    @staticmethod
    def _toast_messages() -> tuple[str | None, str | None]:
        left_toast = _current_toast("left")
        right_toast = _current_toast("right")
        return (
            left_toast.message if left_toast is not None else None,
            right_toast.message if right_toast is not None else None,
        )

    def _has_toasts(self) -> bool:
        left_text, right_text = self._toast_messages()
        return bool(left_text or right_text)

    def _render_toast_line(self, columns: int | None = None) -> FormattedText | str:
        left_text, right_text = self._toast_messages()
        if not left_text and not right_text:
            return ""

        app = get_app_or_none()
        total_columns = (
            columns
            if columns is not None
            else (app.output.get_size().columns if app is not None else 80)
        )
        total_columns = max(1, total_columns)

        available = total_columns
        rendered_right = ""
        if right_text:
            rendered_right = self._truncate_display_text(right_text, max(0, total_columns // 2))
            available = max(0, total_columns - self._display_width(rendered_right) - 2)

        rendered_left = ""
        if left_text:
            rendered_left = self._truncate_display_text(left_text, available)

        fragments: list[tuple[str, str]] = []
        if rendered_left:
            fragments.append(("fg:#cbd5e1 italic", rendered_left))
        if rendered_right:
            if rendered_left:
                padding = max(
                    2,
                    total_columns
                    - self._display_width(rendered_left)
                    - self._display_width(rendered_right),
                )
                fragments.append(("", " " * padding))
            fragments.append(("fg:#cbd5e1 italic", rendered_right))
        return FormattedText(fragments)

    @staticmethod
    def _with_completion_menu(content: Any) -> FloatContainer:
        return FloatContainer(
            content=content,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
        )

    def _build_prompt_application(self) -> tuple[Application[str], TextArea]:
        last_layout_signature: tuple[object, ...] | None = None

        def _app_columns(app: Application[Any] | None = None) -> int:
            current_app = get_app_or_none() if app is None else app
            return current_app.output.get_size().columns if current_app is not None else 80

        text_area = TextArea(
            text="",
            multiline=True,
            completer=(
                self._agent_mode_completer
                if self._mode == PromptMode.AGENT
                else self._shell_mode_completer
            ),
            complete_while_typing=True,
            history=self._history,
            prompt=self._render_textarea_prompt,
            wrap_lines=True,
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
        )

        def _prompt_layout_signature() -> tuple[object, ...]:
            input_height = (
                text_area.window.render_info.window_height
                if text_area.window.render_info is not None
                else max(1, text_area.buffer.document.line_count)
            )
            has_hint = bool(self._show_agent_input_frame() and self._render_hint_line(text_area))
            return (
                self._mode,
                _app_columns(),
                has_hint,
                input_height,
            )

        def _redraw_prompt_view(app: Application[Any]) -> None:
            nonlocal last_layout_signature
            last_layout_signature = self._redraw_for_layout_change(
                app,
                signature=_prompt_layout_signature(),
                last_signature=last_layout_signature,
            )

        @text_area.buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            self._maybe_start_completion(
                buffer,
                allow_slash=True,
                allow_mentions=self._mode == PromptMode.AGENT,
            )
            app = get_app_or_none()
            if app is not None:
                _redraw_prompt_view(app)

        hint_window = Window(
            FormattedTextControl(lambda: self._render_hint_line(text_area)),
            height=1,
            dont_extend_height=True,
        )
        agent_toast_window = Window(
            FormattedTextControl(lambda: self._render_toast_line(_app_columns())),
            height=1,
            dont_extend_height=True,
        )
        shell_toast_window = Window(
            FormattedTextControl(lambda: self._render_toast_line(_app_columns())),
            height=1,
            dont_extend_height=True,
        )
        footer_window = Window(
            FormattedTextControl(self._render_footer_line),
            height=1,
            dont_extend_height=True,
        )
        spacer = Window(height=Dimension(weight=1), char=" ")
        agent_completion_menu = self._build_inline_completion_menu(text_area.buffer)
        shell_completion_menu = self._build_inline_completion_menu(text_area.buffer)
        agent_container = HSplit(
            [
                spacer,
                agent_completion_menu,
                Frame(
                    HSplit(
                        [
                            ConditionalContainer(
                                hint_window,
                                filter=Condition(
                                    lambda: self._show_agent_input_frame()
                                    and bool(self._render_hint_line(text_area))
                                ),
                            ),
                            text_area,
                        ]
                    ),
                    title=self._render_frame_title,
                    style="fg:#38bdf8",
                ),
                ConditionalContainer(
                    agent_toast_window,
                    filter=Condition(self._has_toasts),
                ),
                footer_window,
            ]
        )
        shell_container = HSplit(
            [
                spacer,
                shell_completion_menu,
                text_area,
                ConditionalContainer(
                    shell_toast_window,
                    filter=Condition(self._has_toasts),
                ),
            ]
        )
        container = self._with_completion_menu(
            DynamicContainer(
                lambda: agent_container if self._show_agent_input_frame() else shell_container
            )
        )
        self._hide_original_completion_menu(container, buffer=text_area.buffer)

        accept_kb = KeyBindings()

        @accept_kb.add("enter", filter=has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            buff = event.current_buffer
            if not (buff.complete_state and buff.complete_state.completions):
                return
            if self._current_slash_completer().is_exact_match(buff.document):
                event.app.exit(result=text_area.buffer.text)
                return
            completion = buff.complete_state.current_completion
            if not completion:
                completion = buff.complete_state.completions[0]
            buff.apply_completion(completion)

        @accept_kb.add("enter", filter=~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            event.app.exit(result=text_area.buffer.text)

        @accept_kb.add("c-d", eager=True)
        def _(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            if not buffer.text:
                event.app.exit(exception=EOFError)
                return
            buffer.delete()

        @accept_kb.add("c-c", eager=True)
        def _(event: KeyPressEvent) -> None:
            if self._dismiss_input(event.current_buffer):
                event.app.invalidate()
                return
            event.app.exit(exception=KeyboardInterrupt)

        app = Application[str](
            layout=Layout(container, focused_element=text_area),
            key_bindings=merge_key_bindings([self._base_key_bindings, accept_kb]),
            clipboard=self._clipboard,
            cursor=STEADY_INPUT_CURSOR,
            style=Style.from_dict(_shell_style_dict()),
            full_screen=False,
            erase_when_done=True,
            refresh_interval=None,
            terminal_size_polling_interval=_TERMINAL_SIZE_POLLING_INTERVAL,
        )
        self._install_deferred_erase(app)
        last_layout_signature = _prompt_layout_signature()
        return app, text_area

    def _install_deferred_erase(self, app: Application[Any]) -> None:
        """Monkey-patch the prompt app renderer to defer erase on exit.

        Keeps the input box on-screen until `execute_deferred_erase` is
        called (right before the turn UI starts), eliminating the flash.

        IMPORTANT — resize correctness
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        prompt_toolkit's ``Application._on_resize()`` calls
        ``renderer.erase()`` to wipe the old content before redrawing at
        the new terminal dimensions.  If we unconditionally defer *every*
        erase call, the old content is never cleared and the subsequent
        redraw lands at a different cursor position — producing the
        "ghost frames" / screen corruption visible on window resize.

        The fix: only defer the erase when the app is **shutting down**
        (``app._is_running is False``).  While the app is still running
        (resize via ``_on_resize``, or manual ``Ctrl-L`` hard-redraw),
        the original erase executes immediately so prompt_toolkit's
        differential renderer starts from a clean slate.

        Do NOT remove the ``app._is_running`` guard without verifying
        that terminal resize no longer causes visual artefacts.
        """
        renderer = app.renderer
        original_erase = renderer.erase

        def _deferred_erase(leave_alternate_screen: bool = True) -> None:
            # ── Resize / hard-redraw guard ──────────────────────────
            # While the app is still running, renderer.erase() is being
            # called by _on_resize (SIGWINCH) or _hard_redraw (Ctrl-L).
            # We MUST execute the real erase here; deferring it leaves
            # stale content on screen and causes visible corruption.
            if app._is_running:
                original_erase(leave_alternate_screen=leave_alternate_screen)
                return
            # ── App exit path (erase_when_done) ────────────────────
            # The app is no longer running — this is the final cleanup
            # erase triggered by erase_when_done=True.  Defer it so the
            # input box stays visible until the turn UI seamlessly takes
            # over (via execute_deferred_erase).
            x, y = renderer._cursor_pos.x, renderer._cursor_pos.y
            # (0,0) means the app hasn't rendered yet — this happens when
            # _hard_redraw calls erase() before the first render.  Fall
            # back to the real erase so _hard_redraw keeps working.
            if x == 0 and y == 0:
                original_erase(leave_alternate_screen=leave_alternate_screen)
                return
            self._deferred_erase_x = x
            self._deferred_erase_y = y
            self._deferred_erase_pending = True
            output = renderer.output
            output.reset_attributes()
            output.enable_autowrap()
            output.flush()
            renderer.reset(leave_alternate_screen=leave_alternate_screen)

        renderer.erase = _deferred_erase  # type: ignore[method-assign]

    def execute_deferred_erase(self) -> None:
        """Perform the erase that was deferred when the prompt app exited."""
        if not self._deferred_erase_pending:
            return
        import sys

        if self._deferred_erase_x > 0:
            sys.stdout.write(f"\033[{self._deferred_erase_x}D")
        if self._deferred_erase_y > 0:
            sys.stdout.write(f"\033[{self._deferred_erase_y}A")
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        self._deferred_erase_pending = False

    def _get_prompt_application(self) -> tuple[Application[str], TextArea]:
        if self._prompt_app is None or self._prompt_text_area is None:
            self._prompt_app, self._prompt_text_area = self._build_prompt_application()
        return self._prompt_app, self._prompt_text_area

    def _prepare_prompt_application(self) -> tuple[Application[str], TextArea]:
        app, text_area = self._get_prompt_application()
        self._apply_mode_to_buffer(text_area.buffer)
        if text_area.buffer.complete_state is not None:
            text_area.buffer.cancel_completion()
        text_area.buffer.document = Document(text="", cursor_position=0)
        self._hard_redraw(app)
        return app, text_area

    def _open_in_external_editor(self, event: KeyPressEvent) -> None:
        """Open the current buffer content in an external editor."""
        from prompt_toolkit.application.run_in_terminal import run_in_terminal

        from kimi_cli.utils.editor import edit_text_in_editor, get_editor_command

        configured = self._editor_command_provider()

        if get_editor_command(configured) is None:
            toast("No editor found. Set $VISUAL/$EDITOR or run /editor.")
            return

        buff = event.current_buffer
        original_text = buff.text
        editor_text = self._get_placeholder_manager().expand_for_editor(original_text)

        async def _run_editor() -> None:
            result = await run_in_terminal(
                lambda: edit_text_in_editor(editor_text, configured), in_executor=True
            )
            if result is not None:
                refolded = self._get_placeholder_manager().refold_after_editor(
                    result, original_text
                )
                buff.document = Document(text=refolded, cursor_position=len(refolded))

        event.app.create_background_task(_run_editor())

    def _open_live_view_expansion(self, event: KeyPressEvent, live_view: Any) -> None:
        """Open the current approval/question expansion inside the active PTK app."""
        if live_view.show_more():
            event.current_buffer.document = Document(text="", cursor_position=0)
            event.app.invalidate()

    def _apply_mode_to_buffer(self, buff: Buffer | None) -> None:
        if buff is None:
            return
        if self._mode == PromptMode.SHELL:
            buff.completer = self._shell_mode_completer
            return
        buff.completer = self._agent_mode_completer

    def _apply_mode(self, event: KeyPressEvent | None = None) -> None:
        # Apply mode to the active buffer (not the PromptSession itself)
        try:
            buff = event.current_buffer if event is not None else None
        except Exception:
            buff = None
        if buff is None and self._prompt_text_area is not None:
            buff = self._prompt_text_area.buffer
        if buff is None:
            buff = self._session.default_buffer
        self._apply_mode_to_buffer(buff)

    def _render_signature(self) -> tuple[object, ...]:
        status = self._status_provider()
        input_box_state = self._input_box_state_provider()
        left_toast = _current_toast("left")
        right_toast = _current_toast("right")
        return (
            self._mode,
            self._thinking,
            self._model_name,
            self._working_dir_text(),
            input_box_state,
            status.context_usage,
            status.context_tokens,
            status.max_context_tokens,
            status.yolo_enabled,
            status.plan_mode,
            left_toast.message if left_toast is not None else None,
            right_toast.message if right_toast is not None else None,
        )

    @staticmethod
    def _live_indicator_frame(kind: str, *, now: float | None = None) -> str:
        frames = _INDICATOR_FRAMES.get(kind, _PAUSE_FRAMES)
        current = time.monotonic() if now is None else now
        return frames[int(current * 10) % len(frames)]

    @staticmethod
    def _force_turn_full_repaint(app: Any) -> None:
        app.renderer._last_screen = None
        app.invalidate()

    @classmethod
    def _hard_redraw(cls, app: Any) -> None:
        renderer = getattr(app, "renderer", None)
        erase = getattr(renderer, "erase", None)
        if callable(erase):
            erase(leave_alternate_screen=False)
            invalidate = getattr(app, "invalidate", None)
            if callable(invalidate):
                invalidate()
            return
        cls._force_turn_full_repaint(app)

    @classmethod
    def _redraw_for_layout_change(
        cls,
        app: Any,
        *,
        signature: tuple[object, ...],
        last_signature: tuple[object, ...] | None,
        full_repaint_on_change: bool = True,
    ) -> tuple[object, ...]:
        if last_signature is None:
            app.invalidate()
            return signature
        if signature != last_signature:
            if full_repaint_on_change:
                cls._force_turn_full_repaint(app)
            else:
                app.invalidate()
            return signature
        app.invalidate()
        return last_signature

    @staticmethod
    def _turn_ui_refresh_interval(live_view: Any, *, burst_active: bool = False) -> float:
        indicator = getattr(live_view, "activity_indicator", None)
        if indicator is None:
            return _TURN_UI_REFRESH_INTERVAL
        kind, _ = indicator
        if kind == "composing":
            return (
                _TURN_UI_BURST_REFRESH_INTERVAL if burst_active else _TURN_UI_FAST_REFRESH_INTERVAL
            )
        if kind == "thinking":
            return _TURN_UI_FAST_REFRESH_INTERVAL
        return _TURN_UI_REFRESH_INTERVAL

    @staticmethod
    def _turn_stream_push_thresholds(
        live_view: Any,
        *,
        burst_active: bool = False,
    ) -> tuple[int, int]:
        indicator = getattr(live_view, "activity_indicator", None)
        if indicator is None:
            return _TURN_UI_STREAM_PUSH_PARTS, _TURN_UI_STREAM_PUSH_CHARS
        kind, _ = indicator
        if kind == "composing":
            if burst_active:
                return _TURN_UI_BURST_STREAM_PUSH_PARTS, _TURN_UI_BURST_STREAM_PUSH_CHARS
            return _TURN_UI_COMPOSE_STREAM_PUSH_PARTS, _TURN_UI_COMPOSE_STREAM_PUSH_CHARS
        if kind == "thinking":
            return _TURN_UI_FAST_STREAM_PUSH_PARTS, _TURN_UI_FAST_STREAM_PUSH_CHARS
        return _TURN_UI_STREAM_PUSH_PARTS, _TURN_UI_STREAM_PUSH_CHARS

    @staticmethod
    def _turn_stream_refresh_metrics(msg: object) -> tuple[int, int, bool]:
        match msg:
            case TextPart(text=text) | ThinkPart(think=text):
                return 1, len(text), "\n" in text
            case ToolCallPart(arguments_part=arguments_part):
                if not arguments_part:
                    return 0, 0, False
                return 1, len(arguments_part), "\n" in arguments_part
            case _:
                return 0, 0, False

    @staticmethod
    def _turn_stream_burst_kind(msg: object) -> str | None:
        match msg:
            case TextPart(text=text) if text:
                return "composing"
            case _:
                return None

    @staticmethod
    def _should_start_turn_stream_burst(
        *,
        active_kind: str | None,
        msg_kind: str | None,
        current_burst_kind: str | None,
        now: float,
        burst_until: float,
        last_burstable_stream_at: float,
    ) -> bool:
        if active_kind != "composing" or msg_kind != "composing":
            return False
        if current_burst_kind != "composing":
            return True
        if now < burst_until:
            return False
        if last_burstable_stream_at <= 0:
            return False
        return now - last_burstable_stream_at >= _TURN_UI_STREAM_RESUME_BURST_GAP

    def _refresh_turn_application(
        self,
        app: Any,
        *,
        live_view: Any,
    ) -> bool:
        has_toasts = self._has_toasts()
        had_toasts = bool(getattr(app, "_kimi_had_toasts", False))
        app._kimi_had_toasts = has_toasts
        if (
            not getattr(live_view, "needs_periodic_refresh", False)
            and not has_toasts
            and not had_toasts
        ):
            return False
        app.invalidate()
        return True

    @staticmethod
    def _dismiss_input(buffer: Buffer) -> bool:
        handled = False
        if buffer.complete_state is not None:
            buffer.cancel_completion()
            handled = True
        if buffer.text:
            buffer.document = Document(text="", cursor_position=0)
            handled = True
        return handled

    def _format_live_activity_status(self, live_view: Any) -> tuple[str, str] | None:
        indicator = getattr(live_view, "activity_indicator", None)
        if indicator is None:
            return None
        kind, text = indicator
        if not text:
            return None
        frame = self._live_indicator_frame(kind)
        return f"{frame} {text}", _INDICATOR_STYLES.get(kind, "fg:#22c55e")

    def _render_turn_activity(self, live_view: Any) -> FormattedText | str:
        if getattr(live_view, "has_pending_input_request", False):
            return ""
        live_status = self._format_live_activity_status(live_view)
        if live_status is None:
            return ""
        status_text, status_style = live_status
        return FormattedText(
            [
                ("fg:#6b7280", "│ "),
                (status_style, status_text),
            ]
        )

    def _turn_input_hint_text(
        self,
        *,
        live_view: Any,
        buffer_text: str,
        feedback_message: str,
    ) -> str:
        input_mode = getattr(live_view, "input_mode", "reminder")
        if buffer_text and input_mode == "reminder":
            return ""
        if input_mode == "reminder":
            return ""
        hint = feedback_message or getattr(live_view, "input_hint", "")
        if not hint:
            return ""
        return self._truncate_text(hint, 160)

    @staticmethod
    def _turn_body_cursor_line(
        line_count: int,
        *,
        has_pending_input_request: bool,
        reveal_latest_output: bool,
        history_view_enabled: bool = False,
    ) -> int | None:
        if line_count <= 0:
            return None
        if history_view_enabled:
            return 0
        if has_pending_input_request and not reveal_latest_output:
            return 0
        return line_count - 1

    @staticmethod
    def _format_history_view_position(*, top_line: int, total_lines: int) -> str:
        if total_lines <= 0:
            return _TURN_HISTORY_VIEW_EMPTY_POSITION
        return f"{max(1, min(top_line, total_lines))}/{total_lines}"

    @classmethod
    def _format_history_view_hint(cls, *, top_line: int, total_lines: int) -> str:
        position = cls._format_history_view_position(top_line=top_line, total_lines=total_lines)
        return f"History {position} — ↑/↓, PgUp/PgDn, Home/End, Ctrl-Y live"

    @classmethod
    def _format_live_turn_hint(
        cls,
        *,
        input_hint: str = "",
        show_recent_output_notice: bool = False,
    ) -> str:
        parts = [input_hint.strip()] if input_hint.strip() else []
        if show_recent_output_notice:
            parts.append(_recent_output_notice_text())
        if not parts:
            return ""
        return cls._truncate_text(" · ".join(parts), 160)

    @staticmethod
    def _shorten_footer_path(path: str, width: int) -> str:
        if width <= 0:
            return ""
        if width <= 4:
            return CustomPromptSession._truncate_display_text(path, width)
        return CustomPromptSession._shorten_middle_display_text(path, width)

    def _working_dir_text(self) -> str:
        provider = getattr(self, "_working_dir_provider", None)
        if provider is None:
            return str(KaosPath.cwd())
        return provider().strip()

    def _render_footer_left_text(
        self,
        *,
        status: StatusSnapshot,
        columns: int,
        right_text: str,
        prefix: str = "",
    ) -> str:
        mode_text = self._mode_text(status)
        available = max(1, columns - self._display_width(right_text) - 1)
        if prefix:
            available = max(1, available - self._display_width(prefix))
        base_text = self._truncate_display_text(mode_text, available)
        working_dir = self._working_dir_text()
        if not working_dir:
            return f"{prefix}{base_text}"

        separator = " · "
        mode_width = self._display_width(mode_text)
        separator_width = self._display_width(separator)
        if available <= mode_width + separator_width:
            return f"{prefix}{base_text}"

        path_width = available - mode_width - separator_width
        path_text = self._shorten_footer_path(working_dir, path_width)
        if not path_text:
            return f"{prefix}{base_text}"
        return f"{prefix}{mode_text}{separator}{path_text}"

    def _render_turn_footer(
        self,
        columns: int,
        *,
        status: StatusSnapshot,
    ) -> FormattedText:
        right_text = self._render_right_span(status)
        left_text = self._render_footer_left_text(
            status=status,
            columns=max(1, columns - self._display_width("  ")),
            right_text=right_text,
        )
        fragments: list[tuple[str, str]] = [("fg:#38bdf8 bold", left_text)]
        remaining = columns - self._display_width(left_text) - self._display_width(right_text)
        fragments.append(("", " " * max(1, remaining)))
        fragments.append(("fg:#9ca3af", right_text))
        return FormattedText(fragments)

    def _install_focus_repaint(self) -> None:
        """Monkey-patch the shared Vt100 parser to detect focus-in events.

        When the terminal sends CSI I (focus gained), we force a full
        repaint so the differential renderer doesn't produce artefacts.
        """
        import sys as _sys

        if not _sys.stdin.isatty() or not _sys.stdout.isatty():
            return

        try:
            from prompt_toolkit.input.vt100 import Vt100Input
        except ImportError:
            return

        vt100_input = getattr(self._session.app, "input", None)
        if not isinstance(vt100_input, Vt100Input):
            return

        parser = vt100_input.vt100_parser
        original_callback = parser.feed_key_callback

        def _focus_aware_callback(key_press: Any) -> None:
            if (
                getattr(key_press, "key", None) == Keys.Ignore
                and getattr(key_press, "data", None) == _FOCUS_IN_SEQ
            ):
                app = get_app_or_none()
                if app is not None:
                    self._force_turn_full_repaint(app)
                return  # swallow the event
            original_callback(key_press)

        parser.feed_key_callback = _focus_aware_callback
        self._original_parser_callback = original_callback
        self._focus_parser = parser

    def _uninstall_focus_repaint(self) -> None:
        parser = getattr(self, "_focus_parser", None)
        original = getattr(self, "_original_parser_callback", None)
        if parser is not None and original is not None:
            parser.feed_key_callback = original
        self._focus_parser = None
        self._original_parser_callback = None

    def __enter__(self) -> CustomPromptSession:
        # Enable terminal focus reporting so we receive \x1b[I / \x1b[O
        # when the terminal window gains / loses focus.  This lets us
        # trigger a full repaint on focus-in and avoid screen corruption
        # after the user switches away and back.
        import sys as _sys

        if _sys.stdout.isatty():
            _sys.stdout.write("\x1b[?1004h")
            _sys.stdout.flush()

        self._install_focus_repaint()

        if self._status_refresh_task is not None and not self._status_refresh_task.done():
            return self

        async def _refresh(interval: float) -> None:
            last_signature: tuple[object, ...] | None = None
            try:
                while True:
                    app = get_app_or_none()
                    if app is not None:
                        signature = self._render_signature()
                        if signature != last_signature:
                            app.invalidate()
                            last_signature = signature

                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No running loop found, exiting status refresh task")
                        self._status_refresh_task = None
                        break

                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                # graceful exit
                pass

        self._status_refresh_task = asyncio.create_task(_refresh(_REFRESH_INTERVAL))
        return self

    def __exit__(self, *_) -> None:
        self._uninstall_focus_repaint()

        # Disable terminal focus reporting.
        import sys as _sys

        if _sys.stdout.isatty():
            _sys.stdout.write("\x1b[?1004l")
            _sys.stdout.flush()

        if self._status_refresh_task is not None and not self._status_refresh_task.done():
            self._status_refresh_task.cancel()
        self._status_refresh_task = None

    def _get_placeholder_manager(self) -> PromptPlaceholderManager:
        manager = getattr(self, "_placeholder_manager", None)
        if manager is None:
            attachment_cache = getattr(self, "_attachment_cache", None)
            manager = PromptPlaceholderManager(attachment_cache=attachment_cache)
            self._placeholder_manager = manager
            self._attachment_cache = manager.attachment_cache
        return manager

    def _insert_pasted_text(self, buffer: Buffer, text: str) -> None:
        normalized = normalize_pasted_text(text)
        if self._mode != PromptMode.AGENT:
            buffer.insert_text(normalized)
            return
        token_or_text = self._get_placeholder_manager().maybe_placeholderize_pasted_text(normalized)
        buffer.insert_text(token_or_text)

    def _handle_bracketed_paste(self, event: KeyPressEvent) -> None:
        self._insert_pasted_text(event.current_buffer, event.data)
        event.app.invalidate()

    def _try_paste_media(self, event: KeyPressEvent) -> bool:
        """Try to paste media from the clipboard.

        Reads the clipboard once and handles all detected content:
        non-image files (videos, PDFs, etc.) are inserted as paths,
        image files are cached and inserted as placeholders.
        Returns True if any media was detected.
        """
        result = grab_media_from_clipboard()
        if result is None:
            return False

        parts: list[str] = []

        # 1. Insert file paths (videos, PDFs, etc.)
        if result.file_paths:
            logger.debug("Pasted {count} file path(s) from clipboard", count=len(result.file_paths))
            for p in result.file_paths:
                text = str(p)
                if self._mode == PromptMode.SHELL:
                    text = shlex.quote(text)
                parts.append(text)

        # 2. Insert images via cache.
        if result.images:
            if "image_in" not in self._model_capabilities:
                console.print(
                    "[yellow]Image input is not supported by the selected LLM model[/yellow]"
                )
            else:
                for image in result.images:
                    token = self._get_placeholder_manager().create_image_placeholder(image)
                    if token is None:
                        continue
                    logger.debug(
                        "Pasted image from clipboard placeholder: {token}, {image_size}",
                        token=token,
                        image_size=image.size,
                    )
                    parts.append(token)

        if parts:
            event.current_buffer.insert_text(" ".join(parts))
        event.app.invalidate()
        return bool(parts)

    def _build_user_input(self, command: str) -> UserInput:
        resolved = self._get_placeholder_manager().resolve_command(command)

        return UserInput(
            mode=self._mode,
            command=resolved.display_command,
            resolved_command=resolved.resolved_text,
            content=resolved.content,
        )

    async def prompt(self) -> UserInput:
        restore_text = ""
        while True:
            app, text_area = self._prepare_prompt_application()
            if restore_text:
                text_area.buffer.document = Document(
                    text=restore_text, cursor_position=len(restore_text)
                )
                restore_text = ""
            try:
                with patch_stdout(raw=True):
                    command = str(await app.run_async()).strip()
            except _ClearScreenRequest as req:
                import sys

                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                if self._redraw_callback is not None:
                    await self._redraw_callback()
                restore_text = req.buffer_text
                continue
            self._append_history_entry(command)
            self._tip_rotation_index += 1
            return self._build_user_input(command)

    async def run_turn_ui(
        self,
        *,
        wire: Any,
        live_view: Any,
        submit_handler: Callable[[UserInput], TurnSubmitResult],
        cancel_handler: Callable[[], None],
        turn_prompt: str | None = None,
        pre_rendered_echo: str = "",
    ) -> None:
        self._mode = PromptMode.AGENT
        feedback_message = ""
        reveal_latest_output = False
        history_view_enabled = False
        history_view_snapshot: Any = None
        history_view_revision = 0
        history_view_scroll_offset = 0
        body_window: Window | None = None
        last_layout_signature: tuple[object, ...] | None = None
        stream_refresh_parts = 0
        stream_refresh_chars = 0
        last_stream_refresh_at = 0.0
        stream_burst_kind: str | None = None
        stream_burst_until = 0.0
        last_burstable_stream_at = 0.0

        def _app_columns(app: Application[Any] | None = None) -> int:
            current_app = get_app_or_none() if app is None else app
            return current_app.output.get_size().columns if current_app is not None else 80

        def _app_rows(app: Application[Any] | None = None) -> int:
            current_app = get_app_or_none() if app is None else app
            return current_app.output.get_size().rows if current_app is not None else 24

        prompt_block_control = _RichRenderableControl(
            lambda: render_user_prompt_block(turn_prompt) if turn_prompt else None,
            get_cache_revision=lambda: turn_prompt,
        )

        text_area = TextArea(
            text="",
            multiline=True,
            completer=self._turn_mode_completer,
            complete_while_typing=True,
            history=self._history,
            wrap_lines=True,
            height=Dimension(min=1, max=6),
            dont_extend_height=True,
        )

        def _input_box_height() -> int:
            columns = max(1, _app_columns() - 4)
            return text_area.window.preferred_height(columns, 6).preferred

        @text_area.buffer.on_text_changed.add_handler
        def _(buffer: Buffer) -> None:
            _clear_turn_output_reveal()
            self._maybe_start_completion(
                buffer,
                allow_slash=True,
                allow_mentions=True,
            )
            app = get_app_or_none()
            if app is not None:
                _redraw_turn_view(app)

        def _render_title() -> FormattedText:
            return self._render_turn_prompt_title(live_view)

        def _turn_hint_text() -> str:
            if reveal_latest_output and live_view.has_pending_input_request:
                return "Ctrl-E to return · Ctrl-Y for full history"
            if history_view_enabled and not text_area.buffer.text:
                if feedback_message:
                    return self._truncate_text(feedback_message, 160)
                top_line, total_lines = _history_view_position()
                return self._format_history_view_hint(top_line=top_line, total_lines=total_lines)
            return self._format_live_turn_hint(
                input_hint=self._turn_input_hint_text(
                    live_view=live_view,
                    buffer_text=text_area.buffer.text,
                    feedback_message=feedback_message,
                ),
                show_recent_output_notice=live_view.should_show_recent_output_notice(
                    tail_block_limit=_turn_tail_block_limit(include_hint=False),
                    content_char_limit=MAX_ACTIVE_TURN_CONTENT_CHARS,
                ),
            )

        def _render_hint() -> FormattedText | str:
            hint = _turn_hint_text()
            if not hint:
                return ""
            return FormattedText([("fg:#22d3ee italic", hint)])

        def _render_footer() -> FormattedText:
            status = self._status_provider()
            return self._render_turn_footer(_app_columns(), status=status)

        def _turn_prompt_height(width: int) -> int:
            if not turn_prompt:
                return 0
            return prompt_block_control.line_count(width)

        def _turn_completion_menu_height() -> int:
            complete_state = text_area.buffer.complete_state
            if complete_state is None or not complete_state.completions:
                return 0
            return min(10, len(complete_state.completions))

        def _turn_fixed_height(width: int, *, include_hint: bool = True) -> int:
            input_height = (
                text_area.window.render_info.window_height
                if text_area.window.render_info is not None
                else _input_box_height()
            )
            hint_height = 1 if include_hint and bool(_turn_hint_text()) else 0
            activity_height = 1 if _has_turn_activity() else 0
            toast_height = 1 if self._has_toasts() else 0
            footer_height = 1
            input_frame_height = input_height + hint_height + 2
            return (
                _turn_prompt_height(width)
                + activity_height
                + _turn_completion_menu_height()
                + input_frame_height
                + toast_height
                + footer_height
            )

        def _turn_body_line_budget(
            width: int | None = None,
            *,
            include_hint: bool = True,
        ) -> int:
            body_width = width or (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )
            return max(
                1,
                _app_rows() - _turn_fixed_height(body_width, include_hint=include_hint),
            )

        def _turn_tail_block_limit(*, include_hint: bool = True) -> int:
            if live_view.has_pending_input_request and not reveal_latest_output:
                return 0
            if live_view.has_pending_input_request:
                return MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS
            return max(
                MAX_ACTIVE_TURN_FLUSHED_BLOCKS,
                _turn_body_line_budget(include_hint=include_hint),
            )

        def _history_view_body_width() -> int:
            return (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )

        def _history_view_position() -> tuple[int, int]:
            body_width = _history_view_body_width()
            total_lines = history_body_control.line_count(body_width)
            top_line = history_view_scroll_offset + 1 if total_lines > 0 else 0
            return top_line, total_lines

        history_body_control = _RichRenderableControl(
            lambda: (
                history_view_snapshot
                if history_view_enabled
                else live_view.compose_history_body(
                    tail_block_limit=_turn_tail_block_limit(),
                )
            ),
            get_cache_revision=lambda: (
                history_view_enabled,
                history_view_revision
                if history_view_enabled
                else getattr(live_view, "history_revision", 0),
                None if history_view_enabled else _turn_tail_block_limit(),
            ),
        )
        active_body_control = _RichRenderableControl(
            lambda: (
                None
                if history_view_enabled
                else live_view.compose_active_body(
                    include_running_indicators=False,
                    content_char_limit=MAX_ACTIVE_TURN_CONTENT_CHARS,
                    focus_pending_input_panel=getattr(live_view, "is_inline_panel_expanded", False)
                    or not reveal_latest_output,
                )
            ),
            get_cache_revision=lambda: (
                history_view_enabled,
                getattr(live_view, "active_revision", 0),
                live_view.has_pending_input_request,
                live_view.input_mode,
                reveal_latest_output,
            ),
        )

        def _turn_body_cursor_line(line_count: int) -> int | None:
            if history_view_enabled:
                return history_view_scroll_offset
            if getattr(live_view, "is_inline_panel_expanded", False):
                return None
            return self._turn_body_cursor_line(
                line_count,
                has_pending_input_request=live_view.has_pending_input_request,
                reveal_latest_output=reveal_latest_output,
            )

        body_control = _StackedRichRenderableControl(
            [
                history_body_control,
                active_body_control,
            ],
            get_cursor_line=_turn_body_cursor_line,
            get_max_line_count=_turn_body_line_budget,
            get_window_start=lambda _line_count, _visible_count: (
                history_view_scroll_offset
                if history_view_enabled
                else getattr(live_view, "inline_expanded_scroll_offset", None)
                if getattr(live_view, "is_inline_panel_expanded", False)
                else None
            ),
        )

        def _history_view_total_lines(width: int | None = None) -> int:
            body_width = width or (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )
            return body_control.total_line_count(body_width)

        def _history_view_visible_lines(width: int | None = None) -> int:
            body_width = width or (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )
            return max(1, body_control.line_count(body_width))

        def _clamp_history_view_scroll_offset(value: int, width: int | None = None) -> int:
            total_lines = _history_view_total_lines(width)
            visible_lines = _history_view_visible_lines(width)
            max_start = max(0, total_lines - visible_lines)
            return max(0, min(max_start, value))

        def _scroll_history_view(delta: int) -> None:
            nonlocal history_view_scroll_offset, history_view_revision
            next_offset = _clamp_history_view_scroll_offset(history_view_scroll_offset + delta)
            if next_offset == history_view_scroll_offset:
                return
            history_view_scroll_offset = next_offset
            history_view_revision += 1

        def _page_history_view(direction: int) -> None:
            page_size = max(1, _history_view_visible_lines() - 1)
            _scroll_history_view(direction * page_size)

        def _jump_history_view(to_end: bool) -> None:
            nonlocal history_view_scroll_offset, history_view_revision
            next_offset = _clamp_history_view_scroll_offset(
                _history_view_total_lines() if to_end else 0
            )
            if next_offset == history_view_scroll_offset:
                return
            history_view_scroll_offset = next_offset
            history_view_revision += 1

        def _turn_layout_signature() -> tuple[object, ...]:
            body_width = (
                body_window.render_info.window_width
                if body_window is not None and body_window.render_info is not None
                else _app_columns()
            )
            input_height = (
                text_area.window.render_info.window_height
                if text_area.window.render_info is not None
                else _input_box_height()
            )
            body_line_count = body_control.line_count(body_width)
            has_activity = _has_turn_activity()
            has_hint = bool(_turn_hint_text())
            return (
                getattr(live_view, "render_revision", 0),
                history_view_enabled,
                history_view_revision if history_view_enabled else None,
                history_view_scroll_offset if history_view_enabled else None,
                body_width,
                _app_rows(),
                _turn_body_line_budget(body_width),
                body_line_count,
                has_activity,
                has_hint,
                input_height,
            )

        def _redraw_turn_view(app: Application[Any]) -> None:
            nonlocal last_layout_signature
            last_layout_signature = self._redraw_for_layout_change(
                app,
                signature=_turn_layout_signature(),
                last_signature=last_layout_signature,
                full_repaint_on_change=False,
            )

        def _refresh_turn_view(app: Application[Any]) -> None:
            _redraw_turn_view(app)

        def _turn_stream_activity_kind() -> str | None:
            indicator = getattr(live_view, "activity_indicator", None)
            if indicator is None:
                return None
            kind, _ = indicator
            if kind in {"thinking", "composing"}:
                return kind
            return None

        def _turn_stream_burst_active(now: float | None = None) -> bool:
            current = time.monotonic() if now is None else now
            return (
                _turn_stream_activity_kind() == stream_burst_kind == "composing"
                and current < stream_burst_until
            )

        def _update_turn_stream_burst(msg: object) -> None:
            nonlocal stream_burst_kind, stream_burst_until, last_burstable_stream_at
            active_kind = _turn_stream_activity_kind()
            if active_kind != "composing":
                stream_burst_kind = None
                stream_burst_until = 0.0
                last_burstable_stream_at = 0.0
                return
            now = time.monotonic()
            kind = self._turn_stream_burst_kind(msg)
            if self._should_start_turn_stream_burst(
                active_kind=active_kind,
                msg_kind=kind,
                current_burst_kind=stream_burst_kind,
                now=now,
                burst_until=stream_burst_until,
                last_burstable_stream_at=last_burstable_stream_at,
            ):
                stream_burst_kind = "composing"
                stream_burst_until = now + _TURN_UI_STREAM_BURST_SECONDS
            if kind == "composing":
                last_burstable_stream_at = now

        def _reset_turn_stream_refresh_budget(*, mark_now: bool = False) -> None:
            nonlocal stream_refresh_parts, stream_refresh_chars, last_stream_refresh_at
            stream_refresh_parts = 0
            stream_refresh_chars = 0
            if mark_now:
                last_stream_refresh_at = time.monotonic()

        def _should_push_turn_stream_refresh(msg: object) -> bool:
            nonlocal stream_refresh_parts, stream_refresh_chars, last_stream_refresh_at
            part_count, char_count, force_refresh = self._turn_stream_refresh_metrics(msg)
            if part_count <= 0 and char_count <= 0 and not force_refresh:
                return False
            stream_refresh_parts += part_count
            stream_refresh_chars += char_count
            if force_refresh:
                _reset_turn_stream_refresh_budget(mark_now=True)
                return True
            now = time.monotonic()
            burst_active = _turn_stream_burst_active(now)
            part_limit, char_limit = self._turn_stream_push_thresholds(
                live_view,
                burst_active=burst_active,
            )
            if stream_refresh_parts >= part_limit:
                _reset_turn_stream_refresh_budget(mark_now=True)
                return True
            if stream_refresh_chars >= char_limit:
                _reset_turn_stream_refresh_budget(mark_now=True)
                return True
            refresh_interval = self._turn_ui_refresh_interval(
                live_view,
                burst_active=burst_active,
            )
            if last_stream_refresh_at <= 0 or now - last_stream_refresh_at >= refresh_interval:
                _reset_turn_stream_refresh_budget(mark_now=True)
                return True
            return False

        def _clear_turn_output_reveal() -> None:
            nonlocal reveal_latest_output
            reveal_latest_output = False

        def _reveal_turn_output_tail() -> None:
            nonlocal reveal_latest_output
            reveal_latest_output = True

        def _toggle_history_view() -> None:
            nonlocal history_view_enabled, history_view_snapshot, history_view_revision
            nonlocal history_view_scroll_offset
            _clear_turn_output_reveal()
            history_view_enabled = not history_view_enabled
            if history_view_enabled:
                history_view_snapshot = live_view.compose_history_body(tail_block_limit=None)
                history_view_scroll_offset = 0
                toast("history view ON", topic="turn_history_view", duration=2.0, immediate=True)
            else:
                history_view_snapshot = None
                toast("history view OFF", topic="turn_history_view", duration=2.0, immediate=True)
            history_view_revision += 1

        def _has_turn_activity() -> bool:
            return (
                not getattr(live_view, "has_pending_input_request", False)
                and self._format_live_activity_status(live_view) is not None
            )

        def _render_activity() -> FormattedText | str:
            return self._render_turn_activity(live_view)

        key_bindings = KeyBindings()
        route_expanded_panel_navigation = Condition(
            lambda: not history_view_enabled
            and getattr(live_view, "is_inline_panel_expanded", False)
            and text_area.buffer.complete_state is None
        )
        route_live_navigation = Condition(
            lambda: not history_view_enabled
            and not reveal_latest_output
            and not getattr(live_view, "is_inline_panel_expanded", False)
            and self._should_route_live_navigation(live_view, text_area.buffer.text)
        )
        route_history_view_navigation = Condition(
            lambda: history_view_enabled
            and not text_area.buffer.text.strip()
            and text_area.buffer.complete_state is None
        )

        def _dispatch_live_key(event: KeyPressEvent, event_type: KeyEvent) -> None:
            _clear_turn_output_reveal()
            live_view.dispatch_keyboard_event(event_type)
            _refresh_turn_view(event.app)

        def _set_expanded_panel_scroll(event: KeyPressEvent, value: int) -> None:
            _clear_turn_output_reveal()
            live_view.set_inline_expanded_scroll_offset(value)
            _refresh_turn_view(event.app)

        route_idle_escape_cancel = Condition(
            lambda: (
                not live_view.has_pending_input_request
                and not text_area.buffer.text.strip()
                and text_area.buffer.complete_state is None
            )
        )
        expand_panel = Condition(
            lambda: live_view.has_pending_input_request
            and live_view.input_mode != "question_other"
            and not getattr(live_view, "is_inline_panel_expanded", False)
        )

        @key_bindings.add("enter", filter=has_completions)
        def _(event: KeyPressEvent) -> None:
            buff = event.current_buffer
            if buff.complete_state and buff.complete_state.completions:
                completion = buff.complete_state.current_completion
                if not completion:
                    completion = buff.complete_state.completions[0]
                buff.apply_completion(completion)

        @key_bindings.add("enter", filter=route_live_navigation & ~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.ENTER)

        @key_bindings.add("up", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.UP)

        @key_bindings.add("down", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.DOWN)

        @key_bindings.add("left", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.LEFT)

        @key_bindings.add("right", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.RIGHT)

        @key_bindings.add("tab", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.TAB)

        @key_bindings.add("space", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.SPACE)

        @key_bindings.add("escape", filter=route_live_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.ESCAPE)

        def _bind_live_digit(key: str, event_type: KeyEvent) -> None:
            @key_bindings.add(key, filter=route_live_navigation, eager=True)
            def _(event: KeyPressEvent) -> None:
                _dispatch_live_key(event, event_type)

        _bind_live_digit("1", KeyEvent.NUM_1)
        _bind_live_digit("2", KeyEvent.NUM_2)
        _bind_live_digit("3", KeyEvent.NUM_3)
        _bind_live_digit("4", KeyEvent.NUM_4)
        _bind_live_digit("5", KeyEvent.NUM_5)
        _bind_live_digit("6", KeyEvent.NUM_6)

        @key_bindings.add("up", filter=route_expanded_panel_navigation, eager=True)
        @key_bindings.add("k", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.UP)

        @key_bindings.add("down", filter=route_expanded_panel_navigation, eager=True)
        @key_bindings.add("j", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _dispatch_live_key(event, KeyEvent.DOWN)

        @key_bindings.add("pageup", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            page_size = max(1, _history_view_visible_lines() - 1)
            _set_expanded_panel_scroll(
                event,
                getattr(live_view, "inline_expanded_scroll_offset", 0) - page_size,
            )

        @key_bindings.add("pagedown", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            page_size = max(1, _history_view_visible_lines() - 1)
            _set_expanded_panel_scroll(
                event,
                getattr(live_view, "inline_expanded_scroll_offset", 0) + page_size,
            )

        @key_bindings.add("home", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _set_expanded_panel_scroll(event, 0)

        @key_bindings.add("end", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _set_expanded_panel_scroll(event, _history_view_total_lines())

        @key_bindings.add("q", filter=route_expanded_panel_navigation, eager=True)
        @key_bindings.add("escape", filter=route_expanded_panel_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            event.current_buffer.document = Document(text="", cursor_position=0)
            _dispatch_live_key(event, KeyEvent.ESCAPE)

        @key_bindings.add("up", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(-1)
            _refresh_turn_view(event.app)

        @key_bindings.add("down", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(1)
            _refresh_turn_view(event.app)

        @key_bindings.add("k", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(-1)
            _refresh_turn_view(event.app)

        @key_bindings.add("j", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(1)
            _refresh_turn_view(event.app)

        @key_bindings.add("pageup", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _page_history_view(-1)
            _refresh_turn_view(event.app)

        @key_bindings.add("pagedown", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _page_history_view(1)
            _refresh_turn_view(event.app)

        @key_bindings.add("home", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _jump_history_view(to_end=False)
            _refresh_turn_view(event.app)

        @key_bindings.add("end", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _jump_history_view(to_end=True)
            _refresh_turn_view(event.app)

        @key_bindings.add("g", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _jump_history_view(to_end=False)
            _refresh_turn_view(event.app)

        @key_bindings.add("G", filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _jump_history_view(to_end=True)
            _refresh_turn_view(event.app)

        @key_bindings.add(Keys.ScrollUp, filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(-1)
            _refresh_turn_view(event.app)

        @key_bindings.add(Keys.ScrollDown, filter=route_history_view_navigation, eager=True)
        def _(event: KeyPressEvent) -> None:
            _scroll_history_view(1)
            _refresh_turn_view(event.app)

        @key_bindings.add("c-e", filter=expand_panel, eager=True)
        def _(event: KeyPressEvent) -> None:
            nonlocal feedback_message
            feedback_message = ""
            if live_view.can_expand_current_panel:
                self._open_live_view_expansion(event, live_view)
            elif reveal_latest_output:
                _clear_turn_output_reveal()
                _refresh_turn_view(event.app)
            else:
                _reveal_turn_output_tail()
                _refresh_turn_view(event.app)

        @key_bindings.add("escape", filter=route_idle_escape_cancel, eager=True)
        def _(event: KeyPressEvent) -> None:
            _clear_turn_output_reveal()
            live_view.dispatch_keyboard_event(KeyEvent.ESCAPE)
            _refresh_turn_view(event.app)

        @key_bindings.add("enter", filter=~route_live_navigation & ~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            nonlocal feedback_message
            _clear_turn_output_reveal()
            command = event.current_buffer.text.strip()
            if not command:
                return
            if live_view.has_pending_input_request and live_view.is_expand_command(command):
                if live_view.can_expand_current_panel:
                    feedback_message = ""
                    event.current_buffer.document = Document(text="", cursor_position=0)
                    self._open_live_view_expansion(event, live_view)
                else:
                    feedback_message = live_view.input_hint
                    _refresh_turn_view(event.app)
                return
            user_input = self._build_user_input(command)
            submit_result = submit_handler(user_input)
            if submit_result.accepted:
                if submit_result.persist_history:
                    self._append_history_entry(command)
                    self._tip_rotation_index += 1
                feedback_message = ""
                event.current_buffer.document = Document(text="", cursor_position=0)
            else:
                feedback_message = submit_result.feedback or live_view.input_hint
            _refresh_turn_view(event.app)

        @key_bindings.add("escape", "enter", eager=True)
        @key_bindings.add("c-j", eager=True)
        def _(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-o", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._open_in_external_editor(event)

        @key_bindings.add("c-l", eager=True)
        def _(event: KeyPressEvent) -> None:
            _reveal_turn_output_tail()
            self._hard_redraw(event.app)

        @key_bindings.add("c-y", eager=True)
        def _(event: KeyPressEvent) -> None:
            _toggle_history_view()
            _refresh_turn_view(event.app)

        if self._clipboard is not None:

            @key_bindings.add("c-v", eager=True)
            def _(event: KeyPressEvent) -> None:
                if self._try_paste_media(event):
                    return
                clipboard_data = event.app.clipboard.get_data()
                self._insert_pasted_text(event.current_buffer, clipboard_data.text)
                event.app.invalidate()

        @key_bindings.add("c-c", eager=True)
        def _(event: KeyPressEvent) -> None:
            nonlocal feedback_message
            _clear_turn_output_reveal()
            if self._dismiss_input(event.current_buffer):
                feedback_message = ""
                _refresh_turn_view(event.app)
                return
            cancel_handler()
            _refresh_turn_view(event.app)

        body_window = Window(
            body_control,
            always_hide_cursor=True,
        )
        activity_window = Window(
            FormattedTextControl(_render_activity),
            height=1,
            dont_extend_height=True,
        )
        hint_window = Window(
            FormattedTextControl(_render_hint),
            height=1,
            dont_extend_height=True,
        )
        toast_window = Window(
            FormattedTextControl(lambda: self._render_toast_line(_app_columns())),
            height=1,
            dont_extend_height=True,
        )
        footer_window = Window(
            FormattedTextControl(_render_footer),
            height=1,
            dont_extend_height=True,
        )
        completion_menu = self._build_inline_completion_menu(text_area.buffer)
        container = self._with_completion_menu(
            HSplit(
                [
                    body_window,
                    ConditionalContainer(
                        activity_window,
                        filter=Condition(_has_turn_activity),
                    ),
                    completion_menu,
                    Frame(
                        HSplit(
                            [
                                ConditionalContainer(
                                    hint_window,
                                    filter=Condition(lambda: bool(_turn_hint_text())),
                                ),
                                text_area,
                            ]
                        ),
                        title=_render_title,
                        style="fg:#38bdf8",
                    ),
                    ConditionalContainer(
                        toast_window,
                        filter=Condition(self._has_toasts),
                    ),
                    footer_window,
                ]
            )
        )
        self._hide_original_completion_menu(container, buffer=text_area.buffer)

        app = Application[None](
            layout=Layout(container, focused_element=text_area),
            key_bindings=key_bindings,
            clipboard=self._clipboard,
            cursor=STEADY_INPUT_CURSOR,
            style=Style.from_dict(_shell_style_dict()),
            full_screen=False,
            erase_when_done=True,
            refresh_interval=None,
            mouse_support=Condition(lambda: history_view_enabled),
            terminal_size_polling_interval=_TERMINAL_SIZE_POLLING_INTERVAL,
        )
        last_layout_signature = _turn_layout_signature()

        def _follow_turn_output(_: object) -> None:
            nonlocal last_layout_signature
            signature = _turn_layout_signature()
            if signature != last_layout_signature:
                last_layout_signature = signature
                app.invalidate()

        app.after_render.add_handler(_follow_turn_output)

        async def _consume_wire() -> None:
            nonlocal feedback_message
            while True:
                try:
                    msg = await wire.receive()
                except QueueShutDown:
                    live_view.cleanup(is_interrupt=False)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    app.exit()
                    return

                if isinstance(msg, StepInterrupted):
                    live_view.cleanup(is_interrupt=True)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    app.exit()
                    return

                _clear_turn_output_reveal()
                live_view.dispatch_wire_message(msg)
                _update_turn_stream_burst(msg)
                feedback_message = ""
                significant = is_significant_for_render(msg)
                stream_refresh = False
                if not significant and live_view.needs_periodic_refresh:
                    stream_refresh = _should_push_turn_stream_refresh(msg)
                if significant or stream_refresh or not live_view.needs_periodic_refresh:
                    _refresh_turn_view(app)
                    _reset_turn_stream_refresh_budget(mark_now=True)

                # Yield to the event loop after significant state changes so
                # prompt_toolkit can repaint immediately.  Without this the
                # loop drains every queued message before _redraw runs,
                # delaying tool-result and reminder rendering.
                if significant or stream_refresh:
                    await asyncio.sleep(0)

        async def _animate() -> None:
            while True:
                await asyncio.sleep(
                    self._turn_ui_refresh_interval(
                        live_view,
                        burst_active=_turn_stream_burst_active(),
                    )
                )
                if self._refresh_turn_application(
                    app,
                    live_view=live_view,
                ):
                    _reset_turn_stream_refresh_budget(mark_now=True)

        consume_task = asyncio.create_task(_consume_wire())
        animate_task = asyncio.create_task(_animate())

        # --- Flicker-free transition -------------------------------------------
        # Write the deferred erase and the pre-rendered user echo into the
        # prompt_toolkit *output buffer* (not sys.stdout).  Then suppress
        # every ``output.flush()`` call until the first render is done.
        # This way the terminal receives the erase, the echo, **and** the
        # first full frame of the turn UI in a single write — completely
        # eliminating the visible flash where the PROMPT frame would
        # briefly disappear between the idle prompt and the turn UI.
        output = app.output

        if self._deferred_erase_pending:
            if self._deferred_erase_x > 0:
                output.cursor_backward(self._deferred_erase_x)
            if self._deferred_erase_y > 0:
                output.cursor_up(self._deferred_erase_y)
            output.erase_down()
            self._deferred_erase_pending = False

        if pre_rendered_echo:
            output.write_raw(pre_rendered_echo)

        _original_flush = output.flush
        _first_render_flushed = False

        def _suppressed_flush() -> None:
            pass

        def _flush_after_first_render(_app: object) -> None:
            nonlocal _first_render_flushed
            if _first_render_flushed:
                return
            _first_render_flushed = True
            output.flush = _original_flush
            output.flush()
            app.after_render -= _flush_after_first_render

        output.flush = _suppressed_flush  # type: ignore[method-assign]
        app.after_render += _flush_after_first_render

        try:
            with patch_stdout(raw=True):
                await app.run_async()
        finally:
            # Always restore the original flush in case after_render
            # never fired (e.g. the app errored out before rendering).
            if not _first_render_flushed:
                output.flush = _original_flush  # type: ignore[method-assign]
                output.flush()
            consume_task.cancel()
            animate_task.cancel()
            with suppress(asyncio.CancelledError):
                await consume_task
            with suppress(asyncio.CancelledError):
                await animate_task

        final_renderable = live_view.compose(
            include_status=False,
            include_running_indicators=False,
        )
        console.print(final_renderable, end="")

    def _append_history_entry(self, text: str) -> None:
        safe_history_text = self._get_placeholder_manager().serialize_for_history(text).strip()
        entry = _HistoryEntry(content=safe_history_text)
        if not entry.content:
            return

        # skip if same as last entry
        if entry.content == self._last_history_content:
            return

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json(ensure_ascii=False) + "\n")
            self._history.append_string(entry.content)
            self._last_history_content = entry.content
        except OSError as exc:
            logger.warning(
                "Failed to append user history entry: {file} ({error})",
                file=self._history_file,
                error=exc,
            )

    def _mode_text(self, status: StatusSnapshot) -> str:
        mode = str(self._mode).lower()
        if self._mode == PromptMode.AGENT:
            mode_details: list[str] = []
            if self._model_name:
                mode_details.append(self._model_name)
            if self._thinking:
                mode_details.append("thinking")
            if mode_details:
                mode += f" ({', '.join(mode_details)})"
        flags: list[str] = []
        if status.yolo_enabled:
            flags.append("yolo")
        if status.plan_mode:
            flags.append("plan")
        if flags:
            mode += f" [{' / '.join(flags)}]"
        return mode

    def _rotated_tips_text(self, available: int) -> str | None:
        if available <= 0 or not self._tips:
            return None
        full_text = _TIP_SEPARATOR.join(self._tips)
        if self._display_width(full_text) <= available:
            return full_text

        n = len(self._tips)
        offset = self._tip_rotation_index % n
        rotated = self._tips[offset:] + self._tips[:offset]
        selected: list[str] = []
        total_len = 0
        for tip in rotated:
            needed = self._display_width(tip) + (
                self._display_width(_TIP_SEPARATOR) if selected else 0
            )
            if total_len + needed <= available:
                selected.append(tip)
                total_len += needed
        return _TIP_SEPARATOR.join(selected) if selected else None

    def _render_active_bottom_toolbar(
        self,
        columns: int,
        state: InputBoxState,
        status: StatusSnapshot,
        right_text: str,
    ) -> FormattedText:
        _, border_style, _ = self._input_box_appearance(state)
        footer_text = self._render_footer_left_text(
            status=status,
            columns=columns,
            right_text=right_text,
            prefix="╰─ ",
        )
        padding = max(
            1,
            columns - self._display_width(footer_text) - self._display_width(right_text),
        )
        return FormattedText(
            [
                (border_style, footer_text),
                ("", " " * padding),
                ("fg:#9ca3af", right_text),
            ]
        )

    def _render_bottom_toolbar(self) -> FormattedText:
        app = get_app_or_none()
        assert app is not None
        columns = app.output.get_size().columns
        status = self._status_provider()
        right_text = self._render_right_span(status)
        input_box_state = self._input_box_state_provider()
        if input_box_state.active:
            return self._render_active_bottom_toolbar(columns, input_box_state, status, right_text)

        fragments: list[tuple[str, str]] = []
        fragments.append(("fg:#4d4d4d", "─" * columns))
        fragments.append(("", "\n"))

        left_text = self._render_footer_left_text(
            status=status,
            columns=max(1, columns - self._display_width("  ")),
            right_text=right_text,
        )
        fragments.extend([("", left_text), ("", "  ")])
        columns -= self._display_width(left_text) + self._display_width("  ")

        current_toast_left = _current_toast("left")
        if current_toast_left is not None:
            left_text = current_toast_left.message
        else:
            left_text = self._rotated_tips_text(
                columns - self._display_width(right_text) - self._display_width("   ")
            )

        if left_text:
            left_text = self._truncate_display_text(
                left_text,
                max(0, columns - self._display_width(right_text) - self._display_width("   ")),
            )
            fragments.extend([("", left_text), ("", "  ")])
            columns -= self._display_width(left_text) + self._display_width("  ")

        padding = max(1, columns - self._display_width(right_text))
        fragments.append(("", " " * padding))
        fragments.append(("fg:#9ca3af", right_text))
        return FormattedText(fragments)

    @staticmethod
    def _render_right_span(status: StatusSnapshot) -> str:
        current_toast = _current_toast("right")
        if current_toast is not None:
            return current_toast.message
        return format_context_status(
            status.context_usage,
            status.context_tokens,
            status.max_context_tokens,
        )
