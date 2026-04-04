from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from hashlib import md5
from typing import Any

from kaos.path import KaosPath
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.completion import merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, Filter, has_completions, has_focus, is_done
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
from rich.style import Style as _RichStyle

from kimi_cli.llm import ModelCapability
from kimi_cli.loop import StatusSnapshot
from kimi_cli.share import get_share_dir
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
from kimi_cli.ui.shell.placeholders import sanitize_surrogates
from kimi_cli.ui.shell.prompt_constants import (
    _FOCUS_IN_SEQ,
    _REFRESH_INTERVAL,
    _TERMINAL_SIZE_POLLING_INTERVAL,
    _TURN_HISTORY_VIEW_EMPTY_POSITION,
    _TURN_UI_BURST_REFRESH_INTERVAL,
    _TURN_UI_BURST_STREAM_PUSH_CHARS,
    _TURN_UI_BURST_STREAM_PUSH_PARTS,
    _TURN_UI_COMPOSE_STREAM_PUSH_CHARS,
    _TURN_UI_COMPOSE_STREAM_PUSH_PARTS,
    _TURN_UI_FAST_REFRESH_INTERVAL,
    _TURN_UI_FAST_STREAM_PUSH_CHARS,
    _TURN_UI_FAST_STREAM_PUSH_PARTS,
    _TURN_UI_REFRESH_INTERVAL,
    _TURN_UI_STREAM_PUSH_CHARS,
    _TURN_UI_STREAM_PUSH_PARTS,
    _TURN_UI_STREAM_RESUME_BURST_GAP,
    PROMPT_SYMBOL,
    PROMPT_SYMBOL_SHELL,
    PROMPT_SYMBOL_THINKING,
    STEADY_INPUT_CURSOR,
)
from kimi_cli.ui.shell.prompt_input import PromptInputMixin
from kimi_cli.ui.shell.prompt_types import (
    InputBoxState,
    PromptMode,
    TurnSubmitResult,
    UserInput,
    _load_history_entries,
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
from kimi_cli.ui.shell.terminal_compat import PromptTerminalCompatMixin
from kimi_cli.ui.shell.toast import current_toast as _current_toast
from kimi_cli.ui.shell.toast import toast, toast_queues
from kimi_cli.ui.shell.toolbar import PromptToolbarMixin, _build_toolbar_tips, _shell_style_dict
from kimi_cli.ui.shell.turn_ui import PromptTurnUIMixin
from kimi_cli.ui.shell.visualize import (
    MAX_ACTIVE_TURN_CONTENT_CHARS,
    MAX_ACTIVE_TURN_FLUSHED_BLOCKS,
    MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS,
)
from kimi_cli.utils.clipboard import grab_media_from_clipboard, is_clipboard_available
from kimi_cli.utils.slashcmd import SlashCommand

AttachmentCache = prompt_placeholders.AttachmentCache
CachedAttachment = prompt_placeholders.CachedAttachment
_parse_attachment_kind = prompt_placeholders.parse_attachment_kind
_sanitize_surrogates = sanitize_surrogates  # backward compat re-export
_rich_from_ansi = rich_from_ansi
_rich_style_to_prompt_toolkit = rich_style_to_prompt_toolkit
_toast_queues = toast_queues
RichStyle = _RichStyle

__all__ = [
    "Application",
    "AttachmentCache",
    "CachedAttachment",
    "CustomPromptSession",
    "Document",
    "InputBoxState",
    "InMemoryHistory",
    "LocalFileMentionCompleter",
    "MAX_ACTIVE_TURN_CONTENT_CHARS",
    "MAX_ACTIVE_TURN_FLUSHED_BLOCKS",
    "MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS",
    "PROMPT_SYMBOL",
    "PROMPT_SYMBOL_SHELL",
    "PROMPT_SYMBOL_THINKING",
    "PromptMode",
    "RichStyle",
    "TurnSubmitResult",
    "UserInput",
    "_ClearScreenRequest",
    "_FOCUS_IN_SEQ",
    "_REFRESH_INTERVAL",
    "_RichRenderableControl",
    "_StackedRichRenderableControl",
    "_TERMINAL_SIZE_POLLING_INTERVAL",
    "_TURN_HISTORY_VIEW_EMPTY_POSITION",
    "_TURN_UI_BURST_REFRESH_INTERVAL",
    "_TURN_UI_BURST_STREAM_PUSH_CHARS",
    "_TURN_UI_BURST_STREAM_PUSH_PARTS",
    "_TURN_UI_COMPOSE_STREAM_PUSH_CHARS",
    "_TURN_UI_COMPOSE_STREAM_PUSH_PARTS",
    "_TURN_UI_FAST_REFRESH_INTERVAL",
    "_TURN_UI_FAST_STREAM_PUSH_CHARS",
    "_TURN_UI_FAST_STREAM_PUSH_PARTS",
    "_TURN_UI_REFRESH_INTERVAL",
    "_TURN_UI_STREAM_PUSH_CHARS",
    "_TURN_UI_STREAM_PUSH_PARTS",
    "_TURN_UI_STREAM_RESUME_BURST_GAP",
    "_current_toast",
    "_parse_attachment_kind",
    "_rich_from_ansi",
    "_rich_style_to_prompt_toolkit",
    "_sanitize_surrogates",
    "_shell_style_dict",
    "_toast_queues",
    "console",
    "grab_media_from_clipboard",
    "toast",
]


class _ClearScreenRequest(Exception):
    """Raised when Ctrl-L is pressed to request full screen clear + history replay."""

    def __init__(self, buffer_text: str = ""):
        self.buffer_text = buffer_text
        super().__init__()

class _NotificationAutoTrigger(Exception):
    """Raised to interrupt the idle prompt when background task notifications arrive."""

    pass



class CustomPromptSession(
    PromptTurnUIMixin,
    PromptInputMixin,
    PromptTerminalCompatMixin,
    PromptToolbarMixin,
):
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
        self._input_box_state_provider = input_box_state_provider or (lambda: InputBoxState())
        self._working_dir_provider = working_dir_provider or (lambda: str(KaosPath.cwd()))
        self._redraw_callback = redraw_callback
        self._model_capabilities = model_capabilities
        self._model_name = model_name
        self._last_history_content: str | None = None
        self._mode: PromptMode = PromptMode.AGENT
        self._thinking = thinking
        self._placeholder_manager = prompt_placeholders.PromptPlaceholderManager()
        self._attachment_cache = self._placeholder_manager.attachment_cache
        self._tip_rotation_index = 0
        clipboard_available = is_clipboard_available()
        self._tips = _build_toolbar_tips(clipboard_available)

        history_entries = _load_history_entries(self._history_file)
        history = InMemoryHistory()
        for entry in history_entries:
            history.append_string(entry.content)
        self._history = history

        if history_entries:
            self._last_history_content = history_entries[-1].content

        self._file_mention_completer = LocalFileMentionCompleter(
            KaosPath.cwd().unsafe_to_local_path()
        )
        self._agent_slash_completer = SlashCommandCompleter(agent_mode_slash_commands)
        self._shell_slash_completer = SlashCommandCompleter(shell_mode_slash_commands)
        self._agent_mode_completer = merge_completers(
            [
                self._agent_slash_completer,
                self._file_mention_completer,
            ],
            deduplicate=True,
        )
        turn_slash_commands = [
            c
            for c in (*shell_mode_slash_commands, *agent_mode_slash_commands)
            if c.name in _turn_allowed_commands or c.name.startswith(_skill_prefix)
        ]
        self._turn_mode_completer = merge_completers(
            [SlashCommandCompleter(turn_slash_commands), self._file_mention_completer],
            deduplicate=True,
        )
        self._shell_mode_completer = self._shell_slash_completer

        key_bindings = KeyBindings()

        @key_bindings.add("c-x", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._mode = self._mode.toggle()
            toast(
                f"{self._mode.value} mode",
                topic="prompt_mode",
                duration=2.0,
                immediate=True,
            )
            self._apply_mode(event)
            event.app.invalidate()

        @key_bindings.add("escape", "enter", eager=True)
        @key_bindings.add("c-j", eager=True)
        def _(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @key_bindings.add("c-o", eager=True)
        def _(event: KeyPressEvent) -> None:
            self._open_in_external_editor(event)

        @key_bindings.add("c-l", eager=True)
        def _(event: KeyPressEvent) -> None:
            if self._redraw_callback is not None:
                event.app.exit(exception=_ClearScreenRequest(event.current_buffer.text))
            else:
                self._hard_redraw(event.app)

        @key_bindings.add(Keys.BracketedPaste, eager=True)
        def _(event: KeyPressEvent) -> None:
            self._handle_bracketed_paste(event)

        if clipboard_available:

            @key_bindings.add("c-v", eager=True)
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
        self._base_key_bindings = key_bindings

        self._session = PromptSession[str](
            message=self._render_message,
            prompt_continuation=self._render_prompt_continuation,
            placeholder=self._render_placeholder,
            rprompt=self._render_rprompt,
            cursor=STEADY_INPUT_CURSOR,
            completer=self._agent_mode_completer,
            complete_while_typing=True,
            reserve_space_for_menu=10,
            key_bindings=key_bindings,
            clipboard=clipboard,
            history=history,
            bottom_toolbar=self._render_bottom_toolbar,
            style=Style.from_dict(_shell_style_dict()),
        )
        self._session.app.terminal_size_polling_interval = _TERMINAL_SIZE_POLLING_INTERVAL
        self._install_slash_completion_menu()

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
        symbol = PROMPT_SYMBOL_THINKING if self._thinking else PROMPT_SYMBOL
        return max(1, get_cwidth(f"{symbol} ") - 2)

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

        accept_key_bindings = KeyBindings()

        @accept_key_bindings.add("enter", filter=has_completions, eager=True)
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

        @accept_key_bindings.add("enter", filter=~has_completions, eager=True)
        def _(event: KeyPressEvent) -> None:
            event.app.exit(result=text_area.buffer.text)

        @accept_key_bindings.add("c-d", eager=True)
        def _(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            if not buffer.text:
                event.app.exit(exception=EOFError)
                return
            buffer.delete()

        @accept_key_bindings.add("c-c", eager=True)
        def _(event: KeyPressEvent) -> None:
            if self._dismiss_input(event.current_buffer):
                event.app.invalidate()
                return
            event.app.exit(exception=KeyboardInterrupt)

        app = Application[str](
            layout=Layout(container, focused_element=text_area),
            key_bindings=merge_key_bindings([self._base_key_bindings, accept_key_bindings]),
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

    def _get_prompt_application(self) -> tuple[Application[str], TextArea]:
        if self._prompt_app is None or self._prompt_text_area is None:
            self._prompt_app, self._prompt_text_area = self._build_prompt_application()
        return self._prompt_app, self._prompt_text_area

    def _prepare_prompt_application(self) -> tuple[Application[str], TextArea]:
        app, text_area = self._get_prompt_application()
        self._apply_mode_to_buffer(text_area.buffer)
        text_area.buffer.reset()
        self._hard_redraw(app)
        return app, text_area

    def trigger_notification_auto_turn(self) -> None:
        """Interrupt the idle prompt to auto-trigger a turn for pending notifications.

        Skips interruption if the user has typed content into the buffer
        (they are not truly idle).  Also swallows any ``InvalidStateError``
        from prompt_toolkit if the application has already resolved (e.g.
        the user pressed Enter at the same instant).
        """
        app = self._prompt_app
        if app is None or not app.is_running:
            return
        # Don't interrupt if the user is actively composing input.
        if self._prompt_text_area is not None and self._prompt_text_area.buffer.text.strip():
            return
        with suppress(Exception):
            app.exit(exception=_NotificationAutoTrigger())

    async def prompt(self) -> UserInput | None:
        """Prompt the user for input.

        Returns ``None`` when the prompt is interrupted by a notification
        auto-trigger (background task completed while idle).
        """
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
            except _NotificationAutoTrigger:
                return None
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
