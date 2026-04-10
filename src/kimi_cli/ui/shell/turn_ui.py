from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none as _ptk_get_app_or_none
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout as _patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from kimi_cli.eventbus.types import StepInterrupted, TextPart, ThinkPart, ToolCallPart
from kimi_cli.ui.shell.console import console as _console
from kimi_cli.ui.shell.keyboard import KeyEvent
from kimi_cli.ui.shell.rich_ptk import (
    RichRenderableControl as _RichRenderableControl,
)
from kimi_cli.ui.shell.rich_ptk import (
    StackedRichRenderableControl as _StackedRichRenderableControl,
)
from kimi_cli.ui.shell.toast import toast as _toast
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

from .prompt_constants import (
    _INDICATOR_STYLES,
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
    _TURN_UI_STREAM_BURST_SECONDS,
    _TURN_UI_STREAM_PUSH_CHARS,
    _TURN_UI_STREAM_PUSH_PARTS,
    _TURN_UI_STREAM_RESUME_BURST_GAP,
    STEADY_INPUT_CURSOR,
)
from .prompt_types import PromptMode, TurnSubmitResult, UserInput
from .toolbar import _shell_style_dict


def _prompt_module_attr(name: str, default: object) -> object:
    prompt_module = sys.modules.get("kimi_cli.ui.shell.prompt")
    if prompt_module is None:
        return default
    return getattr(prompt_module, name, default)


def _get_app_or_none() -> Any:
    return cast(Callable[[], Any], _prompt_module_attr("get_app_or_none", _ptk_get_app_or_none))()


class PromptTurnUIMixin:
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
        return f"History {position} - ↑/↓, PgUp/PgDn, Home/End, Ctrl-Y live"

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
            current_app = _get_app_or_none() if app is None else app
            return current_app.output.get_size().columns if current_app is not None else 80

        def _app_rows(app: Application[Any] | None = None) -> int:
            current_app = _get_app_or_none() if app is None else app
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
            app = _get_app_or_none()
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
                cast(Callable[..., None], _prompt_module_attr("toast", _toast))(
                    "history view ON",
                    topic="turn_history_view",
                    duration=2.0,
                    immediate=True,
                )
            else:
                history_view_snapshot = None
                cast(Callable[..., None], _prompt_module_attr("toast", _toast))(
                    "history view OFF",
                    topic="turn_history_view",
                    duration=2.0,
                    immediate=True,
                )
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
                event.current_buffer.reset()
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

        def _on_turn_resize_settled() -> None:
            if app.is_running:
                self._hard_redraw(app)

        self._install_resize_handler(app, _on_turn_resize_settled)

        last_layout_signature = _turn_layout_signature()

        def _follow_turn_output(_: object) -> None:
            nonlocal last_layout_signature
            signature = _turn_layout_signature()
            if signature != last_layout_signature:
                last_layout_signature = signature
                app.invalidate()

        app.after_render.add_handler(_follow_turn_output)

        def _exit_app_if_pending() -> None:
            """Exit the prompt_toolkit Application only if its future is still pending."""
            if app.future is not None and not app.future.done():
                app.exit()

        async def _consume_wire() -> None:
            nonlocal feedback_message
            while True:
                try:
                    msg = await wire.receive()
                except QueueShutDown:
                    live_view.cleanup(is_interrupt=False)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    _exit_app_if_pending()
                    return

                if isinstance(msg, StepInterrupted):
                    live_view.cleanup(is_interrupt=True)
                    live_view.finish_turn()
                    _refresh_turn_view(app)
                    _exit_app_if_pending()
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
            with cast(Callable[..., Any], _prompt_module_attr("patch_stdout", _patch_stdout))(
                raw=True
            ):
                await app.run_async()
        finally:
            self._uninstall_resize_handler(app)
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
        cast(Any, _prompt_module_attr("console", _console)).print(final_renderable, end="")
