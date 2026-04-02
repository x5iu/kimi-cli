from __future__ import annotations

import sys
from collections import deque
from typing import Any, Literal, cast

from kaos.path import KaosPath
from prompt_toolkit.application.current import get_app_or_none as _ptk_get_app_or_none
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.utils import get_cwidth

from kimi_cli.soul import StatusSnapshot, format_context_status

from .prompt_constants import PROMPT_SYMBOL, PROMPT_SYMBOL_SHELL, PROMPT_SYMBOL_THINKING
from .prompt_types import InputBoxState, PromptMode
from .toast import current_toast as _current_toast


def _get_app_or_none() -> Any:
    prompt_module = sys.modules.get("kimi_cli.ui.shell.prompt")
    if prompt_module is None:
        return _ptk_get_app_or_none()
    return getattr(prompt_module, "get_app_or_none", _ptk_get_app_or_none)()


def _build_toolbar_tips(clipboard_available: bool) -> list[str]:
    tips = [
        "ctrl-x: toggle mode",
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


class PromptToolbarMixin:
    def _render_message(self) -> FormattedText:
        if self._mode == PromptMode.SHELL:
            return FormattedText([("bold", f"{PROMPT_SYMBOL_SHELL} ")])

        input_box_state = self._input_box_state_provider()
        if input_box_state.active:
            return self._render_active_input_box_message(input_box_state)

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
        app = _get_app_or_none()
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
        app = _get_app_or_none()
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

        app = _get_app_or_none()
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
    def _shorten_footer_path(path: str, width: int) -> str:
        if width <= 0:
            return ""
        if width <= 4:
            return PromptToolbarMixin._truncate_display_text(path, width)
        return PromptToolbarMixin._shorten_middle_display_text(path, width)

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
        app = _get_app_or_none()
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
