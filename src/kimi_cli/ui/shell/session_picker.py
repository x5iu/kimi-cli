from __future__ import annotations

from pathlib import Path
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Frame, RadioList

from kaos.path import KaosPath
from kimi_cli.session import Session
from kimi_cli.utils.datetime import format_relative_time

SessionScope = Literal["current", "all"]

_EMPTY_SESSION_ID = "__empty__"


def _shorten_work_dir(work_dir: str, max_len: int = 30) -> str:
    home = str(Path.home())
    if work_dir.startswith(home):
        work_dir = "~" + work_dir[len(home) :]
    if len(work_dir) <= max_len:
        return work_dir
    return "..." + work_dir[-(max_len - 3) :]


class SessionPickerApp:
    def __init__(
        self,
        *,
        work_dir: KaosPath,
        current_session: Session,
    ) -> None:
        self._work_dir = work_dir
        self._current_session = current_session
        self._scope: SessionScope = "current"
        self._sessions: list[Session] = []
        self._reload_version: int = 0

        self._radio_list = RadioList[str](
            values=[(_EMPTY_SESSION_ID, "Loading...")],
            default=_EMPTY_SESSION_ID,
            show_numbers=False,
            select_on_focus=True,
            open_character="",
            select_character="\u276f",
            close_character="",
            show_cursor=False,
            show_scrollbar=False,
            container_style="class:task-list",
            checked_style="class:task-list.checked",
        )
        self._app = self._build_app()

    async def run(self) -> tuple[str, KaosPath] | None:
        await self._load_sessions()
        self._sync_radio_list()
        result = await self._app.run_async()
        if result is None:
            return None
        if result == _EMPTY_SESSION_ID:
            return None
        for s in self._sessions:
            if s.id == result:
                return (result, s.work_dir)
        return None

    async def _load_sessions(self) -> None:
        current = self._current_session

        if self._scope == "current":
            sessions = [s for s in await Session.list(self._work_dir) if s.id != current.id]
        else:
            sessions = [s for s in await Session.list_all() if s.id != current.id]

        await current.refresh()
        if not current.is_empty():
            sessions.insert(0, current)
        self._sessions = sessions

    def _build_values(self) -> list[tuple[str, str]]:
        if not self._sessions:
            return [(_EMPTY_SESSION_ID, "No sessions found.")]

        current_id = self._current_session.id
        values: list[tuple[str, str]] = []
        for session in self._sessions:
            time_str = format_relative_time(session.updated_at)
            short_id = session.id[:8]
            marker = " (current)" if session.id == current_id else ""

            title_line = f"{session.title}{marker}"
            meta_parts = [time_str, short_id]
            if self._scope == "all":
                meta_parts.append(_shorten_work_dir(str(session.work_dir)))
            meta_line = "  " + " \u00b7 ".join(meta_parts)
            label = f"{title_line}\n{meta_line}"

            values.append((session.id, label))
        return values

    def _sync_radio_list(self) -> None:
        values = self._build_values()
        self._radio_list.values = values
        default = values[0][0]
        self._radio_list.current_value = default
        self._radio_list.current_values = [default]
        for idx, (val, _) in enumerate(values):
            if val == default:
                self._radio_list._selected_index = idx  # pyright: ignore[reportPrivateUsage]
                break

    def _header_fragments(self) -> StyleAndTextTuples:
        scope_label = "current directory" if self._scope == "current" else "all directories"
        total = len(self._sessions)
        selected = self._radio_list._selected_index + 1  # pyright: ignore[reportPrivateUsage]
        return [
            ("class:header.title", f" SESSIONS ({selected} of {total}) "),
            ("class:header.meta", f" [{scope_label}] "),
        ]

    def _footer_fragments(self) -> StyleAndTextTuples:
        scope_action = "show all projects" if self._scope == "current" else "show current project"
        return [
            ("class:footer.text", f" Ctrl+A to {scope_action}"),
            ("class:footer.text", " \u00b7 "),
            ("class:footer.text", "Enter to select"),
            ("class:footer.text", " \u00b7 "),
            ("class:footer.text", "Esc to cancel "),
        ]

    def _build_app(self) -> Application[str | None]:
        kb = KeyBindings()

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event: KeyPressEvent) -> None:
            event.app.exit(result=None)

        @kb.add("enter", eager=True)
        def _select(event: KeyPressEvent) -> None:
            value = self._radio_list.current_value
            event.app.exit(result=value)

        @kb.add("c-a")
        def _toggle_scope(event: KeyPressEvent) -> None:
            self._scope = "all" if self._scope == "current" else "current"
            self._reload_version += 1
            event.app.create_background_task(self._reload_and_refresh(event.app))

        _ = (_cancel, _select, _toggle_scope)

        header = Window(
            FormattedTextControl(self._header_fragments),
            height=1,
            style="class:header",
        )
        body = Frame(
            Box(self._radio_list, padding=1),
            title=lambda: " Sessions ",
        )
        footer = Window(
            FormattedTextControl(self._footer_fragments),
            height=1,
            style="class:footer",
        )

        return Application(
            layout=Layout(
                HSplit([header, body, footer]),
                focused_element=self._radio_list,
            ),
            key_bindings=kb,
            full_screen=True,
            erase_when_done=True,
            style=_session_picker_style(),
        )

    async def _reload_and_refresh(self, app: Application[str | None]) -> None:
        version = self._reload_version

        self._radio_list.values = [(_EMPTY_SESSION_ID, "Loading...")]
        self._radio_list.current_value = _EMPTY_SESSION_ID
        self._radio_list._selected_index = 0  # pyright: ignore[reportPrivateUsage]
        app.invalidate()

        await self._load_sessions()

        if version != self._reload_version:
            return

        self._sync_radio_list()
        app.invalidate()


def _session_picker_style() -> Style:
    return Style.from_dict(
        {
            "header": "bg:#1f2937 #e5e7eb",
            "header.title": "bg:#1f2937 #67e8f9 bold",
            "header.meta": "bg:#1f2937 #9ca3af",
            "task-list": "bg:#111827 #d1d5db",
            "task-list.checked": "bg:#164e63 #ecfeff bold",
            "frame.border": "#155e75",
            "frame.label": "bg:#0f172a #67e8f9 bold",
            "footer": "bg:#0f172a #cbd5e1",
            "footer.text": "bg:#0f172a #cbd5e1",
        }
    )


async def pick_session(
    *,
    work_dir: KaosPath,
    current_session: Session,
) -> tuple[str, KaosPath] | None:
    return await SessionPickerApp(work_dir=work_dir, current_session=current_session).run()
