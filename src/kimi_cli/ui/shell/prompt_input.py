from __future__ import annotations

import shlex
import sys
from collections.abc import Callable
from typing import cast

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyPressEvent

from kimi_cli.ui.shell.console import console
from kimi_cli.utils.clipboard import grab_media_from_clipboard as _grab_media_from_clipboard
from kimi_cli.utils.logging import logger

from .placeholders import PromptPlaceholderManager, normalize_pasted_text
from .prompt_types import PromptMode, UserInput, _HistoryEntry, _trim_in_memory_history
from .toast import toast as _toast


def _prompt_module_attr(name: str, default: object) -> object:
    prompt_module = sys.modules.get("kimi_cli.ui.shell.prompt")
    if prompt_module is None:
        return default
    return getattr(prompt_module, name, default)


class PromptInputMixin:
    def _open_in_external_editor(self, event: KeyPressEvent) -> None:
        from prompt_toolkit.application.run_in_terminal import run_in_terminal

        from kimi_cli.utils.editor import edit_text_in_editor, get_editor_command

        configured = self._editor_command_provider()

        if get_editor_command(configured) is None:
            cast(Callable[..., None], _prompt_module_attr("toast", _toast))(
                "No editor found. Set $VISUAL/$EDITOR or default_editor in config.toml."
            )
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

    def _open_live_view_expansion(self, event: KeyPressEvent, live_view: object) -> None:
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
        try:
            buff = event.current_buffer if event is not None else None
        except Exception:
            buff = None
        if buff is None and self._prompt_text_area is not None:
            buff = self._prompt_text_area.buffer
        if buff is None:
            buff = self._session.default_buffer
        self._apply_mode_to_buffer(buff)

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
        result = cast(
            Callable[[], object | None],
            _prompt_module_attr(
                "grab_media_from_clipboard",
                _grab_media_from_clipboard,
            ),
        )()
        if result is None:
            return False

        parts: list[str] = []

        if result.file_paths:
            logger.debug("Pasted {count} file path(s) from clipboard", count=len(result.file_paths))
            for p in result.file_paths:
                text = str(p)
                if self._mode == PromptMode.SHELL:
                    text = shlex.quote(text)
                parts.append(text)

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

    def _append_history_entry(self, text: str) -> None:
        safe_history_text = self._get_placeholder_manager().serialize_for_history(text).strip()
        entry = _HistoryEntry(content=safe_history_text)
        if not entry.content:
            return

        if entry.content == self._last_history_content:
            return

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with self._history_file.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json(ensure_ascii=False) + "\n")
            self._history.append_string(entry.content)
            _trim_in_memory_history(self._history)
            self._last_history_content = entry.content
        except OSError as exc:
            logger.warning(
                "Failed to append user history entry: {file} ({error})",
                file=self._history_file,
                error=exc,
            )
