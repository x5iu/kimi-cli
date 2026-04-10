from __future__ import annotations

import asyncio
import time
from typing import Any

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.keys import Keys

from kimi_cli.utils.logging import logger

from .prompt_constants import (
    _FOCUS_IN_SEQ,
    _INDICATOR_FRAMES,
    _REFRESH_INTERVAL,
    _RESIZE_DEBOUNCE_SECONDS,
)
from .toast import current_toast as _current_toast


class PromptTerminalCompatMixin:
    def _install_deferred_erase(self, app: Any) -> None:
        """Monkey-patch the prompt app renderer to defer erase on exit.

        Keeps the input box on-screen until `execute_deferred_erase` is
        called (right before the turn UI starts), eliminating the flash.

        IMPORTANT - resize correctness
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        prompt_toolkit's ``Application._on_resize()`` calls
        ``renderer.erase()`` to wipe the old content before redrawing at
        the new terminal dimensions. If we unconditionally defer *every*
        erase call, the old content is never cleared and the subsequent
        redraw lands at a different cursor position - producing the
        "ghost frames" / screen corruption visible on window resize.

        The fix: only defer the erase when the app is **shutting down**
        (``app._is_running is False``). While the app is still running
        (resize via ``_on_resize``, or manual ``Ctrl-L`` hard-redraw),
        the original erase executes immediately so prompt_toolkit's
        differential renderer starts from a clean slate.

        Do NOT remove the ``app._is_running`` guard without verifying
        that terminal resize no longer causes visual artefacts.
        """
        renderer = app.renderer
        original_erase = renderer.erase

        def _deferred_erase(leave_alternate_screen: bool = True) -> None:
            if app._is_running:
                original_erase(leave_alternate_screen=leave_alternate_screen)
                return
            x, y = renderer._cursor_pos.x, renderer._cursor_pos.y
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
            left_toast.message if left_toast is not None else None,
            right_toast.message if right_toast is not None else None,
        )

    @staticmethod
    def _live_indicator_frame(kind: str, *, now: float | None = None) -> str:
        frames = _INDICATOR_FRAMES.get(kind, ("…",))
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
    def _install_resize_handler(
        app: Any,
        on_resize_settled: Any,
        debounce: float = _RESIZE_DEBOUNCE_SECONDS,
    ) -> None:
        """Monkey-patch ``app._on_resize`` to fire *on_resize_settled* after a debounce.

        The original ``_on_resize`` still runs immediately (erase + redraw)
        so prompt_toolkit's differential renderer stays in sync.  The
        callback fires only after no further resize events arrive within
        *debounce* seconds.

        The handle is stored on the app so ``_uninstall_resize_handler``
        can cancel it and restore the original method.
        """
        original = app._on_resize
        app._kimi_resize_original = original
        app._kimi_resize_handle: asyncio.TimerHandle | None = None

        def _patched_on_resize() -> None:
            original()
            handle: asyncio.TimerHandle | None = app._kimi_resize_handle
            if handle is not None:
                handle.cancel()
            try:
                loop = asyncio.get_running_loop()
                app._kimi_resize_handle = loop.call_later(debounce, on_resize_settled)
            except RuntimeError:
                pass

        app._on_resize = _patched_on_resize

    @staticmethod
    def _cancel_pending_resize(app: Any) -> None:
        """Cancel any pending debounced resize callback without removing the handler."""
        handle: asyncio.TimerHandle | None = getattr(app, "_kimi_resize_handle", None)
        if handle is not None:
            handle.cancel()
            app._kimi_resize_handle = None

    @staticmethod
    def _uninstall_resize_handler(app: Any) -> None:
        """Restore the original ``_on_resize`` and cancel any pending timer."""
        handle: asyncio.TimerHandle | None = getattr(app, "_kimi_resize_handle", None)
        if handle is not None:
            handle.cancel()
            app._kimi_resize_handle = None
        original = getattr(app, "_kimi_resize_original", None)
        if original is not None:
            app._on_resize = original
            app._kimi_resize_original = None

    def _install_focus_repaint(self) -> None:
        """Monkey-patch the shared Vt100 parser to detect focus-in events."""
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
                return
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

    def __enter__(self) -> Any:
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
                pass

        self._status_refresh_task = asyncio.create_task(_refresh(_REFRESH_INTERVAL))
        return self

    def __exit__(self, *_) -> None:
        self._uninstall_focus_repaint()

        import sys as _sys

        if _sys.stdout.isatty():
            _sys.stdout.write("\x1b[?1004l")
            _sys.stdout.flush()

        if self._status_refresh_task is not None and not self._status_refresh_task.done():
            self._status_refresh_task.cancel()
        self._status_refresh_task = None
