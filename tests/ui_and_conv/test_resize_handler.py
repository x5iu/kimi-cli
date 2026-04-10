"""Tests for the debounced terminal resize handler in PromptTerminalCompatMixin."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kimi_cli.ui.shell.prompt_constants import _RESIZE_DEBOUNCE_SECONDS
from kimi_cli.ui.shell.terminal_compat import PromptTerminalCompatMixin


def _make_fake_app() -> SimpleNamespace:
    """Create a minimal fake Application with ``_on_resize`` and ``is_running``."""
    app = SimpleNamespace(
        _on_resize=MagicMock(name="original__on_resize"),
        is_running=True,
    )
    return app


class TestInstallResizeHandler:
    def test_original_on_resize_still_called(self) -> None:
        app = _make_fake_app()
        original = app._on_resize
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.1)

        # Simulate a resize (needs a running loop for call_later)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.sleep(0))  # ensure loop is usable

            def _trigger():
                app._on_resize()

            loop.run_until_complete(loop.run_in_executor(None, _trigger))
        finally:
            loop.close()

        original.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_fires_after_debounce(self) -> None:
        app = _make_fake_app()
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.05)

        app._on_resize()
        assert callback.call_count == 0

        await asyncio.sleep(0.1)
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_debounce_resets_on_repeated_resize(self) -> None:
        app = _make_fake_app()
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.1)

        app._on_resize()
        await asyncio.sleep(0.05)
        # Second resize before debounce expires resets the timer
        app._on_resize()
        await asyncio.sleep(0.05)
        assert callback.call_count == 0

        await asyncio.sleep(0.1)
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_uninstall_cancels_pending_and_restores(self) -> None:
        app = _make_fake_app()
        original = app._on_resize
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.1)
        app._on_resize()

        PromptTerminalCompatMixin._uninstall_resize_handler(app)

        await asyncio.sleep(0.15)
        assert callback.call_count == 0
        assert app._on_resize is original

    def test_uninstall_is_safe_on_clean_app(self) -> None:
        """Calling uninstall on an app with no handler installed should not raise."""
        app = _make_fake_app()
        PromptTerminalCompatMixin._uninstall_resize_handler(app)


class TestCancelPendingResize:
    """Tests for ``_cancel_pending_resize`` – the fix for the stale-timer lifecycle bug.

    When the idle prompt app is cached and reused across prompt cycles,
    a debounced resize timer from cycle N must not fire during cycle N+1.
    ``_cancel_pending_resize`` is called at the start of each new cycle.
    """

    @pytest.mark.asyncio
    async def test_cancel_prevents_stale_timer_firing(self) -> None:
        """A pending timer from a previous cycle is cancelled so it never fires."""
        app = _make_fake_app()
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.1)
        # Simulate a resize in "cycle N"
        app._on_resize()
        assert app._kimi_resize_handle is not None

        # Start of "cycle N+1": cancel the stale timer
        PromptTerminalCompatMixin._cancel_pending_resize(app)
        assert app._kimi_resize_handle is None

        # Wait longer than the debounce – callback must NOT fire
        await asyncio.sleep(0.15)
        assert callback.call_count == 0

    @pytest.mark.asyncio
    async def test_handler_remains_active_after_cancel(self) -> None:
        """After cancelling a stale timer the resize handler is still functional."""
        app = _make_fake_app()
        callback = MagicMock()

        PromptTerminalCompatMixin._install_resize_handler(app, callback, debounce=0.05)
        app._on_resize()

        # Cancel stale timer (simulates new prompt cycle start)
        PromptTerminalCompatMixin._cancel_pending_resize(app)

        # New resize in the current cycle should still schedule and fire
        app._on_resize()
        await asyncio.sleep(0.1)
        assert callback.call_count == 1

    def test_cancel_is_safe_on_clean_app(self) -> None:
        """Calling cancel on an app with no pending timer should not raise."""
        app = _make_fake_app()
        PromptTerminalCompatMixin._cancel_pending_resize(app)

    def test_cancel_is_safe_on_app_without_handler(self) -> None:
        """Calling cancel on an app that never had a handler installed should not raise."""
        app = SimpleNamespace(_on_resize=MagicMock(), is_running=True)
        PromptTerminalCompatMixin._cancel_pending_resize(app)


class TestResizeDebounceConstant:
    def test_constant_is_positive(self) -> None:
        assert _RESIZE_DEBOUNCE_SECONDS > 0

    def test_constant_is_reasonable(self) -> None:
        assert 0.1 <= _RESIZE_DEBOUNCE_SECONDS <= 1.0
