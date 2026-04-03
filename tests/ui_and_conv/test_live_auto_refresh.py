from __future__ import annotations

import importlib
from typing import cast

import pytest

from kimi_cli.eventbus import EventBusConsumer
from kimi_cli.eventbus.types import StatusUpdate, TextPart, ToolCallOutput
from kimi_cli.ui.shell.visualize import (
    LIVE_VIEW_REFRESH_INTERVAL,
    LiveView,
    is_significant_for_render,
)
from kimi_cli.utils.aioqueue import QueueShutDown


class _DummyLive:
    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.updated: list[tuple[object, bool]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, renderable, refresh: bool = False) -> None:
        self.updated.append((renderable, refresh))

    def stop(self) -> None:
        return None

    def start(self) -> None:
        return None


class _DummyWire:
    async def receive(self):
        raise QueueShutDown


class _SequenceWire:
    def __init__(self, messages: list[object]) -> None:
        self._messages = list(messages)

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        raise QueueShutDown


def test_live_view_retains_flushed_content_without_console_output() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.append_content(TextPart(text="hello"))
    view.flush_content()

    rendered = view.render_ansi(80)
    assert "hello" in rendered


def test_live_view_refresh_interval_is_one_second() -> None:
    assert LIVE_VIEW_REFRESH_INTERVAL == 1.0


@pytest.mark.asyncio
async def test_live_view_disables_auto_refresh(monkeypatch) -> None:
    created: list[_DummyLive] = []

    def _fake_live(*args, **kwargs):
        live = _DummyLive(*args, **kwargs)
        created.append(live)
        return live

    visualize_module = importlib.import_module("kimi_cli.ui.shell.visualize")
    monkeypatch.setattr(visualize_module, "Live", _fake_live)

    view = LiveView(StatusUpdate(context_usage=0.0))
    await view.visualize_loop(cast(EventBusConsumer, _DummyWire()))

    assert created
    assert created[0].kwargs["auto_refresh"] is False


@pytest.mark.asyncio
async def test_live_view_refreshes_text_stream_updates_immediately(monkeypatch) -> None:
    created: list[_DummyLive] = []

    def _fake_live(*args, **kwargs):
        live = _DummyLive(*args, **kwargs)
        created.append(live)
        return live

    visualize_module = importlib.import_module("kimi_cli.ui.shell.visualize")
    monkeypatch.setattr(visualize_module, "Live", _fake_live)

    view = LiveView(StatusUpdate(context_usage=0.0))
    await view.visualize_loop(cast(EventBusConsumer, _SequenceWire([TextPart(text="hello")])))

    assert created
    assert len(created[0].updated) == 2


def test_is_significant_for_render_only_defers_shell_output_and_status() -> None:
    assert is_significant_for_render(TextPart(text="hello")) is True
    assert is_significant_for_render(ToolCallOutput(tool_call_id="call-1", text="line\n")) is False
    assert is_significant_for_render(StatusUpdate(context_usage=0.1)) is False
