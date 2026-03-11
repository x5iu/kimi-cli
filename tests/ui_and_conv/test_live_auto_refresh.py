from __future__ import annotations

import importlib

import pytest

from kimi_cli.ui.shell.visualize import LiveView
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire.types import StatusUpdate, TextPart


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


def test_live_view_retains_flushed_content_without_console_output() -> None:
    view = LiveView(StatusUpdate(context_usage=0.0), flush_to_console=False)
    view.append_content(TextPart(text="hello"))
    view.flush_content()

    rendered = view.render_ansi(80)
    assert "hello" in rendered


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
    await view.visualize_loop(_DummyWire())

    assert created
    assert created[0].kwargs["auto_refresh"] is False
