from __future__ import annotations

import asyncio

from rich.live import Live

from kimi_cli.eventbus import EventBusConsumer
from kimi_cli.eventbus.types import StatusUpdate
from kimi_cli.ui.shell.visualize._input_router import InputAction, classify_input
from kimi_cli.ui.shell.visualize._live_view import (
    LIVE_VIEW_REFRESH_INTERVAL,
    MAX_ACTIVE_TURN_CONTENT_CHARS,
    MAX_ACTIVE_TURN_FLUSHED_BLOCKS,
    MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS,
    LiveView,
    is_significant_for_render,
    recent_output_notice_text,
    render_user_prompt_block,
)


async def visualize(
    wire: EventBusConsumer,
    *,
    initial_status: StatusUpdate,
    cancel_event: asyncio.Event | None = None,
    live_view: LiveView | None = None,
):
    view = live_view or LiveView(initial_status, cancel_event)
    await view.visualize_loop(wire)


TurnVisualizer = LiveView

__all__ = [
    "InputAction",
    "Live",
    "LIVE_VIEW_REFRESH_INTERVAL",
    "LiveView",
    "MAX_ACTIVE_TURN_CONTENT_CHARS",
    "MAX_ACTIVE_TURN_FLUSHED_BLOCKS",
    "MAX_ACTIVE_TURN_PENDING_INPUT_BLOCKS",
    "TurnVisualizer",
    "classify_input",
    "is_significant_for_render",
    "recent_output_notice_text",
    "render_user_prompt_block",
    "visualize",
]
