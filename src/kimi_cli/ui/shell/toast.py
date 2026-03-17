from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

_REFRESH_INTERVAL = 1.0


@dataclass(slots=True)
class _ToastEntry:
    topic: str | None
    """There can be only one toast of each non-None topic in the queue."""

    message: str
    expires_at: float


_toast_queues: dict[Literal["left", "right"], deque[_ToastEntry]] = {
    "left": deque(),
    "right": deque(),
}
"""The queue of toasts to show, including the one currently being shown (the first one)."""


def toast(
    message: str,
    duration: float = 5.0,
    topic: str | None = None,
    immediate: bool = False,
    position: Literal["left", "right"] = "left",
) -> None:
    queue = _toast_queues[position]
    duration = max(duration, _REFRESH_INTERVAL)
    entry = _ToastEntry(topic=topic, message=message, expires_at=time.monotonic() + duration)
    if topic is not None:
        for existing in list(queue):
            if existing.topic == topic:
                queue.remove(existing)
    if immediate:
        queue.clear()
        queue.append(entry)
    else:
        queue.append(entry)


def _prune_toasts(position: Literal["left", "right"]) -> None:
    queue = _toast_queues[position]
    now = time.monotonic()
    while queue and queue[0].expires_at <= now:
        queue.popleft()


def _current_toast(position: Literal["left", "right"] = "left") -> _ToastEntry | None:
    _prune_toasts(position)
    queue = _toast_queues[position]
    if not queue:
        return None
    return queue[0]
