from __future__ import annotations

import asyncio
import contextlib
import copy

from kimi_cli.eventbus.log import EventLog
from kimi_cli.eventbus.types import BusMessage, ContentPart, ToolCallPart, is_bus_message
from kimi_cli.utils.aioqueue import Queue, QueueShutDown
from kimi_cli.utils.broadcast import BroadcastQueue
from kimi_cli.utils.logging import logger
from llmkit.message import MergeableMixin

BusMessageQueue = BroadcastQueue[BusMessage]


class EventBus:
    """
    A spmc channel for communication between the agent loop and the UI during a run.
    """

    def __init__(self, *, file_backend: EventLog | None = None):
        self._raw_queue = BusMessageQueue()
        self._merged_queue = BusMessageQueue()

        self._producer_side = EventBusProducer(self._raw_queue, self._merged_queue)

        if file_backend is not None:
            # record all complete EventBus messages to the file backend
            self._recorder = _EventBusRecorder(file_backend, self._merged_queue.subscribe())
        else:
            self._recorder = None

    @property
    def producer_side(self) -> EventBusProducer:
        return self._producer_side

    def ui_side(self, *, merge: bool) -> EventBusConsumer:
        """
        Create a UI side of the `EventBus`.

        Args:
            merge: Whether to merge `EventBus` messages as much as possible.
        """
        if merge:
            return EventBusConsumer(self._merged_queue.subscribe())
        else:
            return EventBusConsumer(self._raw_queue.subscribe())

    def shutdown(self) -> None:
        self.producer_side.flush()
        logger.debug("Shutting down event bus")
        self._raw_queue.shutdown()
        self._merged_queue.shutdown()

    async def join(self) -> None:
        if self._recorder is None:
            return
        try:
            await self._recorder.join()
        except Exception:
            logger.exception("EventBus recorder failed to flush:")


class EventBusProducer:
    """
    The producer side of an `EventBus`.
    """

    def __init__(self, raw_queue: BusMessageQueue, merged_queue: BusMessageQueue):
        self._raw_queue = raw_queue
        self._merged_queue = merged_queue
        self._merge_buffer: MergeableMixin | None = None

    def send(self, msg: BusMessage) -> None:
        if not isinstance(msg, ContentPart | ToolCallPart):
            logger.debug("Sending bus message: {msg}", msg=msg)

        # send raw message
        try:
            self._raw_queue.publish_nowait(msg)
        except QueueShutDown:
            logger.info("Failed to send raw bus message, queue is shut down: {msg}", msg=msg)

        # merge and send merged message
        match msg:
            case MergeableMixin():
                if self._merge_buffer is None:
                    self._merge_buffer = copy.deepcopy(msg)
                elif self._merge_buffer.merge_in_place(msg):
                    pass
                else:
                    self.flush()
                    self._merge_buffer = copy.deepcopy(msg)
            case _:
                self.flush()
                self._send_merged(msg)

    def flush(self) -> None:
        buffer = self._merge_buffer
        if buffer is None:
            return
        assert is_bus_message(buffer)
        self._send_merged(buffer)
        self._merge_buffer = None

    def _send_merged(self, msg: BusMessage) -> None:
        try:
            self._merged_queue.publish_nowait(msg)
        except QueueShutDown:
            logger.info("Failed to send merged bus message, queue is shut down: {msg}", msg=msg)


class EventBusConsumer:
    """
    The UI side of a `EventBus`.
    """

    def __init__(self, queue: Queue[BusMessage]):
        self._queue = queue

    async def receive(self) -> BusMessage:
        msg = await self._queue.get()
        if not isinstance(msg, ContentPart | ToolCallPart):
            logger.debug("Receiving bus message: {msg}", msg=msg)
        return msg


class _EventBusRecorder:
    def __init__(self, event_log: EventLog, queue: Queue[BusMessage]) -> None:
        self._event_log = event_log
        self._task = asyncio.create_task(self._consume_loop(queue))

    async def join(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _consume_loop(self, queue: Queue[BusMessage]) -> None:
        while True:
            try:
                msg = await queue.get()
                await self._record(msg)
            except QueueShutDown:
                break

    async def _record(self, msg: BusMessage) -> None:
        await self._event_log.append_message(msg)
