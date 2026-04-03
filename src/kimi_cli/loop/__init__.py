from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kimi_cli.eventbus import EventBus
from kimi_cli.eventbus.log import EventLog
from kimi_cli.eventbus.types import BusMessage, ContentPart
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.logging import logger

if TYPE_CHECKING:
    from kimi_cli.llm import LLM, ModelCapability
    from kimi_cli.utils.slashcmd import SlashCommand


class LLMNotSet(Exception):
    """Raised when the LLM is not set."""

    def __init__(self) -> None:
        super().__init__("LLM not set")


class LLMNotSupported(Exception):
    """Raised when the LLM does not have required capabilities."""

    def __init__(self, llm: LLM, capabilities: list[ModelCapability]):
        self.llm = llm
        self.capabilities = capabilities
        capabilities_str = "capability" if len(capabilities) == 1 else "capabilities"
        super().__init__(
            f"LLM model '{llm.model_name}' does not support required {capabilities_str}: "
            f"{', '.join(capabilities)}."
        )


class MaxStepsReached(Exception):
    """Raised when the maximum number of steps is reached."""

    n_steps: int
    """The number of steps that have been taken."""

    def __init__(self, n_steps: int):
        super().__init__(f"Max number of steps reached: {n_steps}")
        self.n_steps = n_steps


def format_token_count(n: int) -> str:
    """Format token count as compact string, e.g. 28.5k, 128k, 1.2m."""
    suffix = ""
    if n >= 1_000_000:
        value = n / 1_000_000
        suffix = "m"
    elif n >= 1_000:
        value = n / 1_000
        suffix = "k"
    else:
        return str(n)

    # Keep one decimal when needed, but drop trailing ".0".
    compact = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{compact}{suffix}"


def format_context_status(
    context_usage: float,
    context_tokens: int = 0,
    max_context_tokens: int = 0,
) -> str:
    """Format context status string for display in status bar."""
    bounded = max(0.0, min(context_usage, 1.0))
    if max_context_tokens > 0:
        used = format_token_count(context_tokens)
        total = format_token_count(max_context_tokens)
        return f"context: {bounded:.1%} ({used}/{total})"
    return f"context: {bounded:.1%}"


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    context_usage: float
    """The usage of the context, in percentage."""
    yolo_enabled: bool = False
    """Whether YOLO (auto-approve) mode is enabled."""
    context_tokens: int = 0
    """The number of tokens currently in the context."""
    max_context_tokens: int = 0
    """The maximum number of tokens the context can hold."""


@runtime_checkable
class AgentLoop(Protocol):
    @property
    def name(self) -> str:
        """The name of the soul."""
        ...

    @property
    def model_name(self) -> str:
        """The name of the LLM model used by the soul. Empty string if LLM is not set."""
        ...

    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        """The capabilities of the LLM model used by the soul. None if LLM is not set."""
        ...

    @property
    def thinking(self) -> bool | None:
        """
        Whether thinking mode is currently enabled.
        None if LLM is not set or thinking mode is not set explicitly.
        """
        ...

    @property
    def status(self) -> StatusSnapshot:
        """The current status of the soul. The returned value is immutable."""
        ...

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        """List of available slash commands supported by the soul."""
        ...

    async def run(self, user_input: str | list[ContentPart]):
        """
        Run the agent with the given user input until the max steps or no more tool calls.

        Args:
            user_input (str | list[ContentPart]): The user input to the agent.
                Can be a slash command call or natural language input.

        Raises:
            LLMNotSet: When the LLM is not set.
            LLMNotSupported: When the LLM does not have required capabilities.
            ChatProviderError: When the LLM provider returns an error.
            MaxStepsReached: When the maximum number of steps is reached.
            asyncio.CancelledError: When the run is cancelled by user.
        """
        ...


type UILoopFn = Callable[[EventBus], Coroutine[Any, Any, None]]
"""A long-running async function to visualize the agent behavior."""


class RunCancelled(Exception):
    """The run was cancelled by the cancel event."""


async def run_agent_loop(
    soul: AgentLoop,
    user_input: str | list[ContentPart],
    ui_loop_fn: UILoopFn,
    cancel_event: asyncio.Event,
    event_log: EventLog | None = None,
) -> None:
    """
    Run the soul with the given user input, connecting it to the UI loop with a `EventBus`.

    `cancel_event` is a outside handle that can be used to cancel the run. When the
    event is set, the run will be gracefully stopped and a `RunCancelled` will be raised.

    Raises:
        LLMNotSet: When the LLM is not set.
        LLMNotSupported: When the LLM does not have required capabilities.
        ChatProviderError: When the LLM provider returns an error.
        MaxStepsReached: When the maximum number of steps is reached.
        RunCancelled: When the run is cancelled by the cancel event.
    """
    wire = EventBus(file_backend=event_log)
    bus_token = _current_event_bus.set(wire)

    logger.debug("Starting UI loop with function: {ui_loop_fn}", ui_loop_fn=ui_loop_fn)
    ui_task = asyncio.create_task(ui_loop_fn(wire))

    logger.debug("Starting soul run")
    loop_task = asyncio.create_task(soul.run(user_input))

    cancel_event_task = asyncio.create_task(cancel_event.wait())
    await asyncio.wait(
        [loop_task, cancel_event_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    try:
        if cancel_event.is_set():
            logger.debug("Cancelling the run task")
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                raise RunCancelled from None
        else:
            assert loop_task.done()  # either stop event is set or the run task is done
            cancel_event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_event_task
            loop_task.result()  # this will raise if any exception was raised in the run task
    finally:
        logger.debug("Shutting down the UI loop")
        # shutting down the wire should break the UI loop
        wire.shutdown()
        await wire.join()
        try:
            await asyncio.wait_for(ui_task, timeout=0.5)
        except QueueShutDown:
            logger.debug("UI loop shut down")
            pass
        except TimeoutError:
            logger.warning("UI loop timed out")
        finally:
            _current_event_bus.reset(bus_token)


_current_event_bus = ContextVar[EventBus | None]("current_wire", default=None)


def get_event_bus_or_none() -> EventBus | None:
    """
    Get the current wire or None.
    Expect to be not None when called from anywhere in the agent loop.
    """
    return _current_event_bus.get()


def bus_send(msg: BusMessage) -> None:
    """
    Send a wire message to the current wire.
    Take this as `print` and `input` for souls.
    Souls should always use this function to send wire messages.
    """
    wire = get_event_bus_or_none()
    assert wire is not None, "EventBus is expected to be set when soul is running"
    wire.producer_side.send(msg)
