from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from llmkit.message import ContentPart, Message
from llmkit.tooling import ToolError, ToolOk

from kimi_cli.eventbus import EventBus
from kimi_cli.eventbus.log import EventLog
from kimi_cli.eventbus.types import (
    BusMessage,
    FollowUpInput,
    QuestionRequest,
    StatusUpdate,
    StepBegin,
    TextPart,
    ToolResult,
    TurnBegin,
    is_event,
)
from kimi_cli.notifications.llm import is_notification_message
from kimi_cli.ui.shell.console import console
from kimi_cli.ui.shell.visualize import render_user_prompt_block, visualize
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.utils.logging import logger
from kimi_cli.utils.message import message_stringify
from kimi_cli.utils.slashcmd import parse_slash_command_call
from kimi_cli.utils.turns import is_real_user_turn_start_message

MAX_REPLAY_TURNS = 5


@dataclass(slots=True)
class _ReplayTurn:
    user_message: Message
    events: list[BusMessage]
    n_steps: int = 0


async def replay_recent_history(
    history: Sequence[Message],
    *,
    event_log: EventLog | None = None,
) -> None:
    """
    Replay the most recent user-initiated turns from the provided message history or wire file.
    """
    if not history:
        # if the context history is empty,either this is a new session
        # or the context has been cleared
        return

    turns = await _build_replay_turns_from_wire(event_log)
    if not turns:
        start_idx = _find_replay_start(history)
        if start_idx is None:
            return
        turns = _build_replay_turns_from_history(history[start_idx:])
    if not turns:
        return

    for turn in turns:
        wire = EventBus()
        console.print(render_user_prompt_block(message_stringify(turn.user_message)))
        ui_task = asyncio.create_task(
            visualize(wire.ui_side(merge=False), initial_status=StatusUpdate())
        )
        for event in turn.events:
            wire.producer_side.send(event)
            await asyncio.sleep(0)  # yield to UI loop
        wire.shutdown()
        with contextlib.suppress(QueueShutDown):
            await ui_task


async def _build_replay_turns_from_wire(event_log: EventLog | None) -> list[_ReplayTurn]:
    if event_log is None or not event_log.path.exists():
        return []

    size = event_log.path.stat().st_size
    if size > 20 * 1024 * 1024:
        logger.info(
            "EventBus file too large for replay, skipping: {file} ({size} bytes)",
            file=event_log.path,
            size=size,
        )
        return []

    turns: deque[_ReplayTurn] = deque(maxlen=MAX_REPLAY_TURNS)
    pending_questions: dict[str, QuestionRequest] = {}
    _skip_turn_events = False
    try:
        async for record in event_log.iter_records():
            wire_msg = record.to_wire_message()

            if isinstance(wire_msg, TurnBegin):
                _skip_turn_events = False
                if _is_clear_command_input(wire_msg.user_input):
                    turns.clear()
                    _skip_turn_events = True
                    continue
                if _is_undo_command_input(wire_msg.user_input):
                    if turns:
                        turns.pop()
                    _skip_turn_events = True
                    continue
                turns.append(
                    _ReplayTurn(
                        user_message=Message(role="user", content=wire_msg.user_input),
                        events=[],
                    )
                )
                pending_questions.clear()
                continue

            if _skip_turn_events or not turns:
                continue

            current_turn = turns[-1]

            # Collect QuestionRequest and try to pair it with answers from
            # subsequent events so it can be auto-resolved during replay.
            if isinstance(wire_msg, QuestionRequest):
                pending_questions[wire_msg.tool_call_id] = wire_msg
                current_turn.events.append(wire_msg)
                continue

            # AskUserQuestion tool: the ToolResult for the same tool_call_id
            # contains the structured answers as JSON in its output.
            if isinstance(wire_msg, ToolResult) and wire_msg.tool_call_id in pending_questions:
                question = pending_questions.pop(wire_msg.tool_call_id)
                answers = _extract_answers_from_tool_result(wire_msg)
                if answers:
                    object.__setattr__(question, "_replay_answers", answers)
                # ToolResult still needs to go into events for tool-call block rendering.

            # Turn-end question: FollowUpInput carries the user's answer text.
            if isinstance(wire_msg, FollowUpInput) and pending_questions:
                for tcid, question in list(pending_questions.items()):
                    if tcid.startswith("turn-end-"):
                        answers = _extract_answers_from_followup(wire_msg.text, question)
                        if answers:
                            object.__setattr__(question, "_replay_answers", answers)
                        del pending_questions[tcid]
                        break

            if not is_event(wire_msg):
                continue

            if isinstance(wire_msg, StepBegin):
                current_turn.n_steps = wire_msg.n
            current_turn.events.append(wire_msg)
    except Exception:
        logger.exception("Failed to build replay turns from wire file {file}:", file=event_log.path)
        return []
    return list(turns)


def _extract_answers_from_tool_result(result: ToolResult) -> dict[str, str] | None:
    """Try to extract ``{"answers": {...}}`` from an AskUserQuestion ToolResult."""
    rv = result.return_value
    if rv.is_error:
        return None
    output = rv.output
    if isinstance(output, str):
        text = output
    elif isinstance(output, list):
        text = "".join(part.text for part in output if isinstance(part, TextPart))
    else:
        return None
    try:
        data = json.loads(text)
        answers = data.get("answers")
        if isinstance(answers, dict) and answers:
            return answers
    except Exception:
        pass
    return None


def _extract_answers_from_followup(
    text: str,
    request: QuestionRequest,
) -> dict[str, str]:
    """Build an answers dict from FollowUpInput text and the QuestionRequest."""
    answers: dict[str, str] = {}
    parts = text.split("\n")
    for i, q in enumerate(request.questions):
        if i < len(parts):
            answers[q.question] = parts[i]
    return answers


def _is_clear_command_input(user_input: str | list[ContentPart]) -> bool:
    if isinstance(user_input, list):
        text = Message(role="user", content=user_input).extract_text(" ").strip()
    else:
        text = str(user_input).strip()
    call = parse_slash_command_call(text)
    if call is None:
        return False
    return call.name in {"clear", "reset"}


def _is_undo_command_input(user_input: str | list[ContentPart]) -> bool:
    if isinstance(user_input, list):
        text = Message(role="user", content=user_input).extract_text(" ").strip()
    else:
        text = str(user_input).strip()
    call = parse_slash_command_call(text)
    if call is None:
        return False
    return call.name == "undo"


def _is_user_message(message: Message) -> bool:
    # FIXME: should consider non-text tool call results which are sent as user messages
    if is_notification_message(message):
        return False
    return is_real_user_turn_start_message(message)


def _find_replay_start(history: Sequence[Message]) -> int | None:
    indices = [idx for idx, message in enumerate(history) if _is_user_message(message)]
    if not indices:
        return None
    # only replay last MAX_REPLAY_TURNS messages
    return indices[max(0, len(indices) - MAX_REPLAY_TURNS)]


def _build_replay_turns_from_history(history: Sequence[Message]) -> list[_ReplayTurn]:
    turns: list[_ReplayTurn] = []
    current_turn: _ReplayTurn | None = None
    for message in history:
        if _is_user_message(message):
            # start a new turn
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = _ReplayTurn(user_message=message, events=[])
        elif message.role == "assistant":
            if current_turn is None:
                continue
            current_turn.n_steps += 1
            current_turn.events.append(StepBegin(n=current_turn.n_steps))
            current_turn.events.extend(message.content)
            current_turn.events.extend(message.tool_calls or [])
        elif message.role == "tool":
            if current_turn is None:
                continue
            assert message.tool_call_id is not None
            if any(
                isinstance(part, TextPart) and part.text.startswith("<system>ERROR")
                for part in message.content
            ):
                result = ToolError(message="", output="", brief="")
            else:
                result = ToolOk(output=message.content)
            current_turn.events.append(
                ToolResult(tool_call_id=message.tool_call_id, return_value=result)
            )
    if current_turn is not None:
        turns.append(current_turn)
    return turns
