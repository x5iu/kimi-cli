from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from kosong.chat_provider import StreamedMessagePart, ThinkingEffort, TokenUsage
from kosong.message import Message, TextPart, ToolCall
from kosong.tooling import Tool

from kimi_cli.agentspec import DEFAULT_AGENT_FILE
from kimi_cli.llm import LLM
from kimi_cli.soul import run_soul
from kimi_cli.soul.agent import load_agent
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.utils.aioqueue import QueueShutDown
from kimi_cli.wire import Wire


@dataclass(slots=True)
class _RecordedGenerateCall:
    system_prompt: str
    tool_names: list[str]
    tool_descriptions: dict[str, str]
    history: list[Message]


class _SequenceStreamedMessage:
    def __init__(self, parts: Sequence[StreamedMessagePart], *, message_id: str) -> None:
        self._parts = list(parts)
        self._message_id = message_id

    def __aiter__(self) -> AsyncIterator[StreamedMessagePart]:
        return self._to_stream()

    async def _to_stream(self) -> AsyncIterator[StreamedMessagePart]:
        for part in self._parts:
            yield part

    @property
    def id(self) -> str | None:
        return self._message_id

    @property
    def usage(self) -> TokenUsage | None:
        return None


class _RecordingSequenceProvider:
    name = "recording-sequence"

    def __init__(self, sequences: Sequence[Sequence[StreamedMessagePart]]) -> None:
        self._sequences = [list(sequence) for sequence in sequences]
        self._index = 0
        self.calls: list[_RecordedGenerateCall] = []

    @property
    def model_name(self) -> str:
        return "recording-sequence"

    @property
    def thinking_effort(self) -> ThinkingEffort | None:
        return None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> _SequenceStreamedMessage:
        index = min(self._index, len(self._sequences) - 1)
        self._index += 1
        self.calls.append(
            _RecordedGenerateCall(
                system_prompt=system_prompt,
                tool_names=[tool.name for tool in tools],
                tool_descriptions={tool.name: tool.description for tool in tools},
                history=list(history),
            )
        )
        return _SequenceStreamedMessage(self._sequences[index], message_id=f"seq-{index}")

    def with_thinking(self, effort: ThinkingEffort) -> _RecordingSequenceProvider:
        del effort
        return self


async def _drain_ui(wire: Wire) -> None:
    wire_ui = wire.ui_side(merge=True)
    while True:
        try:
            await wire_ui.receive()
        except QueueShutDown:
            return


@pytest.mark.asyncio
async def test_default_agent_exposes_execute_todo_guidance_to_llm_and_runs_flow(
    runtime,
    tmp_path: Path,
) -> None:
    detailed_subagent_summary = (
        "Parser root cause: the parser normalizes quote tokens too early, so a later pass no longer "
        "distinguishes raw and escaped delimiters. This causes nested quoted segments to be "
        "flattened incorrectly and shifts subsequent offsets. The fix should preserve raw token kind "
        "until after structural grouping, then derive normalized text only in the final formatting "
        "stage."
    )
    provider = _RecordingSequenceProvider(
        [
            [
                ToolCall(
                    id="tc-set-todo",
                    function=ToolCall.FunctionBody(
                        name="SetTodoList",
                        arguments=json.dumps(
                            {
                                "todos": [
                                    {
                                        "title": "Inspect parser",
                                        "status": "pending",
                                        "executor": "task",
                                        "subagent_name": "coder",
                                        "done_when": "root cause is summarized",
                                    }
                                ]
                            }
                        ),
                    ),
                )
            ],
            [
                ToolCall(
                    id="tc-exec-todo",
                    function=ToolCall.FunctionBody(
                        name="ExecuteTodo",
                        arguments=json.dumps(
                            {
                                "title": "Inspect parser",
                                "description": "inspect parser",
                                "prompt": "Inspect the parser module and summarize the root cause.",
                            }
                        ),
                    ),
                )
            ],
            [TextPart(text=detailed_subagent_summary)],
            [TextPart(text="All done.")],
        ]
    )
    runtime.llm = LLM(
        chat_provider=provider,
        max_context_size=100_000,
        capabilities=set(),
    )
    agent = await load_agent(DEFAULT_AGENT_FILE, runtime, mcp_configs=[])
    context = Context(file_backend=tmp_path / "history.jsonl")
    soul = KimiSoul(agent, context=context)

    await run_soul(soul, "Please inspect the parser issue.", _drain_ui, asyncio.Event())

    assert len(provider.calls) >= 4
    first_call = provider.calls[0]
    assert "ExecuteTodo" in first_call.tool_names
    assert "prefer `ExecuteTodo` over manually chaining" in first_call.system_prompt
    assert "ExecuteTodo" in first_call.tool_descriptions["SetTodoList"]
    assert "ExecuteTodo" in first_call.tool_descriptions["Task"]

    assert any(
        "Ready to ExecuteTodo: Inspect parser @coder" in message.extract_text("\n")
        for message in provider.calls[1].history
        if message.role == "tool"
    )

    assert runtime.session.state.todos[0].title == "Inspect parser"
    assert runtime.session.state.todos[0].status == "done"
    assert runtime.session.state.todos[0].subagent_name == "coder"

    tool_messages = [message.extract_text("\n") for message in context.history if message.role == "tool"]
    assert any("Todo list updated" in text for text in tool_messages)
    assert any("Ready to ExecuteTodo: Inspect parser @coder" in text for text in tool_messages)
    assert any('Todo "Inspect parser" completed via Task.' in text for text in tool_messages)
    assert any("[Task output]" in text and "Parser root cause" in text for text in tool_messages)
