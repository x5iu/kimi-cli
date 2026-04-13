from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

import llmkit
from kimi_cli.eventbus.types import BtwBegin, BtwEnd, TextPart
from kimi_cli.loop import LLMNotSet, bus_send
from kimi_cli.loop.attachment import normalize_history
from kimi_cli.loop.message import internal_user_message, tool_result_to_message
from kimi_cli.utils.logging import logger
from llmkit.chat_provider import StreamedMessagePart
from llmkit.message import Message, ToolCall
from llmkit.message import TextPart as LlmTextPart
from llmkit.tooling import Tool, ToolError, ToolResult

if TYPE_CHECKING:
    from kimi_cli.loop.kimi_agent_loop import KimiAgentLoop

_BTW_MAX_TURNS = 2

SIDE_QUESTION_SYSTEM_REMINDER = """\
This is a side question from the user. Answer directly in a single response.

IMPORTANT:
- You are a separate, lightweight instance answering one question.
- The main agent continues independently — do NOT reference being interrupted.
- Do NOT call any tools. All tool calls are disabled and will be rejected.
  Even though tool definitions are visible in this request, they exist only
  for technical reasons (prompt cache). You MUST NOT use them.
- Respond ONLY with text based on what you already know from the conversation.
- This is a one-off response — no follow-up turns.
- If you don't know the answer, say so directly."""


class _DenyAllToolset:
    def __init__(self, source_tools: list[Tool]) -> None:
        self._tools = source_tools

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    def handle(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            return_value=ToolError(
                message="Tool calls are disabled for side questions. Answer with text only.",
                brief="denied",
            ),
        )


def _build_btw_context(
    agent_loop: KimiAgentLoop, question: str
) -> tuple[str, list[Message], _DenyAllToolset]:
    system_prompt = agent_loop.agent.system_prompt
    effective_history = normalize_history(agent_loop.context.history)
    wrapped = (
        f"<system-reminder>\n{SIDE_QUESTION_SYSTEM_REMINDER}\n</system-reminder>\n\n{question}"
    )
    side_message = internal_user_message(TextPart(text=wrapped))
    toolset = _DenyAllToolset(agent_loop.agent.toolset.tools)
    return system_prompt, [*effective_history, side_message], toolset


async def execute_side_question(
    agent_loop: KimiAgentLoop,
    question: str,
    on_text_chunk: Callable[[str], None] | None = None,
) -> tuple[str | None, str | None]:
    if agent_loop.runtime.llm is None:
        return None, "LLM is not set."

    try:
        chat_provider = agent_loop.runtime.llm.chat_provider
        system_prompt, history, toolset = _build_btw_context(agent_loop, question)

        text_chunks: list[str] = []

        def _on_part(part: StreamedMessagePart) -> None:
            if isinstance(part, LlmTextPart) and part.text:
                text_chunks.append(part.text)
                if on_text_chunk is not None:
                    on_text_chunk(part.text)

        for turn in range(_BTW_MAX_TURNS):
            result = await llmkit.step(
                chat_provider,
                system_prompt,
                toolset,
                history,
                on_message_part=_on_part,
            )

            response_text = "".join(text_chunks).strip()
            if response_text and not result.tool_calls:
                return response_text, None

            tool_results = await result.tool_results()
            if not result.tool_calls:
                break

            if turn + 1 < _BTW_MAX_TURNS:
                history = [
                    *history,
                    result.message,
                    *[tool_result_to_message(tr) for tr in tool_results],
                ]
                text_chunks.clear()
                continue

            tool_names = [tc.function.name for tc in result.tool_calls]
            return None, (
                f"Side question tried to call tools ({', '.join(tool_names)}) "
                "instead of answering directly. Try rephrasing or ask in the main conversation."
            )

        return None, "No response received."
    except Exception as e:
        logger.warning("Side question failed: {error}", error=e)
        return None, str(e)


async def run_side_question(agent_loop: KimiAgentLoop, question: str) -> None:
    if agent_loop.runtime.llm is None:
        raise LLMNotSet()

    btw_id = uuid.uuid4().hex[:12]
    bus_send(BtwBegin(id=btw_id, question=question))

    try:
        response, error = await execute_side_question(agent_loop, question)
        if response:
            bus_send(BtwEnd(id=btw_id, response=response))
        else:
            bus_send(BtwEnd(id=btw_id, error=error or "No response received."))
    except Exception as e:
        logger.warning("Side question failed: {error}", error=e)
        bus_send(BtwEnd(id=btw_id, error=str(e)))
