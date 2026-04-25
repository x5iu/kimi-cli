from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import kimi_cli.prompts as prompts
import llmkit
from kimi_cli.eventbus.types import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    VideoURLPart,
)
from kimi_cli.llm import LLM
from kimi_cli.loop.compaction_archive import stringify_tool_calls
from kimi_cli.loop.message import internal_user_message, system
from kimi_cli.utils.logging import logger
from kimi_cli.utils.metrics import emit_metric
from kimi_cli.utils.turns import is_real_user_turn_start_message
from llmkit.chat_provider import ChatProviderError, TokenUsage
from llmkit.message import Message
from llmkit.tooling.empty import EmptyToolset


def _estimate_text_fragment_tokens(text: str) -> int:
    wide = 0
    for c in text:
        if unicodedata.east_asian_width(c) in ("F", "W", "A"):
            wide += 1
    non_wide = len(text) - wide
    return math.ceil(max(non_wide, 0) / 4) + wide


def estimate_text_tokens(messages: Sequence[Message]) -> int:
    total = 0
    for msg in messages:
        total += 4
        for part in msg.content:
            if isinstance(part, TextPart):
                total += _estimate_text_fragment_tokens(part.text)
            elif isinstance(part, ImageURLPart):
                total += 85
            elif isinstance(part, (AudioURLPart, VideoURLPart)):
                total += 200
        if msg.tool_calls:
            tc_text = stringify_tool_calls(msg.tool_calls)
            total += _estimate_text_fragment_tokens(tc_text)
    return total


class CompactionResult(NamedTuple):
    messages: Sequence[Message]
    usage: TokenUsage | None

    @property
    def estimated_token_count(self) -> int:
        if self.usage is not None and len(self.messages) > 0:
            summary_tokens = self.usage.output
            preserved_tokens = estimate_text_tokens(self.messages[1:])
            return summary_tokens + preserved_tokens

        return estimate_text_tokens(self.messages)


def should_auto_compact(
    token_count: int,
    max_context_size: int,
    *,
    trigger_ratio: float,
    reserved_context_size: int,
) -> bool:
    return (
        token_count >= max_context_size * trigger_ratio
        or token_count + reserved_context_size >= max_context_size
    )


@runtime_checkable
class Compaction(Protocol):
    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult: ...


if TYPE_CHECKING:

    def type_check(simple: SimpleCompaction):
        _: Compaction = simple


def _skip_duplicate_assistant_tool_indices(messages: Sequence[Message]) -> set[int]:
    turn_ids: list[int] = []
    tid = 0
    for msg in messages:
        if is_real_user_turn_start_message(msg):
            tid += 1
        turn_ids.append(tid)
    by_turn: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for i, msg in enumerate(messages):
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        by_turn[turn_ids[i]].append((i, stringify_tool_calls(msg.tool_calls)))
    skip: set[int] = set()
    for pairs in by_turn.values():
        j = 0
        while j < len(pairs):
            sig = pairs[j][1]
            k = j + 1
            while k < len(pairs) and pairs[k][1] == sig:
                k += 1
            run = pairs[j:k]
            if len(run) >= 3:
                for idx in range(1, len(run) - 1):
                    skip.add(run[idx][0])
            j = k
    return skip


class SimpleCompaction:
    def __init__(
        self,
        max_preserved_messages: int = 2,
        *,
        dedupe_tool_payloads: bool = False,
    ) -> None:
        self.max_preserved_messages = max_preserved_messages
        self.dedupe_tool_payloads = dedupe_tool_payloads

    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        compact_message, to_preserve = self.prepare(messages, custom_instruction=custom_instruction)
        if compact_message is None:
            return CompactionResult(messages=to_preserve, usage=None)

        max_output_tokens = max(4000, llm.max_context_size // 5)
        logger.debug(
            "Compacting context (max_output_tokens={max_output})",
            max_output=max_output_tokens,
        )
        result = await llmkit.step(
            chat_provider=llm.chat_provider,
            system_prompt=(
                "You are a specialist in compacting agent conversation "
                "context. Produce structured, information-dense summaries "
                "that preserve all actionable details, file paths, "
                "decisions, and ongoing task state. Follow the exact "
                "output format specified in the user instructions."
            ),
            toolset=EmptyToolset(),
            history=[compact_message],
        )
        if result.usage:
            logger.debug(
                "Compaction used {input} input tokens and {output} output tokens",
                input=result.usage.input,
                output=result.usage.output,
            )

        content: list[ContentPart] = [
            system("Previous context has been compacted. Here is the compaction output:")
        ]
        compacted_msg = result.message

        content.extend(part for part in compacted_msg.content if not isinstance(part, ThinkPart))

        if not any(
            isinstance(p, TextPart) and p.text.strip()  # pyright: ignore[reportUnnecessaryIsInstance]
            for p in compacted_msg.content
            if isinstance(p, TextPart)
        ):
            raise ChatProviderError("Compaction produced empty summary")

        compacted_messages: list[Message] = [internal_user_message(content)]
        compacted_messages.extend(to_preserve)
        return CompactionResult(messages=compacted_messages, usage=result.usage)

    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]

    def prepare(
        self, messages: Sequence[Message], *, custom_instruction: str = ""
    ) -> PrepareResult:
        if not messages or self.max_preserved_messages <= 0:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        preserve_start_index = len(messages)

        n_user = 0
        for index in range(len(messages) - 1, -1, -1):
            if is_real_user_turn_start_message(messages[index]):
                n_user += 1
                if n_user == self.max_preserved_messages:
                    preserve_start_index = index
                    break

        if n_user < self.max_preserved_messages:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        while preserve_start_index > 0:
            prev = messages[preserve_start_index - 1]
            if prev.role == "assistant" and prev.tool_calls:
                preserve_start_index -= 1
            else:
                break

        to_compact = messages[:preserve_start_index]
        to_preserve = messages[preserve_start_index:]

        if not to_compact:
            return self.PrepareResult(compact_message=None, to_preserve=to_preserve)

        skip_tool: set[int] = set()
        if self.dedupe_tool_payloads:
            skip_tool = _skip_duplicate_assistant_tool_indices(to_compact)
            if skip_tool:
                emit_metric("compact.dedupe", skipped_indices=len(skip_tool))

        compact_message = Message(role="user", content=[])
        for i, msg in enumerate(to_compact):
            role_label = msg.role
            if msg.role == "tool" and msg.tool_call_id:
                role_label = f"tool (call_id: {msg.tool_call_id})"
            compact_message.content.append(
                TextPart(text=f"## Message {i + 1}\nRole: {role_label}\nContent:\n")
            )
            compact_message.content.extend(
                part for part in msg.content if not isinstance(part, ThinkPart)
            )
            if msg.tool_calls and i not in skip_tool:
                compact_message.content.append(
                    TextPart(text=f"Tool calls: {stringify_tool_calls(msg.tool_calls)}")
                )
        prompt_text = "\n" + prompts.COMPACT
        if custom_instruction:
            prompt_text += (
                "\n\n**User's Custom Compaction Instruction:**\n"
                "The user has specifically requested the following focus during compaction. "
                "You MUST prioritize this instruction above the default compression priorities:\n"
                f"{custom_instruction}"
            )
        compact_message.content.append(TextPart(text=prompt_text))
        return self.PrepareResult(compact_message=compact_message, to_preserve=to_preserve)
