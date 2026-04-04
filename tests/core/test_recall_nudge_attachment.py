from __future__ import annotations

import json
from unittest.mock import MagicMock

from llmkit.message import Message, TextPart, ToolCall

from kimi_cli.loop.attachments.recall_nudge import (
    RecallNudgeAfterCompactionProvider,
)


def _make_agent_loop_mock(
    *, compaction_generation: int = 0
) -> MagicMock:
    mock = MagicMock()
    mock._compaction_generation = compaction_generation
    return mock


def _user_msg(text: str = "hello") -> Message:
    return Message(
        role="user", content=[TextPart(text=text)]
    )


def _compaction_summary_msg() -> Message:
    """First message after compaction."""
    return Message(
        role="user",
        content=[
            TextPart(
                text=(
                    "<system>Previous context has been "
                    "compacted. Summary of prior work ..."
                )
            )
        ],
        name="_kimi_internal",
    )


def _assistant_msg(text: str = "ok") -> Message:
    return Message(
        role="assistant", content=[TextPart(text=text)]
    )


def _tool_call_msg(
    tool_name: str,
    *,
    call_id: str = "c1",
    arguments: str = "{}",
) -> Message:
    return Message(
        role="assistant",
        content=[],
        tool_calls=[
            ToolCall(
                id=call_id,
                function=ToolCall.FunctionBody(
                    name=tool_name, arguments=arguments
                ),
            )
        ],
    )


def _tool_result(
    call_id: str = "c1", text: str = "ok"
) -> Message:
    return Message(
        role="tool",
        content=[TextPart(text=text)],
        tool_call_id=call_id,
    )


def _exploration_steps(
    count: int, *, start: int = 1
) -> list[Message]:
    """Build N Shell exploration steps."""
    msgs: list[Message] = []
    for i in range(start, start + count):
        msgs.append(
            _tool_call_msg(
                "Shell",
                call_id=f"c{i}",
                arguments=json.dumps(
                    {"command": f"ls -la dir{i}"}
                ),
            )
        )
        msgs.append(_tool_result(f"c{i}"))
    return msgs


def _recall_step(*, call_id: str = "recall1") -> Message:
    """RecallCompactedContext tool call."""
    return _tool_call_msg(
        "RecallCompactedContext",
        call_id=call_id,
        arguments=json.dumps({"keywords": "test"}),
    )


class TestRecallNudgeAfterCompactionProvider:
    async def test_no_injection_without_compaction(
        self,
    ) -> None:
        """No nudge when compaction_generation == 0."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=3
        )
        history: list[Message] = [_user_msg()]
        history.extend(_exploration_steps(5))
        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=0)
        )
        assert result == []

    async def test_no_injection_within_min_steps(
        self,
    ) -> None:
        """No nudge before min_steps assistant messages."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=10
        )
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(9))
        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )
        assert result == []

    async def test_injection_at_min_steps_with_exploration(
        self,
    ) -> None:
        """Nudge fires at min_steps with exploration tools."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=10
        )
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(10))
        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )
        assert len(result) == 1
        assert result[0].is_hint is True
        assert "RecallCompactedContext" in result[0].content

    async def test_cooldown_15_steps(self) -> None:
        """After firing, must wait 15 more steps."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=10, cooldown=15, max_fires=2
        )
        mock = _make_agent_loop_mock(compaction_generation=1)
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(10))

        # First fire at step 10
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # Add 14 more steps (total 24) — within cooldown
        history.extend(_exploration_steps(14, start=11))
        r2 = await provider.get_attachments(history, mock)
        assert r2 == []

        # Add 1 more step (total 25 = 10 + 15) — fires
        history.extend(_exploration_steps(1, start=25))
        r3 = await provider.get_attachments(history, mock)
        assert len(r3) == 1

    async def test_max_2_fires_per_compaction(self) -> None:
        """At most 2 nudges per compaction event."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=3, cooldown=3, max_fires=2
        )
        mock = _make_agent_loop_mock(compaction_generation=1)
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(3))

        # Fire 1
        r1 = await provider.get_attachments(history, mock)
        assert len(r1) == 1

        # Fire 2
        history.extend(_exploration_steps(3, start=4))
        r2 = await provider.get_attachments(history, mock)
        assert len(r2) == 1

        # No fire 3
        history.extend(_exploration_steps(3, start=7))
        r3 = await provider.get_attachments(history, mock)
        assert r3 == []

    async def test_new_compaction_resets(self) -> None:
        """A new compaction event resets all counters."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=3, cooldown=3, max_fires=2
        )
        mock1 = _make_agent_loop_mock(compaction_generation=1)
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(3))

        # Fire 1 at gen=1
        r1 = await provider.get_attachments(history, mock1)
        assert len(r1) == 1

        # Fire 2 at gen=1
        history.extend(_exploration_steps(3, start=4))
        r2 = await provider.get_attachments(history, mock1)
        assert len(r2) == 1

        # Exhausted at gen=1
        history.extend(_exploration_steps(3, start=7))
        r3 = await provider.get_attachments(history, mock1)
        assert r3 == []

        # New compaction (gen=2) → reset
        mock2 = _make_agent_loop_mock(compaction_generation=2)
        new_history: list[Message] = [
            _compaction_summary_msg()
        ]
        new_history.extend(_exploration_steps(3))
        r4 = await provider.get_attachments(
            new_history, mock2
        )
        assert len(r4) == 1

    async def test_recall_usage_suppresses(self) -> None:
        """If RecallCompactedContext was used, suppress."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=3
        )
        history: list[Message] = [_compaction_summary_msg()]
        history.extend(_exploration_steps(2))
        history.append(_recall_step(call_id="r1"))
        history.append(_tool_result("r1"))
        history.extend(_exploration_steps(5, start=3))

        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )
        assert result == []

    async def test_no_exploration_no_injection(
        self,
    ) -> None:
        """No nudge if agent is not in exploration mode."""
        provider = RecallNudgeAfterCompactionProvider(
            min_steps=3
        )
        history: list[Message] = [_compaction_summary_msg()]
        # Only text-only assistant messages (no tool calls)
        for _ in range(5):
            history.append(_assistant_msg("thinking..."))
        result = await provider.get_attachments(
            history, _make_agent_loop_mock(compaction_generation=1)
        )
        assert result == []
